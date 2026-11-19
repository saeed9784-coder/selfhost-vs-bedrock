"""Lambda functions and the REST API (FR2.5, FR3.1, FR4.7, FR5.3, FR6.1, NFR4, NFR6).

- Python 3.13 on arm64 with the AWS Lambda Powertools layer (resolved from its public SSM
  parameter at deploy time). Each function publishes a version and is reached only through its
  `live` alias, so a rollback is one alias update and in-flight requests finish on the version
  they started on (NFR6).
- Least privilege: each function gets only the tables, buckets, parameters and model ARNs it
  uses. Bedrock access is limited to the model ids in `config/model_capabilities.json`.
- `/invoke` uses IAM (SigV4) authorization for agents; every staff route uses the Cognito
  authorizer. Access logs, execution logs, X-Ray tracing and stage throttling are enabled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aws_cdk import Aws, Duration, RemovalPolicy
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_ssm as ssm
from constructs import Construct

from infra.constructs.data import DataConstruct
from infra.constructs.ops import SSM_PREFIX, load_config_file

SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
POWERTOOLS_LAYER_PARAMETER = "/aws/service/powertools/python/arm64/python3.13/latest"
THROTTLE_RATE = 50
THROTTLE_BURST = 100


@dataclass(frozen=True)
class ApiSettings:
    allowed_origin: str
    build_run_cost_usd: str


def bedrock_resources() -> list[str]:
    """Model and inference-profile ARNs for every configured model id (FR3.4)."""
    arns: list[str] = []
    for model_id in load_config_file("model_capabilities.json")["models"]:
        base = model_id.split(".", 1)[1] if model_id.startswith(("us.", "global.")) else model_id
        # Geo inference profiles route within US Regions: base model ARN is region-wildcarded.
        arns.append(f"arn:{Aws.PARTITION}:bedrock:*::foundation-model/{base}")
        if base != model_id:
            arns.append(
                f"arn:{Aws.PARTITION}:bedrock:{Aws.REGION}:{Aws.ACCOUNT_ID}:"
                f"inference-profile/{model_id}"
            )
    return arns


class ApiConstruct(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        data: DataConstruct,
        topic: sns.ITopic,
        user_pool: cognito.IUserPool,
        settings: ApiSettings,
    ) -> None:
        super().__init__(scope, construct_id)
        self._data = data
        self._layer = lambda_.LayerVersion.from_layer_version_arn(
            self,
            "Powertools",
            ssm.StringParameter.value_for_string_parameter(self, POWERTOOLS_LAYER_PARAMETER),
        )
        self._environment = {
            "KEEPSAKE_DOCUMENTS_TABLE": data.documents_table.table_name,
            "KEEPSAKE_CACHE_TABLE": data.cache_table.table_name,
            "KEEPSAKE_USAGE_TABLE": data.usage_table.table_name,
            "KEEPSAKE_AUDIT_TABLE": data.audit_table.table_name,
            "KEEPSAKE_DOCUMENTS_BUCKET": data.documents_bucket.bucket_name,
            "KEEPSAKE_REPORTS_BUCKET": data.reports_bucket.bucket_name,
            "KEEPSAKE_SOURCE_INVOCATION_LOGS_BUCKET": data.invocation_logs_bucket.bucket_name,
            "KEEPSAKE_SOURCE_INVOCATION_LOGS_PREFIX": "AWSLogs/",
            "KEEPSAKE_KMS_KEY_ID": data.key.key_arn,
            "KEEPSAKE_ALERTS_TOPIC_ARN": topic.topic_arn,
            "KEEPSAKE_SSM_PREFIX": SSM_PREFIX,
            "KEEPSAKE_ALLOWED_ORIGIN": settings.allowed_origin,
            "KEEPSAKE_BUILD_RUN_COST_USD": settings.build_run_cost_usd,
            "POWERTOOLS_SERVICE_NAME": "keepsake",
        }
        self.functions: dict[str, lambda_.Function] = {}
        self.aliases: dict[str, lambda_.Alias] = {}
        for name, timeout in (
            ("invoke", 29),
            ("documents", 29),
            ("usage", 29),
            ("admin", 29),
            ("analyzer", 900),
            ("budget", 300),
        ):
            self._function(name, timeout)
        self._grant(topic)
        self.api = self._rest_api(user_pool, settings.allowed_origin)

    # ---- functions ------------------------------------------------------------------------
    def _function(self, name: str, timeout_s: int) -> None:
        log_group = logs.LogGroup(
            self,
            f"{name}-logs",
            retention=logs.RetentionDays.THREE_MONTHS,
            encryption_key=self._data.key,
            removal_policy=RemovalPolicy.RETAIN,
        )
        fn = lambda_.Function(
            self,
            f"{name}-fn",
            runtime=lambda_.Runtime.PYTHON_3_13,
            architecture=lambda_.Architecture.ARM_64,
            handler=f"keepsake.handlers.{name}.handler",
            code=lambda_.Code.from_asset(SRC_DIR, exclude=["**/__pycache__", "**/*.pyc"]),
            layers=[self._layer],
            memory_size=1024 if name == "analyzer" else 512,
            timeout=Duration.seconds(timeout_s),
            environment={
                **self._environment,
                **(
                    {
                        "KEEPSAKE_PHASE0_PRICES_JSON": json.dumps(
                            load_config_file("prices.json"), separators=(",", ":")
                        )
                    }
                    if name == "analyzer"
                    else {}
                ),
            },
            log_group=log_group,
            logging_format=lambda_.LoggingFormat.JSON,
            tracing=lambda_.Tracing.ACTIVE,
            description=f"Keepsake {name}",
        )
        self.functions[name] = fn
        self.aliases[name] = lambda_.Alias(
            self, f"{name}-live", alias_name="live", version=fn.current_version
        )

    def _grant(self, topic: sns.ITopic) -> None:
        d, f = self._data, self.functions
        parameter_arn = (
            f"arn:{Aws.PARTITION}:ssm:{Aws.REGION}:{Aws.ACCOUNT_ID}:" f"parameter{SSM_PREFIX}/*"
        )
        read_config = iam.PolicyStatement(actions=["ssm:GetParameter"], resources=[parameter_arn])
        for name in ("invoke", "documents", "admin", "analyzer", "budget"):
            f[name].add_to_role_policy(read_config)
        # Gateway: read documents, read/write cache, write usage and audit, call models.
        d.documents_table.grant_read_data(f["invoke"])
        d.documents_bucket.grant_read(f["invoke"])
        d.cache_table.grant_read_write_data(f["invoke"])
        d.usage_table.grant_write_data(f["invoke"])
        d.audit_table.grant_write_data(f["invoke"])
        f["invoke"].add_to_role_policy(
            iam.PolicyStatement(actions=["bedrock:InvokeModel"], resources=bedrock_resources())
        )
        # Documents: manage documents, invalidate cache, audit.
        d.documents_table.grant_read_write_data(f["documents"])
        d.documents_bucket.grant_read_write(f["documents"])
        d.cache_table.grant_read_write_data(f["documents"])
        d.audit_table.grant_write_data(f["documents"])
        # Usage dashboard API: read usage only.
        d.usage_table.grant_read_data(f["usage"])
        # Admin: cache delete, erasure, model blocks, audit lookup.
        d.cache_table.grant_read_write_data(f["admin"])
        d.usage_table.grant_read_write_data(f["admin"])
        d.audit_table.grant_read_write_data(f["admin"])
        d.documents_table.grant_read_data(f["admin"])
        d.reports_bucket.grant_put(f["admin"])
        f["admin"].add_to_role_policy(
            iam.PolicyStatement(
                actions=["ssm:PutParameter"],
                resources=[
                    f"arn:{Aws.PARTITION}:ssm:{Aws.REGION}:{Aws.ACCOUNT_ID}:"
                    f"parameter{SSM_PREFIX}/models/blocks"
                ],
            )
        )
        # Phase 0 analyzer and budget job.
        d.invocation_logs_bucket.grant_read(f["analyzer"])
        d.reports_bucket.grant_put(f["analyzer"])
        d.usage_table.grant_read_data(f["budget"])
        topic.grant_publish(f["budget"])
        for fn in f.values():
            d.key.grant_encrypt_decrypt(fn)

    # ---- REST API -------------------------------------------------------------------------
    def _rest_api(self, user_pool: cognito.IUserPool, allowed_origin: str) -> apigw.RestApi:
        access_logs = logs.LogGroup(
            self,
            "ApiAccessLogs",
            retention=logs.RetentionDays.ONE_YEAR,
            encryption_key=self._data.key,
            removal_policy=RemovalPolicy.RETAIN,
        )
        api = apigw.RestApi(
            self,
            "Api",
            rest_api_name="keepsake",
            endpoint_types=[apigw.EndpointType.REGIONAL],
            cloud_watch_role=True,
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=THROTTLE_RATE,
                throttling_burst_limit=THROTTLE_BURST,
                access_log_destination=apigw.LogGroupLogDestination(access_logs),
                access_log_format=apigw.AccessLogFormat.json_with_standard_fields(
                    caller=True,
                    http_method=True,
                    ip=True,
                    protocol=True,
                    request_time=True,
                    resource_path=True,
                    response_length=True,
                    status=True,
                    user=True,
                ),
                logging_level=apigw.MethodLoggingLevel.ERROR,
                tracing_enabled=True,
                metrics_enabled=True,
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=[allowed_origin],
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=["Authorization", "Content-Type"],
            ),
        )
        validator = api.add_request_validator(
            "Validator", validate_request_body=True, validate_request_parameters=True
        )
        authorizer = apigw.CognitoUserPoolsAuthorizer(
            self, "StaffAuthorizer", cognito_user_pools=[user_pool]
        )
        staff = {
            "authorizer": authorizer,
            "authorization_type": apigw.AuthorizationType.COGNITO,
            "request_validator": validator,
        }

        def integration(name: str) -> apigw.LambdaIntegration:
            return apigw.LambdaIntegration(self.aliases[name])

        self.invoke_method = api.root.add_resource("invoke").add_method(
            "POST",
            integration("invoke"),
            authorization_type=apigw.AuthorizationType.IAM,
            request_validator=validator,
        )
        documents = api.root.add_resource("documents")
        for method in ("GET", "POST"):
            documents.add_method(method, integration("documents"), **staff)
        document = documents.add_resource("{id}")
        for method in ("GET", "PUT", "DELETE"):
            document.add_method(method, integration("documents"), **staff)
        document.add_resource("versions").add_method("GET", integration("documents"), **staff)
        document.add_resource("approve").add_method("POST", integration("documents"), **staff)
        api.root.add_resource("usage").add_method("GET", integration("usage"), **staff)
        api.root.add_resource("cache").add_method("DELETE", integration("admin"), **staff)
        api.root.add_resource("erasure").add_resource("{customer}").add_method(
            "POST", integration("admin"), **staff
        )
        api.root.add_resource("models").add_resource("blocks").add_method(
            "PUT", integration("admin"), **staff
        )
        api.root.add_resource("audit").add_resource("{requestId}").add_method(
            "GET", integration("admin"), **staff
        )
        return api
