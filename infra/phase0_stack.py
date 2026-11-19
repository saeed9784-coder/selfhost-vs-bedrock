"""Minimal Phase 0 deployment: analyze existing Bedrock invocation logs only.

This stack does not run models, create an inference cache, or deploy the customer-facing Keepsake
prototype. It reads an existing S3 invocation-log bucket and writes aggregate reports to a new,
encrypted S3 bucket. The source bucket is supplied at deploy time or through CDK context.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aws_cdk import Aws, CfnOutput, CfnParameter, Duration, Fn, RemovalPolicy, Stack
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from cdk_nag import NagPackSuppression, NagSuppressions
from constructs import Construct

from infra.constructs.ops import load_config_file

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
PHASE0_COST_CONTEXT = "keepsake:buildAndRunCostUsdPerMonth"
SOURCE_BUCKET_CONTEXT = "keepsake:sourceInvocationLogsBucket"
SOURCE_PREFIX_CONTEXT = "keepsake:sourceInvocationLogsPrefix"
SOURCE_KMS_CONTEXT = "keepsake:sourceInvocationLogsKmsKeyArn"


class Phase0Stack(Stack):
    """Evidence-first Phase 0 analyzer with no model-serving or cache infrastructure."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        source_bucket_name: str | None = None,
        source_prefix: str = "AWSLogs/",
        source_kms_key_arn: str = "",
        build_run_cost_usd: str = "500",
        schedule_enabled: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if not re.fullmatch(r"[A-Za-z0-9/_=-]{1,512}", source_prefix) or ".." in source_prefix:
            raise ValueError(
                "source_prefix must be a simple S3 prefix without traversal or wildcards"
            )

        if source_bucket_name:
            source_bucket = source_bucket_name
        else:
            source_bucket = CfnParameter(
                self,
                "SourceInvocationLogsBucket",
                type="String",
                description="Existing S3 bucket containing Bedrock invocation logs.",
                min_length=3,
                max_length=63,
                allowed_pattern="[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]",
            ).value_as_string

        self.report_key = kms.Key(
            self,
            "ReportKey",
            alias="alias/keepsake-phase0-reports",
            enable_key_rotation=True,
            description="Keepsake Phase 0 aggregate report encryption",
            removal_policy=RemovalPolicy.RETAIN,
        )
        access_logs_bucket = s3.Bucket(
            self,
            "ReportAccessLogs",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            minimum_tls_version=1.2,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            removal_policy=RemovalPolicy.RETAIN,
        )
        NagSuppressions.add_resource_suppressions(
            access_logs_bucket,
            [
                {
                    "id": "AwsSolutions-S1",
                    "reason": "This dedicated S3 access-log destination cannot write access logs to itself.",
                }
            ],
        )
        self.reports_bucket = s3.Bucket(
            self,
            "Reports",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.report_key,
            bucket_key_enabled=True,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            minimum_tls_version=1.2,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            server_access_logs_bucket=access_logs_bucket,
            server_access_logs_prefix="phase0-reports/",
            removal_policy=RemovalPolicy.RETAIN,
        )

        self.analyzer = lambda_.Function(
            self,
            "Analyzer",
            runtime=lambda_.Runtime.PYTHON_3_13,
            architecture=lambda_.Architecture.ARM_64,
            handler="keepsake.handlers.analyzer.handler",
            code=lambda_.Code.from_asset(SRC_DIR, exclude=["**/__pycache__", "**/*.pyc"]),
            memory_size=1024,
            timeout=Duration.seconds(900),
            environment={
                "KEEPSAKE_SOURCE_INVOCATION_LOGS_BUCKET": source_bucket,
                "KEEPSAKE_SOURCE_INVOCATION_LOGS_PREFIX": source_prefix,
                "KEEPSAKE_REPORTS_BUCKET": self.reports_bucket.bucket_name,
                "KEEPSAKE_KMS_KEY_ID": self.report_key.key_arn,
                "KEEPSAKE_BUILD_RUN_COST_USD": str(build_run_cost_usd),
                "KEEPSAKE_PHASE0_PRICES_JSON": json.dumps(
                    load_config_file("prices.json"), separators=(",", ":")
                ),
            },
            description="Keepsake Phase 0 invocation-log analysis",
        )

        bucket_arn = f"arn:{Aws.PARTITION}:s3:::{source_bucket}"
        self.analyzer.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[bucket_arn],
                conditions={"StringLike": {"s3:prefix": [source_prefix, f"{source_prefix}*"]}},
            )
        )
        self.analyzer.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[Fn.join("", [bucket_arn, "/", source_prefix, "*"])],
            )
        )
        if source_kms_key_arn:
            kms.Key.from_key_arn(self, "SourceLogsKey", source_kms_key_arn).grant_decrypt(
                self.analyzer
            )
        self.analyzer.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject"],
                resources=[f"{self.reports_bucket.bucket_arn}/phase0/*"],
            )
        )
        self.analyzer.add_to_role_policy(
            iam.PolicyStatement(
                actions=["kms:GenerateDataKey"],
                resources=[self.report_key.key_arn],
            )
        )

        NagSuppressions.add_stack_suppressions(
            self,
            [
                NagPackSuppression(
                    id="AwsSolutions-IAM4",
                    reason="The Lambda basic execution policy is used only for its CloudWatch log stream.",
                    applies_to=[
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
                    ],
                ),
                NagPackSuppression(
                    id="AwsSolutions-IAM5",
                    reason=(
                        "The analyzer lists only the configured log prefix, reads objects only "
                        "under that prefix, writes only to the report bucket, and receives KMS "
                        "data-key permissions only for the report bucket's CMK."
                    ),
                ),
                NagPackSuppression(
                    id="AwsSolutions-L1",
                    reason="Python 3.13 is the selected managed runtime for the Phase 0 analyzer.",
                ),
            ],
        )

        if schedule_enabled:
            events.Rule(
                self,
                "WeeklyAnalysis",
                schedule=events.Schedule.cron(minute="0", hour="5", week_day="MON"),
                targets=[
                    targets.LambdaFunction(
                        self.analyzer,
                        event=events.RuleTargetInput.from_object({"prefix": source_prefix}),
                    )
                ],
            )

        CfnOutput(self, "AnalyzerFunctionName", value=self.analyzer.function_name)
        CfnOutput(self, "ReportsBucketName", value=self.reports_bucket.bucket_name)
        CfnOutput(self, "SourceLogsBucketName", value=source_bucket)
