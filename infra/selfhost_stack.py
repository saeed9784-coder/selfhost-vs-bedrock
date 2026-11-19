"""Minimal deployable slice of architecture/full-inference-architecture.drawio.

    Viewer -> CloudFront (HTTPS) -> ALB (CloudFront-only, secret header)
           -> GPU instance in a private subnet:
                gateway + cache-aware router (FastAPI, port 8080, serves the chat UI)
                vLLM with prefix caching (L1 KV in GPU memory, port 8000, localhost only)
           -> Amazon Bedrock (same model family) for the live cost comparison

The full diagram (separate prefill/decode pools, an L2 shared KV tier, EKS + llm-d) is in
``infra/selfhost_full_stack.py`` and ``deploy/k8s``; it synthesizes but is not deployed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_assets as s3_assets
from aws_cdk import aws_secretsmanager as secretsmanager
from cdk_nag import NagPackSuppression, NagSuppressions
from constructs import Construct

from selfhost.settings import bedrock_target, load_catalog

ROOT = Path(__file__).resolve().parents[1]
AMI_PARAMETERS = {
    # GPU: Deep Learning Base AMI (NVIDIA driver, Docker, NVIDIA container toolkit, AWS CLI).
    "gpu": "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
    # CPU: plain Ubuntu 24.04; the boot script installs Docker and the AWS CLI.
    "cpu": "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id",
}
ROOT_VOLUME_GB = {"gpu": 200, "cpu": 40}
# Managed prefix list com.amazonaws.global.cloudfront.origin-facing in us-east-1.
CLOUDFRONT_PREFIX_LIST_US_EAST_1 = "pl-3b927c52"
METRICS_NAMESPACE = "SelfHostedInference"
GATEWAY_PORT = 8080
CPU_CONTEXT = 16384  # tokens per llama.cpp slot; room for reasoning plus conversation history
LLAMACPP_PARALLEL = 2  # concurrent sequences (slots); each gets max_model_len of context
PREREQUISITES = """
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt_get() { apt-get -o DPkg::Lock::Timeout=900 "$@"; }  # first boot: unattended-upgrades
apt_get update -y
apt_get install -y python3-venv unzip jq curl
command -v docker >/dev/null || { apt_get install -y docker.io; systemctl enable --now docker; }
# Unattended upgrades restart Docker when it updates, which kills the engine and its prompt cache;
# hold the container runtime (other security updates still apply).
apt-mark hold docker.io containerd runc 2>/dev/null || true
if ! command -v aws >/dev/null; then
  mkdir -p /var/lib/selfhost-boot && cd /var/lib/selfhost-boot
  curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip
  unzip -q -o awscliv2.zip && ./aws/install && rm -rf aws awscliv2.zip && cd /
fi
"""
GATEWAY_PIP = "'fastapi>=0.115,<1' 'uvicorn>=0.30,<1' 'httpx>=0.27,<1' 'boto3>=1.40'"


@dataclass(frozen=True)
class SelfHostConfig:
    # The same model on both sides: MiniMax M2.5 self-hosted (llama.cpp, UD-Q3_K_XL GGUF) and on
    # Bedrock. It needs about 112 GB of RAM, so the node is a 16-vCPU, 128 GiB r-family machine
    # on Spot (r7a first; r7i, r6a and r6i as fallbacks, see catalog.json).
    model_key: str = "minimax-m2.5"
    instance_type: str = "r7a.4xlarge"
    purchase: str = "spot"  # "spot" or "on_demand"
    vllm_image: str = "vllm/vllm-openai:v0.10.1.1"
    llamacpp_image: str = "ghcr.io/ggml-org/llama.cpp:server"
    root_volume_gb: int = 0  # 0 = default for the instance kind
    # Bedrock comparison model: "" = the same model as self-hosted (the point of the demo).
    # A key from catalog.json "bedrock_models" compares against a different model instead.
    bedrock_model: str = ""
    availability_zone: str = ""  # "" = first private subnet; set when an AZ has no capacity
    cloudfront_prefix_list: str = CLOUDFRONT_PREFIX_LIST_US_EAST_1

    @classmethod
    def from_context(cls, node: Construct) -> SelfHostConfig:
        def ctx(key: str, default: str) -> str:
            return str(node.node.try_get_context(f"selfhost:{key}") or default)

        defaults = cls()
        return cls(
            model_key=ctx("model", defaults.model_key),
            instance_type=ctx("instanceType", defaults.instance_type),
            vllm_image=ctx("vllmImage", defaults.vllm_image),
            llamacpp_image=ctx("llamacppImage", defaults.llamacpp_image),
            availability_zone=ctx("availabilityZone", defaults.availability_zone),
            purchase=ctx("purchase", defaults.purchase),
            bedrock_model=str(
                node.node.try_get_context("selfhost:bedrockModel")
                if node.node.try_get_context("selfhost:bedrockModel") is not None
                else defaults.bedrock_model
            ),
            root_volume_gb=int(ctx("rootVolumeGb", str(defaults.root_volume_gb))),
            cloudfront_prefix_list=ctx("cloudfrontPrefixList", defaults.cloudfront_prefix_list),
        )


class SelfHostedInferenceStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *, config: SelfHostConfig, **kwargs: object
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        catalog = load_catalog()
        model = catalog["models"].get(config.model_key)
        if model is None:
            raise ValueError(f"selfhost:model must be one of {sorted(catalog['models'])}")
        if config.instance_type not in catalog["instances"]:
            raise ValueError(f"selfhost:instanceType must be one of {sorted(catalog['instances'])}")
        kind = str(catalog["instances"][config.instance_type].get("kind", "gpu"))
        if kind not in model["throughput_assumption"]:
            raise ValueError(
                f"{config.model_key} cannot run on {kind} instance {config.instance_type}"
            )
        on_cpu = kind == "cpu"
        if on_cpu and not model.get("gguf"):
            raise ValueError(f"{config.model_key} has no GGUF build for CPU instances")
        engine = "llama.cpp" if on_cpu else "vllm"
        info = catalog["instances"][config.instance_type]
        needed = float(model.get("memory_gb", {}).get(kind, 0))
        have = float(info.get("gpu_memory_gib" if kind == "gpu" else "memory_gib", needed))
        if have < needed:
            raise ValueError(
                f"{config.model_key} needs about {needed:.0f} GB of {kind} memory; "
                f"{config.instance_type} has {have:.0f}"
            )

        vpc = self._network()
        if config.availability_zone and config.availability_zone not in vpc.availability_zones:
            raise ValueError(
                f"selfhost:availabilityZone {config.availability_zone!r} is not one of the VPC's "
                f"AZs {vpc.availability_zones}"
            )
        log_bucket = s3.Bucket(
            self,
            "AccessLogs",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            object_ownership=s3.ObjectOwnership.OBJECT_WRITER,
            lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(30))],
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        artifacts = s3.Bucket(
            self,
            "ModelArtifacts",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            server_access_logs_bucket=log_bucket,
            server_access_logs_prefix="model-artifacts/",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        access_token = secretsmanager.Secret(
            self,
            "AccessToken",
            description="Bearer token the chat UI must send to the inference gateway",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                password_length=40, exclude_punctuation=True
            ),
        )
        origin_secret = secretsmanager.Secret(
            self,
            "OriginVerify",
            description="Header value CloudFront adds so the ALB only serves CloudFront traffic",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                password_length=48, exclude_punctuation=True
            ),
        )
        hf_token = secretsmanager.Secret(
            self,
            "HuggingFaceToken",
            description="Hugging Face read token for gated models; replace 'unset' with a token",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"token":"unset"}',  # noqa: S106 - placeholder, not a secret
                generate_string_key="unused",
            ),
        )

        alb_sg = ec2.SecurityGroup(
            self,
            "AlbSg",
            vpc=vpc,
            description="ALB: HTTP from CloudFront only",
            allow_all_outbound=False,
        )
        alb_sg.add_ingress_rule(
            ec2.Peer.prefix_list(config.cloudfront_prefix_list),
            ec2.Port.tcp(80),
            "CloudFront origin-facing",
        )
        node_sg = ec2.SecurityGroup(
            self, "NodeSg", vpc=vpc, description="Inference node: gateway port from the ALB only"
        )
        node_sg.add_ingress_rule(alb_sg, ec2.Port.tcp(GATEWAY_PORT), "ALB to gateway")
        alb_sg.add_egress_rule(node_sg, ec2.Port.tcp(GATEWAY_PORT), "ALB to gateway")

        target = bedrock_target(catalog, config.model_key, config.bedrock_model)
        role = self._node_role(target, artifacts, [access_token, origin_secret, hf_token])
        gateway_asset = s3_assets.Asset(
            self, "GatewayCode", path=str(ROOT / "src" / "selfhost"), exclude=["__pycache__"]
        )
        web_asset = s3_assets.Asset(self, "ChatUi", path=str(ROOT / "web" / "chat"))
        gateway_asset.grant_read(role)
        web_asset.grant_read(role)

        user_data = ec2.UserData.for_linux()
        user_data.add_commands(PREREQUISITES)  # AWS CLI must exist before the asset downloads
        user_data.add_s3_download_command(
            bucket=gateway_asset.bucket,
            bucket_key=gateway_asset.s3_object_key,
            local_file="/var/lib/selfhost-boot/gateway.zip",
        )
        user_data.add_s3_download_command(
            bucket=web_asset.bucket,
            bucket_key=web_asset.s3_object_key,
            local_file="/var/lib/selfhost-boot/web.zip",
        )
        user_data.add_commands(
            bootstrap_script(
                region=self.region,
                model_key=config.model_key,
                hf_model=model["gguf"] if on_cpu else model["hf_model"],
                max_model_len=CPU_CONTEXT if on_cpu else int(model["max_model_len"]),
                hf_gated=not on_cpu and bool(model.get("hf_gated")),
                engine=engine,
                engine_image=config.llamacpp_image if engine == "llama.cpp" else config.vllm_image,
                instance_type=config.instance_type,
                artifacts_bucket=artifacts.bucket_name,
                access_token_arn=access_token.secret_arn,
                origin_secret_arn=origin_secret.secret_arn,
                hf_token_arn=hf_token.secret_arn,
                bedrock_model=config.bedrock_model,
                purchase=config.purchase,
            )
        )

        launch_template = ec2.LaunchTemplate(
            self,
            "NodeTemplate",
            instance_type=ec2.InstanceType(config.instance_type),
            machine_image=ec2.MachineImage.from_ssm_parameter(AMI_PARAMETERS[kind]),
            role=role,
            security_group=node_sg,
            user_data=user_data,
            require_imdsv2=True,
            detailed_monitoring=True,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/sda1",
                    volume=ec2.BlockDeviceVolume.ebs(
                        config.root_volume_gb or int(model.get("disk_gb", ROOT_VOLUME_GB[kind])),
                        encrypted=True,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                        iops=6000,  # model files of ~100 GB load in minutes, not half an hour
                        throughput=500,
                        delete_on_termination=True,
                    ),
                )
            ],
        )
        # One node, but any listed type in either private AZ: EC2 picks one with capacity.
        instance_types = [
            config.instance_type,
            *catalog["instances"][config.instance_type].get("alternatives", []),
        ]
        node_group = autoscaling.AutoScalingGroup(
            self,
            "InferenceNodes",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                availability_zones=[config.availability_zone] if config.availability_zone else None,
            ),
            min_capacity=1,
            max_capacity=1,
            mixed_instances_policy=autoscaling.MixedInstancesPolicy(
                launch_template=launch_template,
                launch_template_overrides=[
                    autoscaling.LaunchTemplateOverrides(instance_type=ec2.InstanceType(t))
                    for t in instance_types
                ],
                instances_distribution=self._distribution(config.purchase),
            ),
            update_policy=autoscaling.UpdatePolicy.replacing_update(),
            capacity_rebalance=config.purchase == "spot",  # replace Spot nodes AWS will reclaim
        )

        alb = elbv2.ApplicationLoadBalancer(
            self,
            "Alb",
            vpc=vpc,
            internet_facing=True,
            security_group=alb_sg,
            idle_timeout=Duration.seconds(300),
            drop_invalid_header_fields=True,
        )
        alb.log_access_logs(log_bucket, prefix="alb")
        listener = alb.add_listener("Http", port=80, open=False)
        listener.add_targets(
            "Gateway",
            port=GATEWAY_PORT,
            protocol=elbv2.ApplicationProtocol.HTTP,
            targets=[node_group],
            health_check=elbv2.HealthCheck(path="/healthz", healthy_threshold_count=2),
            deregistration_delay=Duration.seconds(30),
        )

        distribution = cloudfront.Distribution(
            self,
            "Cdn",
            comment="Self-hosted inference chat",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.HttpOrigin(
                    alb.load_balancer_dns_name,
                    protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
                    read_timeout=Duration.seconds(60),
                    custom_headers={"X-Origin-Verify": origin_secret.secret_value.unsafe_unwrap()},
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                response_headers_policy=cloudfront.ResponseHeadersPolicy.SECURITY_HEADERS,
            ),
            enable_logging=True,
            log_bucket=log_bucket,
            log_file_prefix="cloudfront/",
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
        )

        self._dashboard(node_group, config.model_key)
        CfnOutput(self, "ChatUrl", value=f"https://{distribution.distribution_domain_name}")
        CfnOutput(
            self,
            "GetAccessToken",
            value=f"aws secretsmanager get-secret-value --secret-id {access_token.secret_arn} "
            "--query SecretString --output text",
        )
        CfnOutput(
            self,
            "SetHuggingFaceToken",
            value=f"aws secretsmanager put-secret-value --secret-id {hf_token.secret_arn} "
            '--secret-string \'{"token":"hf_..."}\'',
        )
        CfnOutput(self, "NodeGroupName", value=node_group.auto_scaling_group_name)
        CfnOutput(
            self,
            "StopNodeToSaveMoney",
            value="aws autoscaling update-auto-scaling-group --auto-scaling-group-name "
            f"{node_group.auto_scaling_group_name} --min-size 0 --desired-capacity 0",
        )
        CfnOutput(
            self,
            "StartNode",
            value="aws autoscaling update-auto-scaling-group --auto-scaling-group-name "
            f"{node_group.auto_scaling_group_name} --min-size 1 --desired-capacity 1",
        )
        CfnOutput(self, "ModelArtifactsBucket", value=artifacts.bucket_name)
        self._suppressions()

    @staticmethod
    def _distribution(purchase: str) -> autoscaling.InstancesDistribution:
        if purchase == "spot":
            return autoscaling.InstancesDistribution(
                on_demand_base_capacity=0,
                on_demand_percentage_above_base_capacity=0,
                spot_allocation_strategy=autoscaling.SpotAllocationStrategy.PRICE_CAPACITY_OPTIMIZED,
            )
        if purchase == "on_demand":
            return autoscaling.InstancesDistribution(
                on_demand_allocation_strategy=autoscaling.OnDemandAllocationStrategy.PRIORITIZED,
                on_demand_percentage_above_base_capacity=100,
            )
        raise ValueError(f"selfhost:purchase must be 'spot' or 'on_demand', got {purchase!r}")

    def _network(self) -> ec2.Vpc:
        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="private", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=22
                ),
            ],
        )
        vpc.add_flow_log(
            "FlowLogs",
            destination=ec2.FlowLogDestination.to_cloud_watch_logs(
                logs.LogGroup(
                    self,
                    "FlowLogGroup",
                    retention=logs.RetentionDays.ONE_WEEK,
                    removal_policy=RemovalPolicy.DESTROY,
                )
            ),
            traffic_type=ec2.FlowLogTrafficType.REJECT,
        )
        vpc.add_gateway_endpoint("S3Endpoint", service=ec2.GatewayVpcEndpointAwsService.S3)
        return vpc

    def _node_role(
        self, target: dict, artifacts: s3.Bucket, secrets: list[secretsmanager.Secret]
    ) -> iam.Role:
        role = iam.Role(
            self,
            "GpuNodeRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")
            ],
        )
        artifacts.grant_read_write(role)
        for secret in secrets:
            secret.grant_read(role)
        model_id = target["model_id"]
        resources = [f"arn:{self.partition}:bedrock:*::foundation-model/{target['base_model_id']}"]
        if model_id != target["base_model_id"]:
            resources.append(
                f"arn:{self.partition}:bedrock:{self.region}:{self.account}:inference-profile/{model_id}"
            )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=resources,
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={"StringEquals": {"cloudwatch:namespace": METRICS_NAMESPACE}},
            )
        )
        return role

    def _dashboard(self, node_group: autoscaling.AutoScalingGroup, model_key: str) -> None:
        def metric(name: str, target: str, stat: str = "p50") -> cloudwatch.Metric:
            return cloudwatch.Metric(
                namespace=METRICS_NAMESPACE,
                metric_name=name,
                dimensions_map={"Target": target, "Model": model_key},
                statistic=stat,
                period=Duration.minutes(5),
                label=f"{target} {name} {stat}",
            )

        both = ("selfhost", "bedrock")
        dashboard = cloudwatch.Dashboard(self, "Slo", dashboard_name="self-hosted-inference")
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Time to first token (ms)",
                left=[metric("TimeToFirstToken", t, s) for t in both for s in ("p50", "p90")],
            ),
            cloudwatch.GraphWidget(
                title="Output tokens per second",
                left=[metric("OutputTokensPerSecond", t) for t in both],
            ),
            cloudwatch.GraphWidget(
                title="Cost per million tokens (USD)",
                left=[metric("CostPerMillionTokensUSD", t, "Average") for t in both],
            ),
            cloudwatch.GraphWidget(
                title="Self-hosted input tokens: total vs served from KV cache",
                left=[
                    metric("InputTokens", "selfhost", "Sum"),
                    metric("CachedInputTokens", "selfhost", "Sum"),
                ],
            ),
            cloudwatch.GraphWidget(
                title="Inference node CPU",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/EC2",
                        metric_name="CPUUtilization",
                        dimensions_map={"AutoScalingGroupName": node_group.auto_scaling_group_name},
                    )
                ],
            ),
        )

    def _suppressions(self) -> None:
        NagSuppressions.add_stack_suppressions(
            self,
            [
                NagPackSuppression(
                    id="AwsSolutions-IAM4",
                    reason="AmazonSSMManagedInstanceCore gives Session Manager access to the GPU "
                    "node instead of SSH; CDK helper Lambdas use AWSLambdaBasicExecutionRole.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-IAM5",
                    reason="Wildcards are scoped: objects inside the model bucket and CDK asset "
                    "bucket, Bedrock foundation-model ARNs in any US Region (cross-Region "
                    "inference profile), and PutMetricData limited by namespace condition.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-SMG4",
                    reason="Tokens are rotated by redeploying for this demo; no rotation Lambda.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-EC29",
                    reason="Demo node must be easy to tear down to stop GPU spend.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-ELB2",
                    reason="Access logging is enabled with log_access_logs; nag cannot see it "
                    "through the token-based attribute.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-S1",
                    reason="This is the access-log bucket itself.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-CFR1",
                    reason="No geographic restriction requirement for the demo.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-CFR2",
                    reason="Gateway enforces a bearer token and per-client rate limits; add WAF "
                    "before any production use.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-CFR4",
                    reason="Default CloudFront certificate (no custom domain) only supports "
                    "TLSv1; a custom domain + ACM certificate removes this.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-CFR5",
                    reason="CloudFront to ALB is HTTP inside AWS's network; the ALB accepts only "
                    "the CloudFront prefix list and checks a secret header. A custom domain "
                    "with an ACM certificate on the ALB enables HTTPS here.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-AS3",
                    reason="Single-node demo group; launches and failures show in the CloudWatch "
                    "dashboard and the group's activity history, so no SNS notifications.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-L1",
                    reason="CDK-managed helper Lambdas use the runtime CDK pins.",
                ),
            ],
        )


def bootstrap_script(
    *,
    region: str,
    model_key: str,
    hf_model: str,
    max_model_len: int,
    hf_gated: bool,
    engine: str,
    engine_image: str,
    instance_type: str,
    artifacts_bucket: str,
    access_token_arn: str,
    bedrock_model: str,
    purchase: str,
    origin_secret_arn: str,
    hf_token_arn: str,
) -> str:
    """Installs the gateway and the inference engine as systemd services."""
    inflight = LLAMACPP_PARALLEL * 4 if engine == "llama.cpp" else 64
    replicas = (
        f'[{{"name":"node-0","url":"http://127.0.0.1:8000",'
        f'"max_context":{max_model_len},"max_inflight":{inflight}}}]'
    )
    if engine == "llama.cpp":
        engine_run = (
            f"docker run --name llm -p 127.0.0.1:8000:8000 -v /opt/models/hf:/models \\\n"
            f"  -e LLAMA_CACHE=/models {engine_image} -hf {hf_model} --alias {model_key} \\\n"
            f"  --host 0.0.0.0 --port 8000 -c $(({max_model_len} * SLOTS)) "
            "--parallel $SLOTS --metrics --cache-reuse 256 --jinja"
        )
    else:
        engine_run = (
            f"docker run --name llm --gpus all --ipc=host -p 127.0.0.1:8000:8000 \\\n"
            f'  -v /opt/models/hf:/root/.cache/huggingface -e HF_TOKEN="$HF_TOKEN" \\\n'
            f"  {engine_image} --model {hf_model} --served-model-name {model_key} \\\n"
            f"  --max-model-len {max_model_len} --gpu-memory-utilization 0.90 \\\n"
            "  --enable-prefix-caching --enable-prompt-tokens-details"
        )
    gated = "true" if hf_gated else "false"
    return f"""
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt_get() {{ apt-get -o DPkg::Lock::Timeout=900 "$@"; }}  # first boot: unattended-upgrades
mkdir -p /opt/selfhost/app/selfhost /opt/selfhost/web /opt/models/hf
unzip -o /var/lib/selfhost-boot/gateway.zip -d /opt/selfhost/app/selfhost
unzip -o /var/lib/selfhost-boot/web.zip -d /opt/selfhost/web
python3 -m venv /opt/selfhost/venv
for attempt in 1 2 3; do
  /opt/selfhost/venv/bin/pip install --quiet {GATEWAY_PIP} && break
  [ "$attempt" = 3 ] && exit 1; sleep 20
done

secret() {{ aws secretsmanager get-secret-value --region {region} --secret-id "$1" \\
  --query SecretString --output text; }}
declare -f secret > /opt/selfhost/secret.sh

cat > /opt/selfhost/run-engine.sh <<'EOF'
#!/bin/bash
set -euo pipefail
source /opt/selfhost/secret.sh
HF_TOKEN=$(secret {hf_token_arn} | jq -r .token)
if [ "$HF_TOKEN" = "unset" ]; then HF_TOKEN=""; fi
if [ -z "$HF_TOKEN" ] && [ "{gated}" = "true" ] && [ ! -d /opt/models/hf/hub ]; then
  echo "SELFHOST: {hf_model} is gated on Hugging Face. Set the token with the" \
       "SetHuggingFaceToken stack output; retrying in 30 s." >&2
  exit 1
fi
# Restore the published copy only when it has its symlink manifest. The Hugging Face cache
# links snapshots/ to blobs/, and a copy without the links would double the size on disk.
MODEL_S3=s3://{artifacts_bucket}/models/{model_key}
if aws s3 ls $MODEL_S3.symlinks.tsv >/dev/null 2>&1; then
  aws s3 sync $MODEL_S3/ /opt/models/hf/ --only-show-errors
  aws s3 cp --only-show-errors $MODEL_S3.symlinks.tsv /opt/selfhost/symlinks.tsv
  while IFS=$'\\t' read -r link target; do
    mkdir -p "$(dirname "/opt/models/hf/$link")"
    ln -sfn "$target" "/opt/models/hf/$link"
  done < /opt/selfhost/symlinks.tsv
fi
# Two chat slots need about 14 GiB with the 8B model; 8 GiB machines get one.
MEM_GIB=$(awk '/MemTotal/ {{print int($2 / 1048576)}}' /proc/meminfo)
if [ "$MEM_GIB" -ge 14 ]; then SLOTS={LLAMACPP_PARALLEL}; else SLOTS=1; fi
docker rm -f llm >/dev/null 2>&1 || true
exec {engine_run}
EOF

cat > /opt/selfhost/publish-model.sh <<'EOF'
#!/bin/bash
# Publish the downloaded weights once, so later boots load the pinned copy from S3.
set -euo pipefail
healthy=no
for _ in $(seq 1 360); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then healthy=yes; break; fi
  sleep 10
done
if [ "$healthy" != yes ]; then echo "engine never became healthy; not publishing" >&2; exit 1; fi
MODEL_S3=s3://{artifacts_bucket}/models/{model_key}
if ! aws s3 ls $MODEL_S3.symlinks.tsv >/dev/null 2>&1; then
  # Upload real files once (links skipped), then the link manifest that marks the copy complete.
  cd /opt/models/hf
  find . -type l -printf '%P\\t%l\\n' > /opt/selfhost/symlinks.tsv
  aws s3 sync . $MODEL_S3/ --no-follow-symlinks --only-show-errors
  aws s3 cp --only-show-errors /opt/selfhost/symlinks.tsv $MODEL_S3.symlinks.tsv
fi
EOF

cat > /opt/selfhost/run-gateway.sh <<'EOF'
#!/bin/bash
set -euo pipefail
source /opt/selfhost/secret.sh
export SELFHOST_ACCESS_TOKEN=$(secret {access_token_arn})
export SELFHOST_ORIGIN_SECRET=$(secret {origin_secret_arn})
# The node group may launch any listed type; price the one we actually got.
IMDS=http://169.254.169.254/latest
IMDS_TOKEN=$(curl -sfX PUT $IMDS/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
ACTUAL_TYPE=$(curl -sf -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" $IMDS/meta-data/instance-type)
export SELFHOST_MODEL={model_key} SELFHOST_INSTANCE_TYPE=${{ACTUAL_TYPE:-{instance_type}}}
export SELFHOST_BEDROCK_MODEL={bedrock_model} SELFHOST_PURCHASE={purchase}
export AWS_REGION={region}
export SELFHOST_REPLICAS='{replicas}'
export SELFHOST_METRICS_NAMESPACE={METRICS_NAMESPACE} SELFHOST_WEB_DIR=/opt/selfhost/web
exec /opt/selfhost/venv/bin/uvicorn selfhost.app:app --app-dir /opt/selfhost/app \\
  --host 0.0.0.0 --port {GATEWAY_PORT}
EOF
chmod +x /opt/selfhost/*.sh

cat > /etc/systemd/system/selfhost-engine.service <<'EOF'
[Unit]
Description=Inference engine ({engine})
After=docker.service network-online.target
Requires=docker.service
[Service]
ExecStart=/opt/selfhost/run-engine.sh
ExecStop=/usr/bin/docker stop llm
Restart=always
RestartSec=30
[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/selfhost-publish.service <<'EOF'
[Unit]
Description=Publish model weights to S3 once vLLM is healthy
After=selfhost-engine.service
[Service]
Type=oneshot
ExecStart=/opt/selfhost/publish-model.sh
[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/selfhost-gateway.service <<'EOF'
[Unit]
Description=Inference gateway and chat UI
After=network-online.target
[Service]
ExecStart=/opt/selfhost/run-gateway.sh
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now selfhost-gateway.service selfhost-engine.service
systemctl enable selfhost-publish.service
systemctl start --no-block selfhost-publish.service
"""
