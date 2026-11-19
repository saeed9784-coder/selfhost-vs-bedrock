"""Full-scale reference for architecture/full-inference-architecture.drawio. Synth only.

What it creates, mapped to the diagram:
* Control plane: an EKS cluster (llm-d / vLLM workloads in deploy/k8s/inference.yaml).
* PREFILL GPU POOL and DECODE GPU POOL: two p5.48xlarge managed node groups that scale apart.
* L2 shared prefix KV tier: FSx for Lustre (the diagram's alternative to a 3FS PoC).
* Amazon S3 model artifacts, versioned; observability via Container Insights add-on.

Deploying it costs thousands of dollars a day (each p5.48xlarge is about $55/hour), so the CDK
app only synthesizes it. EFA networking for fast KV transfer needs a launch template with EFA
interfaces and a cluster placement group; that is noted, not built here.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_eks as eks
from aws_cdk import aws_fsx as fsx
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from cdk_nag import NagPackSuppression, NagSuppressions
from constructs import Construct

EKS_VERSION = "1.33"
GPU_INSTANCE = "p5.48xlarge"
LUSTRE_GIB = 2400
LUSTRE_PORTS = (988, 1018, 1023)


class SelfHostedInferenceFullStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs: object) -> None:
        super().__init__(scope, construct_id, **kwargs)
        vpc = ec2.Vpc(self, "Vpc", max_azs=3, nat_gateways=3)
        vpc.add_flow_log(
            "FlowLogs",
            destination=ec2.FlowLogDestination.to_cloud_watch_logs(
                logs.LogGroup(self, "FlowLogGroup", retention=logs.RetentionDays.ONE_MONTH)
            ),
        )
        private = vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS).subnet_ids

        models = s3.Bucket(
            self,
            "ModelArtifacts",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        cluster_role = iam.Role(
            self,
            "ClusterRole",
            assumed_by=iam.ServicePrincipal("eks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSClusterPolicy")
            ],
        )
        cluster_sg = ec2.SecurityGroup(self, "ClusterSg", vpc=vpc, description="EKS cluster")
        cluster = eks.CfnCluster(
            self,
            "Cluster",
            name="inference",
            version=EKS_VERSION,
            role_arn=cluster_role.role_arn,
            resources_vpc_config=eks.CfnCluster.ResourcesVpcConfigProperty(
                subnet_ids=private,
                security_group_ids=[cluster_sg.security_group_id],
                endpoint_private_access=True,
                endpoint_public_access=False,
            ),
            access_config=eks.CfnCluster.AccessConfigProperty(authentication_mode="API"),
            logging=eks.CfnCluster.LoggingProperty(
                cluster_logging=eks.CfnCluster.ClusterLoggingProperty(
                    enabled_types=[
                        eks.CfnCluster.LoggingTypeConfigProperty(type=t)
                        for t in ("api", "audit", "authenticator")
                    ]
                )
            ),
        )

        node_role = iam.Role(
            self,
            "NodeRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(name)
                for name in (
                    "AmazonEKSWorkerNodePolicy",
                    "AmazonEKS_CNI_Policy",
                    "AmazonEC2ContainerRegistryReadOnly",
                    "AmazonSSMManagedInstanceCore",
                    "CloudWatchAgentServerPolicy",
                )
            ],
        )
        models.grant_read(node_role)

        def nodegroup(name: str, instance: str, minimum: int, maximum: int, gpu: bool) -> None:
            group = eks.CfnNodegroup(
                self,
                f"{name.title()}Nodes",
                cluster_name=cluster.ref,
                nodegroup_name=name,
                node_role=node_role.role_arn,
                subnets=private,
                instance_types=[instance],
                ami_type="AL2023_x86_64_NVIDIA" if gpu else "AL2023_x86_64_STANDARD",
                disk_size=500 if gpu else 50,
                scaling_config=eks.CfnNodegroup.ScalingConfigProperty(
                    min_size=minimum, desired_size=minimum, max_size=maximum
                ),
                labels={"inference.pool": name},
                taints=(
                    [
                        eks.CfnNodegroup.TaintProperty(
                            key="nvidia.com/gpu", value="present", effect="NO_SCHEDULE"
                        )
                    ]
                    if gpu
                    else None
                ),
            )
            group.add_dependency(cluster)

        nodegroup("system", "m7i.2xlarge", 2, 4, gpu=False)
        nodegroup("prefill", GPU_INSTANCE, 1, 4, gpu=True)
        nodegroup("decode", GPU_INSTANCE, 1, 8, gpu=True)

        addon = eks.CfnAddon(
            self,
            "Observability",
            cluster_name=cluster.ref,
            addon_name="amazon-cloudwatch-observability",
        )
        addon.add_dependency(cluster)

        lustre_sg = ec2.SecurityGroup(
            self, "LustreSg", vpc=vpc, description="FSx Lustre L2 KV tier"
        )
        for port in LUSTRE_PORTS:
            lustre_sg.add_ingress_rule(ec2.Peer.ipv4(vpc.vpc_cidr_block), ec2.Port.tcp(port))
        lustre = fsx.CfnFileSystem(
            self,
            "SharedKvTier",
            file_system_type="LUSTRE",
            storage_capacity=LUSTRE_GIB,
            subnet_ids=[private[0]],
            security_group_ids=[lustre_sg.security_group_id],
            lustre_configuration=fsx.CfnFileSystem.LustreConfigurationProperty(
                deployment_type="PERSISTENT_2",
                per_unit_storage_throughput=1000,
                data_compression_type="NONE",
            ),
        )

        CfnOutput(self, "ClusterName", value=cluster.ref)
        CfnOutput(self, "SharedKvTierDns", value=lustre.attr_dns_name)
        CfnOutput(self, "ModelArtifactsBucket", value=models.bucket_name)
        NagSuppressions.add_stack_suppressions(
            self,
            [
                NagPackSuppression(
                    id="AwsSolutions-IAM4", reason="EKS requires these managed policies."
                ),
                NagPackSuppression(
                    id="AwsSolutions-S1", reason="Reference stack; add access logs at deploy."
                ),
            ],
        )
