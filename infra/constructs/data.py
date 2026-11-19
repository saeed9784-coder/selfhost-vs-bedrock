"""Data layer: one customer-managed KMS key, S3 buckets and DynamoDB tables (FR2.1, FR2.3, FR4.4,
NFR4).

Every bucket and table that holds Keepsake data is encrypted with the same rotating CMK, blocks
public access and refuses non-TLS requests (TLS 1.2 minimum on S3). Buckets and tables are
retained on stack deletion so data is never lost by accident.
"""

from __future__ import annotations

from aws_cdk import Aws, Duration, RemovalPolicy
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_s3 as s3
from constructs import Construct

TLS_MIN = 1.2


class DataConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str) -> None:
        super().__init__(scope, construct_id)
        self.key = self._key()
        self.access_logs = s3.Bucket(
            self,
            "AccessLogs",
            encryption=s3.BucketEncryption.S3_MANAGED,  # S3 server access logs require SSE-S3
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            minimum_tls_version=TLS_MIN,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(365))],
            removal_policy=RemovalPolicy.RETAIN,
        )
        self.documents_bucket = self._bucket("Documents", "documents/", versioned=True)
        self.reports_bucket = self._bucket("Reports", "reports/", versioned=True)
        self.invocation_logs_bucket = self._bucket(
            "InvocationLogs", "invocation-logs/", versioned=False
        )
        self._allow_bedrock_logging()
        self.documents_table = self._table("DocumentsTable")
        self.cache_table = self._table("CacheTable", ttl_attribute="expires_at")
        for name, key in (
            ("by-doc", "doc_ref"),
            ("by-customer", "customer_id"),
            ("by-agent", "agent_id"),
        ):
            self._index(self.cache_table, name, key)
        self.usage_table = self._table("UsageTable")
        for name, key in (("by-agent", "agent_id"), ("by-customer", "customer_id")):
            self._index(self.usage_table, name, key)
        self.audit_table = self._table("AuditTable")

    def _key(self) -> kms.Key:
        key = kms.Key(
            self,
            "Key",
            alias="alias/keepsake",
            enable_key_rotation=True,
            description="Keepsake customer-managed key for all data at rest",
            removal_policy=RemovalPolicy.RETAIN,
        )
        # CloudWatch Logs (encrypted log groups) and CloudWatch alarms (encrypted SNS topic).
        key.add_to_resource_policy(
            iam.PolicyStatement(
                principals=[iam.ServicePrincipal(f"logs.{Aws.REGION}.amazonaws.com")],
                actions=[
                    "kms:Encrypt*",
                    "kms:Decrypt*",
                    "kms:ReEncrypt*",
                    "kms:GenerateDataKey*",
                    "kms:Describe*",
                ],
                resources=["*"],
                conditions={
                    "ArnLike": {
                        "kms:EncryptionContext:aws:logs:arn": (
                            f"arn:{Aws.PARTITION}:logs:{Aws.REGION}:{Aws.ACCOUNT_ID}:*"
                        )
                    }
                },
            )
        )
        key.add_to_resource_policy(
            iam.PolicyStatement(
                principals=[iam.ServicePrincipal("cloudwatch.amazonaws.com")],
                actions=["kms:Decrypt", "kms:GenerateDataKey*"],
                resources=["*"],
            )
        )
        return key

    def _bucket(self, construct_id: str, log_prefix: str, *, versioned: bool) -> s3.Bucket:
        return s3.Bucket(
            self,
            construct_id,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.key,
            bucket_key_enabled=True,
            versioned=versioned,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            minimum_tls_version=TLS_MIN,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            server_access_logs_bucket=self.access_logs,
            server_access_logs_prefix=log_prefix,
            removal_policy=RemovalPolicy.RETAIN,
        )

    def _allow_bedrock_logging(self) -> None:
        """Let Bedrock model-invocation logging write into the invocation-logs bucket (FR1.1)."""
        source = {
            "StringEquals": {"aws:SourceAccount": Aws.ACCOUNT_ID},
            "ArnLike": {
                "aws:SourceArn": f"arn:{Aws.PARTITION}:bedrock:{Aws.REGION}:" f"{Aws.ACCOUNT_ID}:*"
            },
        }
        bedrock = iam.ServicePrincipal("bedrock.amazonaws.com")
        self.invocation_logs_bucket.add_to_resource_policy(
            iam.PolicyStatement(
                principals=[bedrock],
                actions=["s3:PutObject"],
                resources=[self.invocation_logs_bucket.arn_for_objects("AWSLogs/*")],
                conditions=source,
            )
        )
        self.key.add_to_resource_policy(
            iam.PolicyStatement(
                principals=[bedrock],
                actions=["kms:GenerateDataKey"],
                resources=["*"],
                conditions=source,
            )
        )

    def _table(self, construct_id: str, *, ttl_attribute: str | None = None) -> dynamodb.Table:
        return dynamodb.Table(
            self,
            construct_id,
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.key,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            time_to_live_attribute=ttl_attribute,
            deletion_protection=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

    @staticmethod
    def _index(table: dynamodb.Table, name: str, partition: str) -> None:
        table.add_global_secondary_index(
            index_name=name,
            partition_key=dynamodb.Attribute(name=partition, type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            projection_type=dynamodb.ProjectionType.ALL,
        )
