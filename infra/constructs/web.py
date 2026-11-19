"""Staff web pages on S3 behind CloudFront (FR2.6, FR5.3, NFR4).

The site bucket is private and KMS-encrypted; CloudFront reads it through Origin Access Control
and serves HTTPS only with the managed security-headers policy. `config.js` is generated at
deploy time from stack outputs (API URL, Cognito domain and client id).
"""

from __future__ import annotations

import json
from pathlib import Path

from aws_cdk import Duration, RemovalPolicy
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from constructs import Construct

from infra.constructs.data import TLS_MIN, DataConstruct

WEB_DIR = str(Path(__file__).resolve().parents[2] / "web")


class WebConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str, *, data: DataConstruct) -> None:
        super().__init__(scope, construct_id)
        self.site_bucket = s3.Bucket(
            self,
            "Site",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=data.key,
            bucket_key_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            minimum_tls_version=TLS_MIN,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            server_access_logs_bucket=data.access_logs,
            server_access_logs_prefix="site/",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        # CloudFront standard logs need ACL-enabled buckets (SSE-S3 only).
        cdn_logs = s3.Bucket(
            self,
            "CdnLogs",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            minimum_tls_version=TLS_MIN,
            object_ownership=s3.ObjectOwnership.OBJECT_WRITER,
            server_access_logs_bucket=data.access_logs,
            server_access_logs_prefix="cdn-logs/",
            lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(365))],
            removal_policy=RemovalPolicy.RETAIN,
        )
        self.distribution = cloudfront.Distribution(
            self,
            "Distribution",
            default_root_object="index.html",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(self.site_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                response_headers_policy=cloudfront.ResponseHeadersPolicy.SECURITY_HEADERS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            http_version=cloudfront.HttpVersion.HTTP2_AND_3,
            enable_logging=True,
            log_bucket=cdn_logs,
            log_file_prefix="cloudfront/",
        )
        self.site_url = f"https://{self.distribution.distribution_domain_name}"

    def deploy_pages(self, *, api_url: str, cognito_domain: str, client_id: str) -> None:
        config = {
            "apiUrl": api_url.rstrip("/"),
            "cognitoDomain": cognito_domain,
            "clientId": client_id,
            "redirectUri": f"{self.site_url}/index.html",
        }
        script = f"window.KEEPSAKE_CONFIG = Object.freeze({json.dumps(config)});\n"
        s3deploy.BucketDeployment(
            self,
            "Pages",
            destination_bucket=self.site_bucket,
            sources=[
                s3deploy.Source.asset(WEB_DIR, exclude=["config.example.js"]),
                s3deploy.Source.data("config.js", script),
            ],
            distribution=self.distribution,
            distribution_paths=["/*"],
        )
