"""The Keepsake stack: data, identity, web, API and operations in us-east-1 (NFR4, NFR6).

Every cdk-nag (AWS Solutions) finding that is not fixed is suppressed in `infra/nag.py` with a
written reason (NFR8).
"""

from __future__ import annotations

from typing import Any

from aws_cdk import CfnOutput, Stack
from constructs import Construct

from infra.constructs.api import ApiConstruct, ApiSettings
from infra.constructs.auth import AuthConstruct
from infra.constructs.data import DataConstruct
from infra.constructs.ops import OpsConstruct
from infra.constructs.web import WebConstruct
from infra.nag import apply_suppressions


class KeepsakeStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs: Any) -> None:
        super().__init__(scope, construct_id, **kwargs)
        context = self.node.try_get_context
        self.data = DataConstruct(self, "Data")
        self.ops = OpsConstruct(
            self, "Ops", key=self.data.key, alarm_email=context("keepsake:alarmEmail") or ""
        )
        self.auth = AuthConstruct(self, "Auth")
        self.web = WebConstruct(self, "Web", data=self.data)
        client = self.auth.add_web_client(self.web.site_url)
        settings = ApiSettings(
            allowed_origin=context("keepsake:allowedOrigin") or self.web.site_url,
            build_run_cost_usd=str(context("keepsake:buildAndRunCostUsdPerMonth") or "500"),
        )
        self.api = ApiConstruct(
            self,
            "Api",
            data=self.data,
            topic=self.ops.topic,
            user_pool=self.auth.user_pool,
            settings=settings,
        )
        self.auth.grant_invoke(self.api.invoke_method.method_arn)
        self.web.deploy_pages(
            api_url=self.api.api.url,
            cognito_domain=self.auth.domain.base_url(),
            client_id=client.user_pool_client_id,
        )
        self.ops.add_monitoring(
            dict(self.api.functions),
            {"budget": self.api.aliases["budget"], "analyzer": self.api.aliases["analyzer"]},
        )
        CfnOutput(self, "ApiUrl", value=self.api.api.url)
        CfnOutput(self, "SiteUrl", value=self.web.site_url)
        CfnOutput(self, "InvocationLogsBucket", value=self.data.invocation_logs_bucket.bucket_name)
        CfnOutput(self, "UserPoolId", value=self.auth.user_pool.user_pool_id)
        apply_suppressions(self)
