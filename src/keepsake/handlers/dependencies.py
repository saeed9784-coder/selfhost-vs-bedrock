"""Build every service once per Lambda execution environment (reused across warm invocations).

Configuration comes only from environment variables set by the CDK stack (see
common/settings.py); credentials come from the Lambda execution role. Nothing secret is in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from keepsake.cache.invalidation import make_invalidator
from keepsake.cache.repository import CacheRepository
from keepsake.common.audit import AuditLog
from keepsake.common.aws import client, dynamodb_table
from keepsake.common.config import ConfigProvider
from keepsake.common.settings import Settings
from keepsake.gateway.bedrock_client import BedrockGateway, bedrock_runtime_client
from keepsake.gateway.router import Router
from keepsake.gateway.service import GatewayDependencies, GatewayService
from keepsake.governance.erasure import ErasureService
from keepsake.store.repository import DocumentRepository
from keepsake.store.service import DocumentService
from keepsake.usage.metrics import MetricsEmitter
from keepsake.usage.repository import UsageRepository


@dataclass(frozen=True)
class Container:
    settings: Settings
    config: ConfigProvider
    documents: DocumentService
    cache: CacheRepository
    usage: UsageRepository
    audit: AuditLog
    erasure: ErasureService
    metrics: MetricsEmitter
    s3: Any
    ssm: Any
    sns: Any
    bedrock_client: Any

    def gateway(self) -> GatewayService:
        return GatewayService(
            GatewayDependencies(
                config=self.config,
                router=Router(self.config),
                documents=self.documents,
                cache=self.cache,
                usage=self.usage,
                audit=self.audit,
                bedrock=BedrockGateway(self.bedrock_client),
                metrics=self.metrics,
            )
        )


def build_container(settings: Settings | None = None, *, bedrock_client: Any = None) -> Container:
    settings = settings or Settings.from_env()
    s3, ssm, sns = client("s3"), client("ssm"), client("sns")
    cache = CacheRepository(table=dynamodb_table(settings.cache_table))
    usage = UsageRepository(table=dynamodb_table(settings.usage_table))
    audit = AuditLog(table=dynamodb_table(settings.audit_table))
    documents = DocumentService(
        repository=DocumentRepository(
            s3=s3,
            table=dynamodb_table(settings.documents_table),
            bucket=settings.documents_bucket,
            kms_key_id=settings.kms_key_id,
        ),
        audit=audit,
        invalidate=make_invalidator(cache),
    )
    erasure = ErasureService(
        cache=cache,
        usage=usage,
        audit=audit,
        s3=s3,
        reports_bucket=settings.reports_bucket,
        kms_key_id=settings.kms_key_id,
    )
    return Container(
        settings=settings,
        config=ConfigProvider(ssm=ssm, prefix=settings.ssm_prefix),
        documents=documents,
        cache=cache,
        usage=usage,
        audit=audit,
        erasure=erasure,
        metrics=MetricsEmitter(),
        s3=s3,
        ssm=ssm,
        sns=sns,
        bedrock_client=bedrock_client or bedrock_runtime_client(),
    )


_CONTAINER: Container | None = None


def container() -> Container:
    global _CONTAINER
    if _CONTAINER is None:
        _CONTAINER = build_container()
    return _CONTAINER


def set_container(value: Container | None) -> None:
    """Replace (tests) or clear the cached container."""
    global _CONTAINER
    _CONTAINER = value
