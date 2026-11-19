"""Wire real Keepsake services to moto resources and the Stubber-based fake Bedrock."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from keepsake.cache.invalidation import make_invalidator
from keepsake.cache.repository import CacheRepository
from keepsake.common.audit import AuditLog
from keepsake.common.config import ConfigProvider
from keepsake.gateway.bedrock_client import BedrockGateway
from keepsake.gateway.router import Router
from keepsake.gateway.service import GatewayDependencies, GatewayService
from keepsake.store.repository import DocumentRepository
from keepsake.store.service import DocumentService, NewDocument, StaffIdentity
from keepsake.usage.metrics import MetricsEmitter
from keepsake.usage.repository import UsageRepository
from tests.conftest import (
    AGENT_CONFIGS,
    DOCUMENTS_BUCKET,
    SAMPLE_DOCUMENTS,
    AwsResources,
    FakeBedrock,
)

DOC_ADMIN = StaffIdentity(username="ana@example.com", groups=frozenset({"doc-admin"}))

SEED_DOCUMENTS = (
    ("policy-manual", "company", "policy"),
    ("runbook", "company", "runbook"),
    ("contract", "company", "contract"),
    ("source-snapshot", "reviewer", "source-code"),
)


@dataclass(frozen=True)
class Services:
    gateway: GatewayService
    documents: DocumentService
    config: ConfigProvider
    cache: CacheRepository
    usage: UsageRepository
    audit: AuditLog
    metric_lines: list[str]

    def metrics(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.metric_lines]


def build_services(
    aws: AwsResources,
    bedrock: FakeBedrock,
    clock: Callable[[], Any],
    request_ids: list[str] | None = None,
) -> Services:
    config = ConfigProvider(ssm=aws.ssm, ttl_seconds=0)
    cache = CacheRepository(table=aws.tables["cache"])
    usage = UsageRepository(table=aws.tables["usage"])
    audit = AuditLog(table=aws.tables["audit"], now=clock)
    documents = DocumentService(
        repository=DocumentRepository(
            s3=aws.s3,
            table=aws.tables["documents"],
            bucket=DOCUMENTS_BUCKET,
            kms_key_id=aws.kms_key_id,
        ),
        audit=audit,
        invalidate=make_invalidator(cache),
        now=clock,
    )
    lines: list[str] = []
    ids = iter(request_ids) if request_ids else (f"req-{n}" for n in itertools.count(1))
    gateway = GatewayService(
        GatewayDependencies(
            config=config,
            router=Router(config),
            documents=documents,
            cache=cache,
            usage=usage,
            audit=audit,
            bedrock=BedrockGateway(bedrock.client, sleep=lambda _: None),
            metrics=MetricsEmitter(writer=lines.append),
            now=clock,
            new_request_id=lambda: next(ids),
        )
    )
    return Services(gateway, documents, config, cache, usage, audit, lines)


def seed_documents(services: Services, overrides: dict[str, str] | None = None) -> None:
    texts = {**SAMPLE_DOCUMENTS, **(overrides or {})}
    for document_id, owner, classification in SEED_DOCUMENTS:
        services.documents.create(
            DOC_ADMIN,
            NewDocument(
                document_id=document_id,
                title=document_id.replace("-", " ").title(),
                owner_team=owner,
                classification=classification,
                text=texts[document_id],
            ),
        )


def set_agent_config(aws: AwsResources, agent_id: str, **changes: Any) -> None:
    config = {**AGENT_CONFIGS[agent_id], **changes}
    aws.ssm.put_parameter(
        Name=f"/keepsake/agents/{agent_id}/config",
        Type="String",
        Value=json.dumps(config),
        Overwrite=True,
    )


def large_text(paragraphs: int = 12) -> str:
    """Policy text above the 1,024-token prompt-cache minimum (about 6,000+ characters)."""
    return "\n\n".join(
        f"Policy section {i}. " + "Customers may request refunds within thirty days. " * 10
        for i in range(paragraphs)
    )
