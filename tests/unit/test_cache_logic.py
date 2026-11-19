"""Cache logic tests: invalidation on document update and key determinism (FR4.5, FR4.6)."""

from __future__ import annotations

import pytest

from keepsake.cache.invalidation import invalidate_document_version
from keepsake.cache.keys import compute_reuse_key
from keepsake.common.errors import ValidationError
from keepsake.common.models import InvokeRequest, Message
from tests.conftest import AwsResources, FakeBedrock, converse_response
from tests.unit.builders import DOC_ADMIN, build_services, seed_documents

KEY_ARGS = {
    "agent_id": "helpdesk",
    "customer_id": "cust-1",
    "model_id": "deepseek.v3.2",
    "model_version": "v3.2",
    "system_prompt": "Be  helpful.",
    "document_versions": ["runbook@000001", "policy-manual@000002"],
    "messages": [Message("user", "How do   refunds work?")],
}


def test_document_update_invalidates_dependent_entries_and_next_request_misses(
    aws: AwsResources,
    fake_bedrock: FakeBedrock,
    fixed_now: object,
) -> None:
    services = build_services(aws, fake_bedrock, fixed_now)
    seed_documents(services)
    request = InvokeRequest(
        agent_id="helpdesk",
        customer_id="cust-1",
        document_ids=("policy-manual",),
        messages=(Message("user", "How do refunds work?"),),
    )
    fake_bedrock.queue(converse_response("30 days"))
    services.gateway.invoke(request)
    assert len(services.cache.list_for_customer("cust-1")) == 1

    services.documents.update(DOC_ADMIN, "policy-manual", "Refunds within 45 days.")

    assert services.cache.list_for_customer("cust-1") == ()
    fake_bedrock.queue(converse_response("45 days"))
    result = services.gateway.invoke(request)
    assert result.cache_outcome == "miss"
    assert result.document_versions == ("policy-manual@000002",)


def test_reuse_key_is_deterministic_and_changes_with_model_version_or_documents() -> None:
    base = compute_reuse_key(**KEY_ARGS)
    reordered = compute_reuse_key(
        **{
            **KEY_ARGS,
            "document_versions": list(reversed(KEY_ARGS["document_versions"])),
            "system_prompt": "Be helpful.",
            "messages": [Message("user", "How do refunds work?")],
        }
    )

    assert base == reordered
    assert compute_reuse_key(**{**KEY_ARGS, "model_version": "v3.2-r2"}) != base
    assert compute_reuse_key(**{**KEY_ARGS, "document_versions": ["runbook@000002"]}) != base


def test_invalidation_rejects_malformed_document_references(aws: AwsResources) -> None:
    from keepsake.cache.repository import CacheRepository

    with pytest.raises(ValidationError):
        invalidate_document_version(CacheRepository(table=aws.tables["cache"]), "policy-manual")
