"""Data-model validation tests (FR2.1, FR3.1, FR4.3)."""

from __future__ import annotations

import pytest

from keepsake.common.errors import ValidationError
from keepsake.common.models import (
    AgentConfig,
    DocumentVersion,
    InvokeRequest,
    Message,
)


def _agent(**overrides: object) -> AgentConfig:
    values: dict[str, object] = {
        "agent_id": "helpdesk",
        "team": "helpdesk",
        "model_id": "deepseek.v3.2",
        "retention_hours": 72,
        "monthly_budget_usd": 50.0,
        "allowed_classifications": frozenset({"policy", "runbook"}),
    }
    values.update(overrides)
    return AgentConfig(**values)  # type: ignore[arg-type]


def test_valid_models_build() -> None:
    agent = _agent()
    version = DocumentVersion(
        document_id="policy-manual",
        version_id="000002",
        owner_team="company",
        classification="policy",
        status="active",
        author="ana@example.com",
        created_at="2026-11-10T12:00:00Z",
        s3_key="documents/policy-manual/000002.json",
        chunk_hashes=("a" * 64,),
    )
    request = InvokeRequest(
        agent_id="helpdesk",
        customer_id="cust-1",
        document_ids=("policy-manual",),
        messages=(Message(role="user", text="How do refunds work?"),),
    )

    assert agent.retention_hours == 72
    assert version.ref == "policy-manual@000002"
    assert request.messages[0].text == "How do refunds work?"


@pytest.mark.parametrize("hours", [0, 169, -1])
def test_retention_outside_one_hour_to_seven_days_is_rejected(hours: int) -> None:
    with pytest.raises(ValidationError, match="retention_hours"):
        _agent(retention_hours=hours)


@pytest.mark.parametrize("hours", [1, 168])
def test_retention_boundaries_are_accepted(hours: int) -> None:
    assert _agent(retention_hours=hours).retention_hours == hours


def test_unknown_classification_is_rejected() -> None:
    with pytest.raises(ValidationError, match="classification"):
        _agent(allowed_classifications=frozenset({"top-secret"}))


def test_invoke_request_rejects_unsafe_ids_and_bad_role_order() -> None:
    user = Message(role="user", text="hi")
    with pytest.raises(ValidationError, match="customer_id"):
        InvokeRequest(agent_id="helpdesk", customer_id="../etc", document_ids=(), messages=(user,))
    with pytest.raises(ValidationError, match="alternate"):
        InvokeRequest(agent_id="helpdesk", customer_id="c1", document_ids=(), messages=(user, user))
