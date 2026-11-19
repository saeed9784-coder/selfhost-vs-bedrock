"""Gateway tests (FR3.1-FR3.9, FR4.1, FR4.2, FR4.5)."""

from __future__ import annotations

import json
import logging

import pytest
from botocore.exceptions import ClientError

from keepsake.common.config import ConfigProvider, default_config_dir
from keepsake.common.errors import AccessDenied, DependencyUnavailable, ModelBlocked
from keepsake.common.models import InvokeRequest, Message, ModelCapability
from keepsake.gateway.assembler import SharedDocument, assemble
from keepsake.gateway.bedrock_client import BedrockGateway
from tests.conftest import CLAUDE, DEEPSEEK, NOVA, AwsResources, FakeBedrock, converse_response
from tests.unit.builders import build_services, large_text, seed_documents, set_agent_config


def _request(
    agent: str,
    docs: tuple[str, ...] = ("policy-manual",),
    customer: str = "cust-1",
    question: str = "How do refunds work?",
) -> InvokeRequest:
    return InvokeRequest(
        agent_id=agent,
        customer_id=customer,
        document_ids=docs,
        messages=(Message(role="user", text=question),),
    )


def test_assembly_order_and_cache_point_only_where_supported(aws: AwsResources) -> None:
    capabilities = ConfigProvider(ssm=aws.ssm).capabilities()
    docs = [
        SharedDocument("runbook", "000001", large_text()),
        SharedDocument("policy-manual", "000002", large_text()),
    ]
    history = (Message("user", "Hi"), Message("assistant", "Hello"))
    request = (Message("user", "Refund?"),)

    def build(model: str):
        return assemble(
            system_prompt="Be helpful.",
            documents=docs,
            history=history,
            messages=request,
            capability=capabilities[model],
        )

    for model in (CLAUDE, NOVA):
        prompt = build(model)
        assert prompt.system[0] == {"text": "Be helpful."}
        assert prompt.system[1]["text"].startswith('<document id="policy-manual"')
        assert prompt.system[2]["text"].startswith('<document id="runbook"')
        assert prompt.system[3] == {"cachePoint": {"type": "default"}}
        assert [m["role"] for m in prompt.messages] == ["user", "assistant", "user"]
    deepseek = build(DEEPSEEK)
    assert all("cachePoint" not in block for block in deepseek.system)
    assert deepseek.document_versions == ("policy-manual@000002", "runbook@000001")


def test_configured_model_is_used_and_a_config_switch_changes_it_without_code(
    aws: AwsResources,
    fake_bedrock: FakeBedrock,
    fixed_now: object,
) -> None:
    services = build_services(aws, fake_bedrock, fixed_now)
    seed_documents(services)
    fake_bedrock.queue()
    fake_bedrock.queue()

    first = services.gateway.invoke(_request("helpdesk"))
    set_agent_config(aws, "helpdesk", model_id=NOVA)
    second = services.gateway.invoke(_request("helpdesk", question="And exchanges?"))

    assert fake_bedrock.model_ids == [DEEPSEEK, NOVA]
    assert (first.model_id, second.model_id) == (DEEPSEEK, NOVA)
    assert first.document_versions == ("policy-manual@000001",)


def test_three_model_families_route_through_converse(
    aws: AwsResources,
    fake_bedrock: FakeBedrock,
    fixed_now: object,
) -> None:
    services = build_services(aws, fake_bedrock, fixed_now)
    seed_documents(services, {"policy-manual": large_text()})
    for _ in range(3):
        fake_bedrock.queue(converse_response(cache_read=900))

    results = [
        services.gateway.invoke(_request(agent)) for agent in ("reviewer", "helpdesk", "atlas")
    ]

    assert fake_bedrock.model_ids == [CLAUDE, DEEPSEEK, NOVA]
    assert "cachePoint" in json.dumps(fake_bedrock.requests[0]["system"])
    assert "cachePoint" not in json.dumps(fake_bedrock.requests[1]["system"])
    assert results[0].cache_outcome == "provider_hit"


def test_shadow_mode_returns_primary_and_logs_both(
    aws: AwsResources,
    fake_bedrock: FakeBedrock,
    fixed_now: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    services = build_services(aws, fake_bedrock, fixed_now)
    seed_documents(services)
    set_agent_config(aws, "helpdesk", shadow_model_id=NOVA)
    fake_bedrock.queue(converse_response("primary answer"))
    fake_bedrock.queue(converse_response("shadow answer", input_tokens=1300))

    with caplog.at_level(logging.INFO, logger="keepsake.gateway.shadow"):
        result = services.gateway.invoke(_request("helpdesk"))

    assert result.answer == "primary answer"
    assert fake_bedrock.model_ids == [DEEPSEEK, NOVA]
    logged = next(r for r in caplog.records if "shadow comparison" in r.getMessage())
    assert "primary answer" in logged.getMessage() and "shadow answer" in logged.getMessage()


def test_shadow_failure_does_not_affect_the_primary_response(
    aws: AwsResources,
    fake_bedrock: FakeBedrock,
    fixed_now: object,
) -> None:
    services = build_services(aws, fake_bedrock, fixed_now)
    seed_documents(services)
    set_agent_config(aws, "helpdesk", shadow_model_id=NOVA)
    fake_bedrock.queue(converse_response("primary answer"))
    fake_bedrock.queue_error("ValidationException")

    assert services.gateway.invoke(_request("helpdesk")).answer == "primary answer"


def test_blocked_model_is_rejected_and_audited(
    aws: AwsResources,
    fake_bedrock: FakeBedrock,
    fixed_now: object,
) -> None:
    services = build_services(aws, fake_bedrock, fixed_now, request_ids=["req-blocked"])
    seed_documents(services)
    blocks = {"groups": {"eu-regulated": {"customers": ["cust-eu"], "blocked_models": [DEEPSEEK]}}}
    aws.ssm.put_parameter(
        Name="/keepsake/models/blocks", Type="String", Value=json.dumps(blocks), Overwrite=True
    )

    with pytest.raises(ModelBlocked):
        services.gateway.invoke(_request("helpdesk", customer="cust-eu"))

    events = services.audit.lookup_request("req-blocked")
    assert [e.event_type for e in events] == ["model.blocked"]
    assert events[0].details["customer_group"] == "eu-regulated"
    assert fake_bedrock.requests == []


def test_access_denial_is_audited_and_no_model_call_is_made(
    aws: AwsResources,
    fake_bedrock: FakeBedrock,
    fixed_now: object,
) -> None:
    services = build_services(aws, fake_bedrock, fixed_now, request_ids=["req-denied"])
    seed_documents(services)

    with pytest.raises(AccessDenied):
        services.gateway.invoke(_request("reviewer", docs=("contract",)))

    assert services.audit.lookup_request("req-denied")[0].event_type == "document.access_denied"
    assert fake_bedrock.requests == []


def test_store_down_still_calls_the_model_and_emits_degraded(
    aws: AwsResources,
    fake_bedrock: FakeBedrock,
    fixed_now: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = build_services(aws, fake_bedrock, fixed_now)
    seed_documents(services)
    fake_bedrock.queue()

    def unavailable(**_: object) -> None:
        raise ClientError({"Error": {"Code": "ServiceUnavailable", "Message": "down"}}, "GetItem")

    monkeypatch.setattr(aws.tables["documents"], "get_item", unavailable)
    result = services.gateway.invoke(_request("helpdesk"))

    assert result.degraded is True
    assert result.document_versions == ()
    assert fake_bedrock.model_ids == [DEEPSEEK]
    assert any(
        m.get("Degraded") == 1 and m.get("reason") == "context-store" for m in services.metrics()
    )


def test_identical_request_is_reused_and_a_model_version_change_misses(
    aws: AwsResources,
    fake_bedrock: FakeBedrock,
    fixed_now: object,
) -> None:
    services = build_services(aws, fake_bedrock, fixed_now)
    seed_documents(services)
    fake_bedrock.queue(converse_response("Refunds take 5 days."))

    first = services.gateway.invoke(_request("helpdesk"))
    second = services.gateway.invoke(_request("helpdesk"))
    other_customer = _request("helpdesk", customer="cust-2")

    assert (first.cache_outcome, second.cache_outcome) == ("miss", "response_reuse")
    assert second.answer == "Refunds take 5 days."
    assert len(fake_bedrock.requests) == 1

    capabilities = json.loads(
        (default_config_dir() / "model_capabilities.json").read_text(encoding="utf-8")
    )
    capabilities["models"][DEEPSEEK]["model_version"] = "deepseek-v3.2-r2"
    aws.ssm.put_parameter(
        Name="/keepsake/models/capabilities", Type="String", Value=json.dumps(capabilities)
    )
    fake_bedrock.queue()
    fake_bedrock.queue()
    assert services.gateway.invoke(_request("helpdesk")).cache_outcome == "miss"
    assert services.gateway.invoke(other_customer).cache_outcome == "miss"  # NFR3 isolation


def test_bedrock_throttling_is_retried_then_surfaces_as_unavailable(
    fake_bedrock: FakeBedrock,
) -> None:
    gateway = BedrockGateway(fake_bedrock.client, max_attempts=2, sleep=lambda _: None)
    capability = ModelCapability(
        model_id=DEEPSEEK,
        family="deepseek",
        model_version="v3.2",
        supports_prompt_cache=False,
        min_cache_tokens=0,
        cache_ttl_seconds=0,
    )
    prompt = assemble(
        system_prompt="",
        documents=(),
        history=(),
        messages=(Message("user", "hi"),),
        capability=capability,
    )
    fake_bedrock.queue_error("ThrottlingException", 429)
    fake_bedrock.queue(converse_response("ok after retry"))
    assert gateway.converse(DEEPSEEK, prompt).text == "ok after retry"

    fake_bedrock.queue_error("ThrottlingException", 429)
    fake_bedrock.queue_error("ThrottlingException", 429)
    with pytest.raises(DependencyUnavailable):
        gateway.converse(DEEPSEEK, prompt)
