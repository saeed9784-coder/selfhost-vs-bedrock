"""Lambda handler tests via API Gateway events (FR2.5, FR3.1, FR4.7, FR5.4, FR6.1, NFR3)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from keepsake.common.models import UsageRecord
from keepsake.common.settings import Settings
from keepsake.handlers import admin, documents, invoke, usage
from keepsake.handlers.dependencies import Container, build_container, set_container
from tests.conftest import SAMPLE_DOCUMENTS, AwsResources, FakeBedrock, converse_response

HELPDESK_ARN = "arn:aws:sts::123456789012:assumed-role/keepsake-agent-helpdesk/session"
ADMIN = {"email": "ana@example.com", "cognito:groups": "doc-admin"}


@dataclass(frozen=True)
class LambdaContext:
    function_name: str = "keepsake-test"
    memory_limit_in_mb: int = 512
    invoked_function_arn: str = "arn:aws:lambda:us-east-1:123456789012:function:keepsake-test"
    aws_request_id: str = "test-request"


CTX = LambdaContext()


@pytest.fixture
def deps(aws: AwsResources, fake_bedrock: FakeBedrock) -> Iterator[Container]:
    built = build_container(Settings.from_env(), bedrock_client=fake_bedrock.client)
    set_container(built)
    yield built
    set_container(None)


def staff_event(
    method: str,
    resource: str,
    claims: dict[str, str],
    *,
    body: Any = None,
    path: dict[str, str] | None = None,
    query: dict[str, str] | None = None,
) -> dict:
    return {
        "httpMethod": method,
        "resource": resource,
        "pathParameters": path,
        "queryStringParameters": query,
        "body": None if body is None else json.dumps(body),
        "requestContext": {"authorizer": {"claims": claims}},
    }


def invoke_event(body: Any, arn: str = HELPDESK_ARN) -> dict:
    return {
        "httpMethod": "POST",
        "resource": "/invoke",
        "body": body if isinstance(body, str) else json.dumps(body),
        "requestContext": {"identity": {"userArn": arn}},
    }


def _body(response: dict) -> dict:
    return json.loads(response["body"])


def _upload(
    document_id: str, owner: str, classification: str, text: str, claims: dict[str, str] = ADMIN
) -> dict:
    return documents.handler(
        staff_event(
            "POST",
            "/documents",
            claims,
            body={
                "document_id": document_id,
                "title": document_id,
                "owner_team": owner,
                "classification": classification,
                "text": text,
            },
        ),
        CTX,
    )


INVOKE_BODY = {
    "customer_id": "cust-1",
    "document_ids": ["policy-manual"],
    "messages": [{"role": "user", "text": "How do refunds work?"}],
}


def test_invoke_happy_path_returns_the_answer_and_records_document_versions(
    deps: Container,
    fake_bedrock: FakeBedrock,
) -> None:
    _upload("policy-manual", "company", "policy", SAMPLE_DOCUMENTS["policy-manual"])
    fake_bedrock.queue(converse_response("Refunds take 30 days."))

    response = invoke.handler(invoke_event(INVOKE_BODY), CTX)

    body = _body(response)
    assert response["statusCode"] == 200 and body["success"] is True
    assert body["data"]["answer"] == "Refunds take 30 days."
    assert body["data"]["document_versions"] == ["policy-manual@000001"]
    event = deps.audit.lookup_request(body["data"]["request_id"])[0]
    assert event.details["document_versions"] == ["policy-manual@000001"]


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        {"customer_id": "cust-1", "messages": []},
        {"customer_id": "cust 1", "messages": [{"role": "user", "text": "x"}]},
    ],
)
def test_invalid_invoke_body_returns_400(deps: Container, raw: Any) -> None:
    response = invoke.handler(invoke_event(raw), CTX)

    assert response["statusCode"] == 400
    assert _body(response)["success"] is False


def test_unknown_caller_or_impersonation_returns_403(deps: Container) -> None:
    stranger = "arn:aws:sts::123456789012:assumed-role/someone-else/session"
    impersonating = {**INVOKE_BODY, "agent_id": "reviewer"}

    assert invoke.handler(invoke_event(INVOKE_BODY, arn=stranger), CTX)["statusCode"] == 403
    assert invoke.handler(invoke_event(impersonating), CTX)["statusCode"] == 403


def test_document_upload_update_history_retire_through_the_api(deps: Container) -> None:
    created = _upload("runbook", "helpdesk", "runbook", "Step one.\n\nStep two.")
    path = {"id": "runbook"}
    updated = documents.handler(
        staff_event(
            "PUT",
            "/documents/{id}",
            ADMIN,
            path=path,
            body={"text": "Step one.\n\nStep two, revised."},
        ),
        CTX,
    )
    history = documents.handler(
        staff_event("GET", "/documents/{id}/versions", ADMIN, path=path), CTX
    )
    current = documents.handler(staff_event("GET", "/documents/{id}", ADMIN, path=path), CTX)
    retired = documents.handler(staff_event("DELETE", "/documents/{id}", ADMIN, path=path), CTX)

    assert created["statusCode"] == 201 and updated["statusCode"] == 200
    assert [v["version_id"] for v in _body(history)["data"]["versions"]] == ["000001", "000002"]
    assert _body(current)["data"]["text"].endswith("revised.")
    assert _body(retired)["data"]["document"]["status"] == "retired"


def test_staff_cannot_manage_another_teams_documents(deps: Container) -> None:
    _upload("runbook", "helpdesk", "runbook", "Step one.")
    reviewer_lead = {"email": "rae@example.com", "cognito:groups": "team-reviewer"}

    response = documents.handler(
        staff_event(
            "PUT", "/documents/{id}", reviewer_lead, path={"id": "runbook"}, body={"text": "x"}
        ),
        CTX,
    )
    unauthenticated = documents.handler({"httpMethod": "GET", "resource": "/documents"}, CTX)

    assert response["statusCode"] == 403
    assert unauthenticated["statusCode"] == 403


def _usage_record(n: int, team: str) -> UsageRecord:
    return UsageRecord(
        request_id=f"r{n}",
        timestamp="2026-11-05T10:00:00.000Z",
        agent_id=team,
        team=team,
        customer_id="cust-1",
        model_id="m",
        input_tokens=100,
        cache_read_tokens=0,
        cache_write_tokens=0,
        full_price_tokens=100,
        output_tokens=10,
        cost_usd=0.1,
        saving_usd=0.0,
        storage_cost_usd=0.0,
        latency_ms=80,
        cache_outcome="miss",
    )


def test_usage_for_a_team_lead_is_restricted_to_their_team(deps: Container) -> None:
    deps.usage.put(_usage_record(1, "reviewer"))
    deps.usage.put(_usage_record(2, "helpdesk"))
    lead = {"email": "rae@example.com", "cognito:groups": "team-reviewer"}
    window = {"from": "2026-11-01", "to": "2026-11-30"}

    own = usage.handler(staff_event("GET", "/usage", lead, query={**window, "scope": "team"}), CTX)
    other = usage.handler(
        staff_event("GET", "/usage", lead, query={**window, "scope": "team", "team": "helpdesk"}),
        CTX,
    )
    company = usage.handler(
        staff_event("GET", "/usage", lead, query={**window, "scope": "company"}), CTX
    )
    finance = usage.handler(
        staff_event(
            "GET",
            "/usage",
            {"email": "fin@example.com", "cognito:groups": "finance"},
            query={**window, "scope": "company"},
        ),
        CTX,
    )

    assert [g["key"] for g in _body(own)["data"]["groups"]] == ["reviewer"]
    assert other["statusCode"] == 403 and company["statusCode"] == 403
    assert {g["key"] for g in _body(finance)["data"]["groups"]} == {"helpdesk", "reviewer"}


def test_delete_cache_for_a_customer_returns_the_count(
    deps: Container,
    fake_bedrock: FakeBedrock,
) -> None:
    _upload("policy-manual", "company", "policy", SAMPLE_DOCUMENTS["policy-manual"])
    fake_bedrock.queue()
    invoke.handler(invoke_event(INVOKE_BODY), CTX)

    response = admin.handler(
        staff_event("DELETE", "/cache", ADMIN, query={"customer": "cust-1"}), CTX
    )
    forbidden = admin.handler(
        staff_event(
            "DELETE",
            "/cache",
            {"email": "x@example.com", "cognito:groups": "finance"},
            query={"all": "true"},
        ),
        CTX,
    )

    assert _body(response)["data"] == {"deleted": 1, "selector": "customer"}
    assert forbidden["statusCode"] == 403


def test_audit_lookup_returns_document_versions_and_editor(
    deps: Container,
    fake_bedrock: FakeBedrock,
) -> None:
    _upload("policy-manual", "company", "policy", SAMPLE_DOCUMENTS["policy-manual"])
    fake_bedrock.queue()
    request_id = _body(invoke.handler(invoke_event(INVOKE_BODY), CTX))["data"]["request_id"]

    response = admin.handler(
        staff_event("GET", "/audit/{requestId}", ADMIN, path={"requestId": request_id}), CTX
    )

    data = _body(response)["data"]
    assert data["documents"] == [
        {
            "document_version": "policy-manual@000001",
            "author": "ana@example.com",
            "created_at": data["documents"][0]["created_at"],
        }
    ]
    assert data["events"][0]["event_type"] == "gateway.request"


def test_model_block_update_is_validated_and_applied(deps: Container) -> None:
    good = {
        "groups": {"eu-regulated": {"customers": ["cust-eu"], "blocked_models": ["deepseek.v3.2"]}}
    }

    applied = admin.handler(staff_event("PUT", "/models/blocks", ADMIN, body=good), CTX)
    rejected = admin.handler(
        staff_event("PUT", "/models/blocks", ADMIN, body={"groups": {"bad name!": {}}}), CTX
    )

    assert applied["statusCode"] == 200
    stored = deps.ssm.get_parameter(Name="/keepsake/models/blocks")["Parameter"]["Value"]
    assert json.loads(stored) == good
    assert rejected["statusCode"] == 400


def test_unknown_route_is_404_and_unexpected_errors_are_generic_500(
    deps: Container,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = admin.handler(staff_event("GET", "/nope", ADMIN), CTX)

    def explode(*_: object, **__: object) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(deps.documents, "list_documents", explode)
    crashed = documents.handler(staff_event("GET", "/documents", ADMIN), CTX)

    assert missing["statusCode"] == 404
    assert crashed["statusCode"] == 500
    assert "secret" not in crashed["body"]
