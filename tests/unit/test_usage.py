"""Usage, cost, scope, budget and metrics tests (FR5.1, FR5.2, FR5.4, FR5.5, FR5.6, NFR7)."""

from __future__ import annotations

import json

import pytest

from keepsake.common.config import ConfigProvider
from keepsake.common.errors import AccessDenied, ValidationError
from keepsake.common.models import ModelPrice, TokenUsage, UsageRecord
from keepsake.usage.aggregation import hit_rate_over_time, percentile, summarize
from keepsake.usage.budget import check_agent, publish_alerts
from keepsake.usage.costing import price_model_call, price_reuse
from keepsake.usage.metrics import MetricsEmitter
from keepsake.usage.repository import UsageRepository
from keepsake.usage.scope import resolve_usage_scope
from tests.conftest import AwsResources

PRICE = ModelPrice(
    "m",
    input_per_million=3.0,
    output_per_million=15.0,
    cache_read_per_million=0.3,
    cache_write_per_million=3.75,
)


def _record(
    n: int,
    team: str = "reviewer",
    agent: str = "reviewer",
    cost: float = 1.0,
    saving: float = 0.5,
    storage: float = 0.0,
    outcome: str = "miss",
    latency: int = 100,
    day: int = 10,
) -> UsageRecord:
    return UsageRecord(
        request_id=f"r{n}",
        timestamp=f"2026-11-{day:02d}T12:00:{n % 60:02d}.000Z",
        agent_id=agent,
        team=team,
        customer_id="cust-1",
        model_id="m",
        input_tokens=1000,
        cache_read_tokens=800 if outcome != "miss" else 0,
        cache_write_tokens=0,
        full_price_tokens=200,
        output_tokens=50,
        cost_usd=cost,
        saving_usd=saving,
        storage_cost_usd=storage,
        latency_ms=latency,
        cache_outcome=outcome,
    )


def test_cost_and_saving_math() -> None:
    usage = TokenUsage(input_tokens=200_000, output_tokens=10_000, cache_read_tokens=800_000)

    call = price_model_call(
        usage, PRICE, stored_bytes=1_000_000_000, retention_hours=730, usd_per_gb_month=0.25
    )
    reuse = price_reuse(usage, PRICE)

    assert call.cost_usd == pytest.approx(0.6 + 0.24 + 0.15)
    assert call.saving_usd == pytest.approx((3.0 + 0.15) - 0.99)
    assert call.storage_cost_usd == pytest.approx(0.25)
    assert (reuse.cost_usd, reuse.saving_usd) == (0.0, pytest.approx(3.15))


def test_team_lead_sees_only_their_team_and_finance_sees_all(aws: AwsResources) -> None:
    repo = UsageRepository(table=aws.tables["usage"])
    repo.put(_record(1, team="reviewer", agent="reviewer"))
    repo.put(_record(2, team="helpdesk", agent="helpdesk"))
    lead = {"team-reviewer"}

    scope = resolve_usage_scope(lead, "team", None)
    team_records = repo.query_team(scope.team, "2026-11-01", "2026-11-30")
    assert {r.team for r in team_records} == {"reviewer"}
    with pytest.raises(AccessDenied):
        resolve_usage_scope(lead, "team", "helpdesk")
    with pytest.raises(AccessDenied):
        resolve_usage_scope(lead, "company", None)
    assert resolve_usage_scope({"finance"}, "company", None).team is None
    assert len(repo.query_company("2026-11-01", "2026-11-30")) == 2
    with pytest.raises(ValidationError):
        resolve_usage_scope({"finance"}, "everything", None)


def test_aggregates_hit_rate_over_time_and_latency_percentiles() -> None:
    records = [
        _record(i, outcome="provider_hit" if i % 2 else "miss", latency=i * 10, day=10 + i % 2)
        for i in range(1, 11)
    ]

    summary = summarize(records, "agent")[0]
    series = hit_rate_over_time(records, bucket_hours=24)

    assert summary.requests == 10
    assert summary.hit_rate == pytest.approx(0.5)
    assert summary.p50_latency_ms == 50 and summary.p99_latency_ms == 100
    assert [p.hit_rate for p in series] == [0.0, 1.0]
    assert percentile([], 99) == 0.0


def test_budget_alert_at_eighty_percent_is_published(aws: AwsResources) -> None:
    helpdesk = ConfigProvider(ssm=aws.ssm).agent_config("helpdesk")  # budget $50
    under = [_record(i, team="helpdesk", agent="helpdesk", cost=3.0) for i in range(13)]  # $39
    over = [*under, _record(99, team="helpdesk", agent="helpdesk", cost=1.5)]  # $40.50

    assert check_agent(helpdesk, under) == ()
    alerts = check_agent(helpdesk, over)
    assert [a.kind for a in alerts] == ["budget"]
    assert publish_alerts(aws.sns, aws.topic_arn, alerts) == 1


def test_storage_cost_above_savings_alerts(aws: AwsResources) -> None:
    atlas = ConfigProvider(ssm=aws.ssm).agent_config("atlas")
    records = [_record(1, team="atlas", agent="atlas", cost=0.01, saving=0.02, storage=0.05)]

    assert [a.kind for a in check_agent(atlas, records)] == ["storage-exceeds-savings"]


def test_metrics_are_emitted_in_emf_per_agent_and_team() -> None:
    lines: list[str] = []
    emitter = MetricsEmitter(writer=lines.append, clock_ms=lambda: 1)

    emitter.gateway_request(
        agent_id="helpdesk",
        team="helpdesk",
        cache_outcome="response_reuse",
        input_tokens=0,
        cache_read_tokens=1000,
        output_tokens=0,
        latency_ms=12,
        cost_usd=0.0,
        saving_usd=0.01,
        degraded=False,
    )
    emitter.component_request("documents", is_error=True)

    gateway, component = (json.loads(line) for line in lines)
    directive = gateway["_aws"]["CloudWatchMetrics"][0]
    assert directive["Dimensions"] == [["agent"], ["team"]]
    assert gateway["agent"] == "helpdesk" and gateway["CacheHits"] == 1
    assert component["Requests"] == 1 and component["Errors"] == 1
