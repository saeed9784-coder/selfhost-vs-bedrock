"""Phase 0 analyzer tests (FR1.1-FR1.6)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from keepsake.analyzer import sample_logs
from keepsake.analyzer.costing import NOT_WORTH_BUILDING, WORTH_BUILDING, cost_agent
from keepsake.analyzer.logs import parse_lines
from keepsake.analyzer.pipeline import run_analysis
from keepsake.analyzer.repeats import analyze
from keepsake.analyzer.report import render_markdown
from keepsake.common.models import ModelPrice

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
MANUAL = "\n\n".join(
    f"Manual section {i}. " + "Reset the router before escalating. " * 6 for i in range(10)
)
PRICES = {"deepseek.v3.2": ModelPrice("deepseek.v3.2", 1.0, 2.0, 0.1, 1.25)}


def _log(agent: str, when: datetime, text: str, request_id: str) -> str:
    return json.dumps(
        {
            "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "requestId": request_id,
            "modelId": "deepseek.v3.2",
            "identity": {"arn": f"arn:aws:sts::111122223333:assumed-role/keepsake-agent-{agent}/s"},
            "input": {
                "inputBodyJson": {"messages": [{"role": "user", "content": [{"text": text}]}]},
                "inputTokenCount": len(text) // 4,
            },
        }
    )


def _stats(*offsets: timedelta, text: str = MANUAL):
    lines = [
        _log("helpdesk", T0 + off, f"{text}\n\nQuestion {i}?", f"r{i}")
        for i, off in enumerate(offsets)
    ]
    return analyze(parse_lines(lines).records)[0]


def test_repeated_share_and_length_per_agent_from_bedrock_log_format() -> None:
    parsed = parse_lines(
        [
            _log("helpdesk", T0, MANUAL, "r1"),
            _log("helpdesk", T0 + timedelta(minutes=5), MANUAL, "r2"),
            "not json",
        ]
    )

    stats = analyze(parsed.records)[0]

    assert parsed.skipped == 1
    assert stats.agent_id == "keepsake-agent-helpdesk"
    assert stats.requests[0].repeated_share == 0.0
    # Paragraph separators are not content, so a fully repeated request is just under 100 %.
    assert stats.requests[1].repeated_share == pytest.approx(1.0, abs=0.02)
    assert stats.max_repeated_tokens == pytest.approx(len(MANUAL) // 4, rel=0.02)


def test_gap_distribution_buckets() -> None:
    stats = _stats(timedelta(0), timedelta(minutes=30), timedelta(days=2), timedelta(days=12))

    gaps = dict(stats.gap_tokens)
    per_send = len(MANUAL) // 4
    assert gaps["<1h"] == pytest.approx(per_send, rel=0.05)
    assert gaps["<1d"] == 0
    assert gaps["<3d"] == pytest.approx(per_send, rel=0.05)
    assert gaps[">=7d"] == pytest.approx(per_send, rel=0.05)


def test_cached_window_pricing_includes_storage_and_longer_windows_cache_more() -> None:
    stats = _stats(timedelta(0), timedelta(minutes=30), timedelta(days=2))

    costing = cost_agent(stats, PRICES, storage_price=0.25, build_and_run_cost_usd=0.0)

    windows = {w.window: w for w in costing.windows}
    assert windows["1h"].cacheable_tokens < windows["3d"].cacheable_tokens
    assert all(w.storage_cost_usd > 0 for w in costing.windows)
    assert windows["3d"].cost_usd == pytest.approx(
        costing.cost_today_usd - windows["3d"].saving_usd
    )
    assert costing.verdict == WORTH_BUILDING


def test_not_worth_building_when_savings_do_not_exceed_cost() -> None:
    stats = _stats(timedelta(0), timedelta(minutes=30))
    no_repeats = _stats(timedelta(0), text="Short unique question without shared context.")

    expensive = cost_agent(stats, PRICES, 0.25, build_and_run_cost_usd=1_000_000.0)
    nothing = cost_agent(no_repeats, PRICES, 0.25, build_and_run_cost_usd=10.0)

    assert expensive.verdict == NOT_WORTH_BUILDING
    assert nothing.best_monthly_saving_usd == 0.0
    assert nothing.verdict == NOT_WORTH_BUILDING


def test_sample_generator_yields_fifty_helpdesk_calls_over_three_days() -> None:
    records = sample_logs.generate(seed=7)
    helpdesk = [r for r in records if r["requestMetadata"]["agent"] == "helpdesk"]
    times = sorted(datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in helpdesk)

    assert len(helpdesk) == 50
    assert times[-1] - times[0] <= timedelta(days=3)
    assert helpdesk[0]["input"]["inputTokenCount"] >= 100_000
    assert sample_logs.generate(seed=7)[0]["requestId"] == records[0]["requestId"]


def test_full_pipeline_on_sample_logs_reports_every_agent_and_a_verdict() -> None:
    parsed = parse_lines(sample_logs.to_lines(sample_logs.generate(seed=7)))
    prices = {
        sample_logs.HELPDESK_MODEL: ModelPrice(sample_logs.HELPDESK_MODEL, 0.62, 1.85, 0.062, 0.62),
        sample_logs.REVIEWER_MODEL: ModelPrice(sample_logs.REVIEWER_MODEL, 3.0, 15.0, 0.3, 3.75),
        sample_logs.ATLAS_MODEL: ModelPrice(sample_logs.ATLAS_MODEL, 0.8, 3.2, 0.2, 0.8),
    }

    report = run_analysis(parsed, prices, 0.25, 500.0, T0)

    agents = {a["agent_id"]: a for a in report.agents}
    assert set(agents) == {"atlas", "helpdesk", "reviewer"}
    assert agents["helpdesk"]["repeated_share"] > 0.9
    assert report.verdict in {WORTH_BUILDING, NOT_WORTH_BUILDING}
    assert "Overall verdict" in render_markdown(report)
