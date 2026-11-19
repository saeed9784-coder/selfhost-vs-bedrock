"""Phase 0 report as JSON and Markdown, optionally written to S3 (FR1.1-FR1.5)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from keepsake.analyzer.costing import BYTES_PER_TOKEN, AgentCosting, overall_verdict
from keepsake.analyzer.repeats import WINDOW_CHARS, AgentRepeatStats
from keepsake.common.aws import AWS_ERRORS, translate_aws_error
from keepsake.common.clock import iso


@dataclass(frozen=True)
class Phase0Report:
    generated_at: str
    build_and_run_cost_usd: float
    verdict: str
    agents: tuple[dict[str, Any], ...]
    models: tuple[dict[str, Any], ...]
    skipped_log_lines: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "build_and_run_cost_usd_per_month": self.build_and_run_cost_usd,
            "verdict": self.verdict,
            "verdict_scope": "preliminary_per_agent_screen",
            "investment_gate_status": "INCOMPLETE_PLATFORM_TELEMETRY_REQUIRED",
            "skipped_log_lines": self.skipped_log_lines,
            "estimate_limitations": {
                "repeated_text_is_prefix_hit_proxy": True,
                "agent_savings_estimate_scope": "within_agent_and_model_only",
                "model_reuse_scope": "all_agents_in_input_logs_by_model",
                "model_prefix_chunk_chars": WINDOW_CHARS,
                "account_scope": "one_aws_account_per_analysis",
                "bytes_per_cached_token_assumption": BYTES_PER_TOKEN,
                "prefill_gpu_time_saved": None,
                "prefill_gpu_time_note": (
                    "Bedrock invocation logs do not expose serving-side GPU time or KV-cache bytes; "
                    "these require Bedrock platform telemetry."
                ),
            },
            "agents": list(self.agents),
            "models": list(self.models),
        }


def _agent_section(stats: AgentRepeatStats, costing: AgentCosting) -> dict[str, Any]:
    return {
        "agent_id": stats.agent_id,
        "model_id": stats.model_id,
        "request_count": stats.request_count,
        "total_input_tokens": stats.total_input_tokens,
        "repeated_tokens": stats.repeated_tokens,
        "repeated_share": round(stats.repeated_share, 4),
        "mean_repeated_tokens_per_request": round(stats.mean_repeated_tokens, 1),
        "max_repeated_tokens_per_request": stats.max_repeated_tokens,
        "gap_distribution_tokens": dict(stats.gap_tokens),
        "observation_hours": round(stats.observation_hours, 2),
        "cost_today_usd": round(costing.cost_today_usd, 4),
        "windows": [
            {
                "window": w.window,
                "cacheable_tokens": w.cacheable_tokens,
                "cost_usd": round(w.cost_usd, 4),
                "storage_cost_usd": round(w.storage_cost_usd, 6),
                "saving_usd": round(w.saving_usd, 4),
            }
            for w in costing.windows
        ],
        "best_window": costing.best_window,
        "best_monthly_saving_usd": round(costing.best_monthly_saving_usd, 2),
        "verdict": costing.verdict,
        "requests": [
            {
                "request_id": r.request_id,
                "repeated_share": round(r.repeated_share, 4),
                "repeated_tokens": r.repeated_tokens,
            }
            for r in stats.requests
        ],
    }


def _model_section(stats: AgentRepeatStats) -> dict[str, Any]:
    """Account-level reuse view, grouped by model across all agents in the input logs."""
    return {
        "model_id": stats.model_id,
        "request_count": stats.request_count,
        "total_input_tokens": stats.total_input_tokens,
        "repeated_tokens": stats.repeated_tokens,
        "repeated_share": round(stats.repeated_share, 4),
        "gap_distribution_tokens": dict(stats.gap_tokens),
        "observation_hours": round(stats.observation_hours, 2),
    }


def build_report(
    stats: tuple[AgentRepeatStats, ...],
    costings: tuple[AgentCosting, ...],
    build_and_run_cost_usd: float,
    generated_at: datetime,
    skipped_log_lines: int = 0,
    model_stats: tuple[AgentRepeatStats, ...] = (),
) -> Phase0Report:
    by_agent_model = {(c.agent_id, c.model_id): c for c in costings}
    return Phase0Report(
        generated_at=iso(generated_at),
        build_and_run_cost_usd=build_and_run_cost_usd,
        verdict=overall_verdict(costings, build_and_run_cost_usd),
        agents=tuple(_agent_section(s, by_agent_model[(s.agent_id, s.model_id)]) for s in stats),
        models=tuple(_model_section(s) for s in model_stats),
        skipped_log_lines=skipped_log_lines,
    )


def render_markdown(report: Phase0Report) -> str:
    lines = [
        "# Keepsake Phase 0 — repeated-context report",
        "",
        f"Generated: {report.generated_at}",
        f"Configured build-and-run cost: ${report.build_and_run_cost_usd:,.2f} per month",
        f"**Preliminary per-agent screen: {report.verdict}**",
        "**Proposal investment gate: incomplete — validate GPU time and KV-cache size with "
        "Bedrock platform telemetry.**",
        "",
        "> Decision aid only: repeated text segments are a proxy for reusable prompt prefixes, "
        "not observed Bedrock KV-cache hits. Agent estimates are within-agent; the model summary "
        "scans across agents. Storage uses a planning assumption of "
        f"{BYTES_PER_TOKEN} bytes per cached token. Bedrock GPU time and actual KV-cache size "
        "are not present in customer invocation logs and must come from platform telemetry.",
        "",
        "## Account-level repeated input by model",
        "",
        f"This scan compares common leading prompt chunks of {WINDOW_CHARS} characters across "
        "agents using the same model. It is not a Bedrock KV-cache hit measurement.",
        "",
        "| Model | Requests | Input tokens | Repeated tokens | Repeated share |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in report.models:
        lines.append(
            f"| {model['model_id']} | {model['request_count']} | {model['total_input_tokens']:,} "
            f"| {model['repeated_tokens']:,} | {model['repeated_share']:.1%} |"
        )
    lines += [
        "",
        "### Model-level reuse interval",
        "",
        "| Model | <1h | <1d | <3d | <7d | >=7d |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in report.models:
        gaps = model["gap_distribution_tokens"]
        lines.append(
            f"| {model['model_id']} | "
            + " | ".join(f"{gaps[b]:,}" for b in ("<1h", "<1d", "<3d", "<7d", ">=7d"))
            + " |"
        )
    lines += [
        "",
        "| Agent | Model | Requests | Repeated share | Mean repeated tokens | Best window "
        "| Best monthly saving | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for agent in report.agents:
        lines.append(
            f"| {agent['agent_id']} | {agent['model_id']} | {agent['request_count']} "
            f"| {agent['repeated_share']:.1%} | {agent['mean_repeated_tokens_per_request']:,.0f} "
            f"| {agent['best_window'] or '-'} | ${agent['best_monthly_saving_usd']:,.2f} "
            f"| {agent['verdict']} |"
        )
    lines += [
        "",
        "## Gap before re-send by agent (tokens)",
        "",
        "| Agent | <1h | <1d | <3d | <7d | >=7d |",
        "|---|---|---|---|---|---|",
    ]
    for agent in report.agents:
        gaps = agent["gap_distribution_tokens"]
        lines.append(
            f"| {agent['agent_id']} | "
            + " | ".join(f"{gaps[b]:,}" for b in ("<1h", "<1d", "<3d", "<7d", ">=7d"))
            + " |"
        )
    if report.skipped_log_lines:
        lines += ["", f"Skipped {report.skipped_log_lines} malformed log lines."]
    return "\n".join(lines) + "\n"


def write_report(
    s3: Any, bucket: str, prefix: str, report: Phase0Report, kms_key_id: str
) -> tuple[str, str]:
    """Write `<prefix>/<timestamp>/report.json` and `report.md` with SSE-KMS (CMK)."""
    base = f"{prefix.rstrip('/')}/{report.generated_at}"
    json_key, md_key = f"{base}/report.json", f"{base}/report.md"
    sse = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key_id}
    try:
        s3.put_object(
            Bucket=bucket,
            Key=json_key,
            ContentType="application/json",
            Body=json.dumps(report.to_dict(), indent=2).encode("utf-8"),
            **sse,
        )
        s3.put_object(
            Bucket=bucket,
            Key=md_key,
            ContentType="text/markdown",
            Body=render_markdown(report).encode("utf-8"),
            **sse,
        )
    except AWS_ERRORS as exc:
        raise translate_aws_error(exc, "write Phase 0 report") from exc
    return json_key, md_key
