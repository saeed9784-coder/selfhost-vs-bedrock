"""Run the Phase 0 analysis end to end: parse -> detect repeats -> price -> report."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from keepsake.analyzer.costing import cost_agent
from keepsake.analyzer.logs import ParseResult
from keepsake.analyzer.repeats import analyze, analyze_by_model
from keepsake.analyzer.report import Phase0Report, build_report
from keepsake.common.models import ModelPrice


def run_analysis(
    parsed: ParseResult,
    prices: Mapping[str, ModelPrice],
    storage_price: float,
    build_and_run_cost_usd: float,
    generated_at: datetime,
) -> Phase0Report:
    stats = analyze(parsed.records)
    model_stats = analyze_by_model(parsed.records)
    costings = tuple(cost_agent(s, prices, storage_price, build_and_run_cost_usd) for s in stats)
    return build_report(
        stats,
        costings,
        build_and_run_cost_usd,
        generated_at,
        parsed.skipped,
        model_stats,
    )
