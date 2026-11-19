"""Price repeated context today versus cached for 1 h / 1 d / 3 d / 7 d (FR1.4, FR1.5).

For a retention window W, repeated tokens whose gap is shorter than W could have been served
from cache: they are billed at the cache-read price instead of the input price. Each distinct
repeated segment is written once at the cache-write price instead of the input price, and the
cached bytes are held for (observation period + W) at the storage price per GB-month.

Figures are computed for the observed period, then projected to a 730-hour month so they can
be compared with the configured monthly build-and-run cost of the cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from keepsake.analyzer.repeats import GAP_BUCKETS, AgentRepeatStats
from keepsake.common.errors import ConfigurationError
from keepsake.common.models import ModelPrice

HOURS_PER_MONTH = 730.0
BYTES_PER_TOKEN = 4
WINDOWS: tuple[tuple[str, int], ...] = (("1h", 1), ("1d", 24), ("3d", 72), ("7d", 168))
WORTH_BUILDING = "WORTH BUILDING"
NOT_WORTH_BUILDING = "NOT WORTH BUILDING"
_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class WindowCost:
    window: str
    cacheable_tokens: int
    cost_usd: float
    storage_cost_usd: float
    saving_usd: float


@dataclass(frozen=True)
class AgentCosting:
    agent_id: str
    model_id: str
    monthly_factor: float
    cost_today_usd: float
    windows: tuple[WindowCost, ...]
    best_window: str | None
    best_monthly_saving_usd: float
    build_and_run_cost_usd: float
    verdict: str


def cacheable_tokens(stats: AgentRepeatStats, window_hours: int) -> int:
    """Repeated tokens whose gap bucket lies entirely inside the window."""
    limit_seconds = window_hours * 3600
    return sum(
        tokens
        for (name, upper), (_, tokens) in zip(GAP_BUCKETS, stats.gap_tokens, strict=True)
        if upper <= limit_seconds and name != ">=7d"
    )


def _window_cost(
    stats: AgentRepeatStats, price: ModelPrice, window: str, hours: int, storage_price: float
) -> WindowCost:
    hits = cacheable_tokens(stats, hours)
    written = min(stats.unique_repeated_tokens, hits)
    full_price = stats.total_input_tokens - hits - written
    tokens_cost = (
        full_price * price.input_per_million
        + hits * price.cache_read_per_million
        + written * price.cache_write_per_million
    ) / _PER_MILLION
    stored_gb = stats.unique_repeated_tokens * BYTES_PER_TOKEN / 1e9 if hits else 0.0
    storage = stored_gb * storage_price * (stats.observation_hours + hours) / HOURS_PER_MONTH
    cost_today = stats.total_input_tokens * price.input_per_million / _PER_MILLION
    total = tokens_cost + storage
    return WindowCost(window, hits, total, storage, cost_today - total)


def monthly_factor(observation_hours: float) -> float:
    return HOURS_PER_MONTH / max(observation_hours, 1.0)


def cost_agent(
    stats: AgentRepeatStats,
    prices: Mapping[str, ModelPrice],
    storage_price: float,
    build_and_run_cost_usd: float,
) -> AgentCosting:
    price = prices.get(stats.model_id)
    if price is None:
        raise ConfigurationError(f"no price configured for model {stats.model_id}")
    factor = monthly_factor(stats.observation_hours)
    windows = tuple(
        _window_cost(stats, price, name, hours, storage_price) for name, hours in WINDOWS
    )
    best = max(windows, key=lambda w: w.saving_usd)
    best_saving = max(best.saving_usd, 0.0) * factor
    verdict = WORTH_BUILDING if best_saving > build_and_run_cost_usd else NOT_WORTH_BUILDING
    return AgentCosting(
        agent_id=stats.agent_id,
        model_id=stats.model_id,
        monthly_factor=factor,
        cost_today_usd=stats.total_input_tokens * price.input_per_million / _PER_MILLION,
        windows=windows,
        best_window=best.window if best_saving > 0 else None,
        best_monthly_saving_usd=best_saving,
        build_and_run_cost_usd=build_and_run_cost_usd,
        verdict=verdict,
    )


def overall_verdict(costings: tuple[AgentCosting, ...], build_and_run_cost_usd: float) -> str:
    total = sum(c.best_monthly_saving_usd for c in costings)
    return WORTH_BUILDING if total > build_and_run_cost_usd else NOT_WORTH_BUILDING
