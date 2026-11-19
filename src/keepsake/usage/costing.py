"""Per-request cost, saving versus no cache, and cache storage cost (FR5.1, FR5.6)."""

from __future__ import annotations

from dataclasses import dataclass

from keepsake.common.models import ModelPrice, TokenUsage

HOURS_PER_MONTH = 730.0
_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class RequestCost:
    cost_usd: float
    saving_usd: float
    storage_cost_usd: float


def model_cost(usage: TokenUsage, price: ModelPrice) -> float:
    """What Bedrock bills for this usage, with cache reads and writes at their own prices."""
    return (
        usage.input_tokens * price.input_per_million
        + usage.cache_read_tokens * price.cache_read_per_million
        + usage.cache_write_tokens * price.cache_write_per_million
        + usage.output_tokens * price.output_per_million
    ) / _PER_MILLION


def no_cache_cost(usage: TokenUsage, price: ModelPrice) -> float:
    """What the same request would cost with every input token at the full input price."""
    return (
        usage.total_input_tokens * price.input_per_million
        + usage.output_tokens * price.output_per_million
    ) / _PER_MILLION


def storage_cost(stored_bytes: int, retention_hours: int, usd_per_gb_month: float) -> float:
    """Cost of holding `stored_bytes` for the retention window."""
    return stored_bytes / 1e9 * usd_per_gb_month * retention_hours / HOURS_PER_MONTH


def price_model_call(
    usage: TokenUsage,
    price: ModelPrice,
    stored_bytes: int,
    retention_hours: int,
    usd_per_gb_month: float,
) -> RequestCost:
    """A request that called the model (miss or provider cache hit)."""
    cost = model_cost(usage, price)
    return RequestCost(
        cost_usd=cost,
        saving_usd=no_cache_cost(usage, price) - cost,
        storage_cost_usd=storage_cost(stored_bytes, retention_hours, usd_per_gb_month),
    )


def price_reuse(original_usage: TokenUsage, price: ModelPrice) -> RequestCost:
    """A request served from response reuse: no model call, the full no-cache cost is saved."""
    return RequestCost(
        cost_usd=0.0, saving_usd=no_cache_cost(original_usage, price), storage_cost_usd=0.0
    )
