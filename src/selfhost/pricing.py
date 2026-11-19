"""Cost model: what one answer costs on Bedrock versus on our own GPU, and at monthly scale.

Two self-hosted figures are reported, because they answer different questions:

* ``actual`` charges the request for the GPU time it held, split across the requests that shared
  the GPU at that moment. With one person chatting, the GPU is mostly idle, so this is high.
* ``at_capacity`` charges only the GPU time the tokens need when the GPU is kept busy at the
  target utilization. Prefix-cache hits skip prefill, so they cost almost nothing here.

Bedrock on-demand has no idle cost and (for these models) no prompt cache, so every turn pays
for the whole conversation history again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MILLION = 1_000_000
SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0


@dataclass(frozen=True)
class Throughput:
    prefill_tps: float
    decode_tps: float


def bedrock_cost(usage: Usage, input_per_million: float, output_per_million: float) -> float:
    return (
        usage.input_tokens * input_per_million + usage.output_tokens * output_per_million
    ) / MILLION


def gpu_seconds_needed(usage: Usage, throughput: Throughput) -> float:
    """GPU-seconds the tokens need on a saturated GPU; cached prefix tokens skip prefill."""
    uncached = max(0, usage.input_tokens - usage.cached_tokens)
    return uncached / throughput.prefill_tps + usage.output_tokens / throughput.decode_tps


def selfhost_capacity_cost(
    usage: Usage, throughput: Throughput, usd_per_hour: float, utilization: float
) -> float:
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    return gpu_seconds_needed(usage, throughput) * usd_per_hour / SECONDS_PER_HOUR / utilization


def selfhost_actual_cost(
    elapsed_seconds: float, avg_concurrency: float, usd_per_hour: float
) -> float:
    share = elapsed_seconds / max(1.0, avg_concurrency)
    return share * usd_per_hour / SECONDS_PER_HOUR


@dataclass(frozen=True)
class Workload:
    """A month of traffic. ``prefix_reuse`` is the share of input tokens served from KV cache."""

    input_tokens: float
    output_tokens: float
    prefix_reuse: float


@dataclass(frozen=True)
class MonthlyComparison:
    bedrock_usd: float
    selfhost_usd: float
    instances: int
    gpu_hours_needed: float
    utilization: float

    @property
    def savings_usd(self) -> float:
        return self.bedrock_usd - self.selfhost_usd


def monthly_comparison(
    workload: Workload,
    *,
    input_per_million: float,
    output_per_million: float,
    throughput: Throughput,
    usd_per_hour: float,
    fixed_monthly_usd: float,
    hours_per_month: float,
    target_utilization: float,
    online_hours: float | None = None,
    committed: bool = False,
) -> MonthlyComparison:
    """Compare a month of traffic on Bedrock with self-hosted instances.

    ``online_hours`` is how long the instances run each month (scaled to zero otherwise); the
    traffic must fit inside those hours. ``committed`` (Savings Plans) bills every hour of the
    month whether the instances run or not.
    """
    if not 0 <= workload.prefix_reuse < 1:
        raise ValueError("prefix_reuse must be in [0, 1)")
    if not 0 < target_utilization <= 1:
        raise ValueError("target_utilization must be in (0, 1]")
    online = hours_per_month if online_hours is None else online_hours
    if not 0 < online <= hours_per_month:
        raise ValueError("online_hours must be in (0, hours_per_month]")
    bedrock = (
        workload.input_tokens * input_per_million + workload.output_tokens * output_per_million
    ) / MILLION
    uncached = workload.input_tokens * (1 - workload.prefix_reuse)
    gpu_hours = (
        uncached / throughput.prefill_tps + workload.output_tokens / throughput.decode_tps
    ) / SECONDS_PER_HOUR
    instances = max(1, math.ceil(gpu_hours / (online * target_utilization)))
    billed_hours = hours_per_month if committed else online
    selfhost = instances * billed_hours * usd_per_hour + fixed_monthly_usd
    return MonthlyComparison(
        bedrock_usd=bedrock,
        selfhost_usd=selfhost,
        instances=instances,
        gpu_hours_needed=gpu_hours,
        utilization=min(1.0, gpu_hours / (instances * online)),
    )
