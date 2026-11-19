"""Dashboard aggregates per agent or team (FR5.3).

Tokens sent, tokens from cache, full-price tokens, spend, saving, cache storage cost, hit rate,
hit rate over time and p50/p99 latency.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta

from keepsake.common.clock import iso, parse_iso
from keepsake.common.errors import ValidationError
from keepsake.common.models import UsageRecord

GROUPINGS = frozenset({"agent", "team"})


@dataclass(frozen=True)
class UsageSummary:
    key: str
    requests: int
    tokens_sent: int
    tokens_from_cache: int
    full_price_tokens: int
    output_tokens: int
    spend_usd: float
    saved_usd: float
    storage_cost_usd: float
    hit_rate: float
    p50_latency_ms: float
    p99_latency_ms: float
    degraded_requests: int


@dataclass(frozen=True)
class HitRatePoint:
    bucket_start: str
    requests: int
    hit_rate: float


def percentile(values: Iterable[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 for no values."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return float(ordered[rank - 1])


def _is_hit(record: UsageRecord) -> bool:
    return record.cache_outcome != "miss"


def _summarize(key: str, records: list[UsageRecord]) -> UsageSummary:
    latencies = [r.latency_ms for r in records]
    return UsageSummary(
        key=key,
        requests=len(records),
        tokens_sent=sum(
            r.input_tokens + r.cache_read_tokens + r.cache_write_tokens for r in records
        ),
        tokens_from_cache=sum(r.cache_read_tokens for r in records),
        full_price_tokens=sum(r.full_price_tokens for r in records),
        output_tokens=sum(r.output_tokens for r in records),
        spend_usd=round(sum(r.cost_usd for r in records), 6),
        saved_usd=round(sum(r.saving_usd for r in records), 6),
        storage_cost_usd=round(sum(r.storage_cost_usd for r in records), 6),
        hit_rate=sum(1 for r in records if _is_hit(r)) / len(records) if records else 0.0,
        p50_latency_ms=percentile(latencies, 50),
        p99_latency_ms=percentile(latencies, 99),
        degraded_requests=sum(1 for r in records if r.degraded),
    )


def summarize(records: Iterable[UsageRecord], group_by: str) -> tuple[UsageSummary, ...]:
    if group_by not in GROUPINGS:
        raise ValidationError(f"group_by must be one of {sorted(GROUPINGS)}")
    groups: dict[str, list[UsageRecord]] = defaultdict(list)
    for record in records:
        groups[record.agent_id if group_by == "agent" else record.team].append(record)
    return tuple(_summarize(key, groups[key]) for key in sorted(groups))


def totals(records: Iterable[UsageRecord]) -> UsageSummary:
    return _summarize("total", list(records))


def hit_rate_over_time(
    records: Iterable[UsageRecord], bucket_hours: int = 24
) -> tuple[HitRatePoint, ...]:
    if bucket_hours < 1:
        raise ValidationError("bucket_hours must be at least 1")
    buckets: dict[str, list[UsageRecord]] = defaultdict(list)
    width = timedelta(hours=bucket_hours)
    for record in records:
        moment = parse_iso(record.timestamp)
        start = (
            moment - (moment - moment.replace(hour=0, minute=0, second=0, microsecond=0)) % width
        )
        buckets[iso(start)].append(record)
    return tuple(
        HitRatePoint(
            bucket_start=key,
            requests=len(items),
            hit_rate=sum(1 for r in items if _is_hit(r)) / len(items),
        )
        for key, items in sorted(buckets.items())
    )
