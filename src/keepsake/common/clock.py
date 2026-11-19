"""Injectable clock helpers so tests control time (TTLs, gaps, budget periods)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(moment: datetime) -> str:
    """ISO-8601 UTC with millisecond precision and a Z suffix; sorts lexicographically."""
    return (
        moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"
    )


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def epoch_seconds(moment: datetime) -> int:
    return int(moment.timestamp())
