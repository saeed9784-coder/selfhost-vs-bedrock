"""Running totals for the comparison panel, and CloudWatch metrics for the SLO dashboard."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

LOGGER = logging.getLogger("selfhost.telemetry")
FLUSH_SECONDS = 60
MAX_BUFFER = 1000


@dataclass(frozen=True)
class TargetTotals:
    requests: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    capacity_cost_usd: float = 0.0
    ttft_seconds_sum: float = 0.0
    generation_seconds_sum: float = 0.0

    def add(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        cost_usd: float,
        capacity_cost_usd: float,
        ttft_seconds: float,
        elapsed_seconds: float,
    ) -> TargetTotals:
        return replace(
            self,
            requests=self.requests + 1,
            input_tokens=self.input_tokens + input_tokens,
            output_tokens=self.output_tokens + output_tokens,
            cached_tokens=self.cached_tokens + cached_tokens,
            cost_usd=self.cost_usd + cost_usd,
            capacity_cost_usd=self.capacity_cost_usd + capacity_cost_usd,
            ttft_seconds_sum=self.ttft_seconds_sum + ttft_seconds,
            generation_seconds_sum=self.generation_seconds_sum
            + max(0.0, elapsed_seconds - ttft_seconds),
        )

    def failed(self) -> TargetTotals:
        return replace(self, errors=self.errors + 1)

    def summary(self) -> dict[str, Any]:
        done = max(1, self.requests)
        return {
            **asdict(self),
            "avg_ttft_ms": round(1000 * self.ttft_seconds_sum / done, 1),
            "output_tokens_per_second": (
                round(self.output_tokens / self.generation_seconds_sum, 1)
                if self.generation_seconds_sum
                else 0.0
            ),
            "cache_hit_ratio": (
                round(self.cached_tokens / self.input_tokens, 4) if self.input_tokens else 0.0
            ),
        }


class Stats:
    """Thread-safe holder of per-target totals since the gateway started."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, TargetTotals] = {}
        self._clock = clock
        self.started_at = clock()

    def record(self, target: str, **values: Any) -> None:
        with self._lock:
            self._totals[target] = self._totals.get(target, TargetTotals()).add(**values)

    def record_error(self, target: str) -> None:
        with self._lock:
            self._totals[target] = self._totals.get(target, TargetTotals()).failed()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: totals.summary() for name, totals in self._totals.items()}

    def uptime_seconds(self) -> float:
        return self._clock() - self.started_at


class CloudWatchMetrics:
    """Buffers datapoints and sends them with PutMetricData once a minute, off the event loop."""

    def __init__(self, client: Any, namespace: str) -> None:
        self._client = client
        self._namespace = namespace
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cw-metrics", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.flush()

    def put(self, target: str, model: str, values: dict[str, tuple[float, str]]) -> None:
        dims = [{"Name": "Target", "Value": target}, {"Name": "Model", "Value": model}]
        points = [
            {"MetricName": name, "Value": value, "Unit": unit, "Dimensions": dims}
            for name, (value, unit) in values.items()
        ]
        with self._lock:
            self._buffer = (self._buffer + points)[-MAX_BUFFER:]

    def flush(self) -> None:
        with self._lock:
            pending, self._buffer = self._buffer, []
        for start in range(0, len(pending), 500):
            try:
                self._client.put_metric_data(
                    Namespace=self._namespace, MetricData=pending[start : start + 500]
                )
            except Exception:  # metrics must never break chat; log and drop this batch
                LOGGER.exception("PutMetricData failed; dropped %d datapoints", len(pending))

    def _run(self) -> None:
        while not self._stop.wait(FLUSH_SECONDS):
            self.flush()
