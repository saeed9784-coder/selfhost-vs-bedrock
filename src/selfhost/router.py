"""Cache-aware router and global scheduler for the self-hosted vLLM replicas.

Order of decisions, following the architecture diagram:
1. Context-length tier: only replicas whose max context fits the prompt plus the answer.
2. Admission control: replicas at their in-flight limit, or marked unhealthy, are skipped;
   if none are left the caller gets ``Overloaded`` (HTTP 429, backpressure).
3. Prefix affinity: the replica holding the longest cached prefix wins.
4. Load: ties go to the replica with the lowest load score (running, waiting, KV pressure),
   read from each replica's Prometheus ``/metrics``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from selfhost.prefix_index import PrefixIndex
from selfhost.settings import Replica

UNHEALTHY_SECONDS = 15.0
METRIC_LINE = re.compile(r"^((?:vllm|llamacpp):[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")


class Overloaded(Exception):
    """No replica can admit the request right now."""


class NoReplicaForContext(Exception):
    """The prompt is longer than every replica's context tier."""


@dataclass(frozen=True)
class ReplicaLoad:
    running: float = 0.0
    waiting: float = 0.0
    kv_usage: float = 0.0  # 0..1
    prefix_hits: float = 0.0
    prefix_queries: float = 0.0

    @property
    def score(self) -> float:
        return self.running + 2 * self.waiting + 10 * self.kv_usage


@dataclass(frozen=True)
class RouteDecision:
    replica: Replica
    matched_blocks: int
    total_blocks: int


def parse_vllm_metrics(text: str) -> ReplicaLoad:
    """Read the gauges the scheduler needs from vLLM or llama.cpp (``--metrics``) output."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC_LINE.match(line.strip())
        if match:
            values[match.group(1)] = values.get(match.group(1), 0.0) + float(match.group(2))

    def first(*names: str) -> float:
        return next((values[n] for n in names if n in values), 0.0)

    return ReplicaLoad(
        running=first("vllm:num_requests_running", "llamacpp:requests_processing"),
        waiting=first("vllm:num_requests_waiting", "llamacpp:requests_deferred"),
        kv_usage=first(
            "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "llamacpp:kv_cache_usage_ratio"
        ),
        prefix_hits=first("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
        prefix_queries=first("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"),
    )


class Router:
    def __init__(
        self,
        replicas: Sequence[Replica],
        index: PrefixIndex,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not replicas:
            raise ValueError("Router needs at least one replica")
        self._replicas = {r.name: r for r in replicas}
        self._index = index
        self._clock = clock
        self._inflight = dict.fromkeys(self._replicas, 0)
        self._unhealthy_until = dict.fromkeys(self._replicas, 0.0)
        self._load = {name: ReplicaLoad() for name in self._replicas}
        self._busy_integral = 0.0  # integral of total in-flight requests over time
        self._busy_mark = clock()

    @property
    def replicas(self) -> tuple[Replica, ...]:
        return tuple(self._replicas.values())

    def load(self, name: str) -> ReplicaLoad:
        return self._load[name]

    def set_load(self, name: str, load: ReplicaLoad) -> None:
        self._load[name] = load

    def mark_unhealthy(self, name: str) -> None:
        self._unhealthy_until[name] = self._clock() + UNHEALTHY_SECONDS

    def is_healthy(self, name: str) -> bool:
        return self._unhealthy_until[name] <= self._clock()

    def busy_integral(self) -> float:
        self._advance()
        return self._busy_integral

    def route(
        self,
        hashes: Sequence[str],
        estimated_tokens: int,
        exclude: frozenset[str] = frozenset(),
    ) -> RouteDecision:
        fits = [r for r in self._replicas.values() if r.max_context >= estimated_tokens]
        if not fits:
            raise NoReplicaForContext(f"about {estimated_tokens} tokens exceeds every replica")
        admissible = [
            r
            for r in fits
            if r.name not in exclude
            and self.is_healthy(r.name)
            and self._inflight[r.name] < r.max_inflight
        ]
        if not admissible:
            raise Overloaded("all replicas are busy or unhealthy")
        best = max(
            admissible,
            key=lambda r: (
                self._index.matched_blocks(hashes, r.name),
                -self._load[r.name].score - self._inflight[r.name],
            ),
        )
        return RouteDecision(
            replica=best,
            matched_blocks=self._index.matched_blocks(hashes, best.name),
            total_blocks=len(hashes),
        )

    @contextmanager
    def lease(self, name: str) -> Iterator[None]:
        self._advance()
        self._inflight[name] += 1
        try:
            yield
        finally:
            self._advance()
            self._inflight[name] -= 1

    def _advance(self) -> None:
        now = self._clock()
        self._busy_integral += sum(self._inflight.values()) * (now - self._busy_mark)
        self._busy_mark = now
