"""CloudWatch metrics in Embedded Metric Format (FR3.9, FR5.2, NFR7).

Lambda forwards stdout to CloudWatch Logs, which extracts EMF documents into metrics with no
extra API call (so metric publishing never adds latency or a failure mode to a request).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping

NAMESPACE = "Keepsake"
Writer = Callable[[str], None]


def _stdout(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class MetricsEmitter:
    def __init__(
        self,
        *,
        namespace: str = NAMESPACE,
        writer: Writer = _stdout,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        self._namespace = namespace
        self._writer = writer
        self._clock_ms = clock_ms

    def emit(
        self,
        dimensions: Mapping[str, str],
        metrics: Mapping[str, tuple[float, str]],
        dimension_sets: list[list[str]] | None = None,
    ) -> None:
        """Write one EMF document. `metrics` maps name -> (value, unit)."""
        document: dict[str, object] = {
            "_aws": {
                "Timestamp": self._clock_ms(),
                "CloudWatchMetrics": [
                    {
                        "Namespace": self._namespace,
                        "Dimensions": dimension_sets or [list(dimensions)],
                        "Metrics": [
                            {"Name": name, "Unit": unit} for name, (_, unit) in metrics.items()
                        ],
                    }
                ],
            },
            **dict(dimensions),
            **{name: value for name, (value, _) in metrics.items()},
        }
        self._writer(json.dumps(document))

    def component_request(self, component: str, *, is_error: bool) -> None:
        """Request count and error count for any component (NFR7)."""
        self.emit(
            {"component": component},
            {"Requests": (1, "Count"), "Errors": (1 if is_error else 0, "Count")},
        )

    def gateway_request(
        self,
        *,
        agent_id: str,
        team: str,
        cache_outcome: str,
        input_tokens: int,
        cache_read_tokens: int,
        output_tokens: int,
        latency_ms: int,
        cost_usd: float,
        saving_usd: float,
        degraded: bool,
    ) -> None:
        """Per-agent and per-team gateway metrics (FR5.2); degraded calls are counted (FR3.9)."""
        self.emit(
            {"agent": agent_id, "team": team},
            {
                "GatewayRequests": (1, "Count"),
                "CacheHits": (0 if cache_outcome == "miss" else 1, "Count"),
                "ResponseReuse": (1 if cache_outcome == "response_reuse" else 0, "Count"),
                "InputTokens": (input_tokens, "Count"),
                "CachedTokens": (cache_read_tokens, "Count"),
                "OutputTokens": (output_tokens, "Count"),
                "Latency": (latency_ms, "Milliseconds"),
                "CostUSD": (cost_usd, "None"),
                "SavingUSD": (saving_usd, "None"),
                "Degraded": (1 if degraded else 0, "Count"),
            },
            dimension_sets=[["agent"], ["team"]],
        )

    def degraded(self, component: str, reason: str) -> None:
        self.emit(
            {"component": component, "reason": reason},
            {"Degraded": (1, "Count")},
            dimension_sets=[["component"]],
        )
