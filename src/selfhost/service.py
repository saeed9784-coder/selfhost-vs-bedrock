"""One chat turn sent to the self-hosted pool and to Bedrock at once, with costs per answer."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

from selfhost.backends import (
    BackendError,
    BedrockBackend,
    ChatRequest,
    Delta,
    Final,
    Reasoning,
    ReplicaUnavailable,
    VllmBackend,
    openai_messages,
)
from selfhost.prefix_index import PrefixIndex, block_hashes
from selfhost.pricing import (
    Throughput,
    bedrock_cost,
    selfhost_actual_cost,
    selfhost_capacity_cost,
)
from selfhost.router import NoReplicaForContext, Overloaded, Router
from selfhost.settings import Settings
from selfhost.telemetry import CloudWatchMetrics, Stats

LOGGER = logging.getLogger("selfhost.service")
SELFHOST = "selfhost"
BEDROCK = "bedrock"
CHARS_PER_TOKEN = 4
HEARTBEAT_SECONDS = 15.0
# asyncio.wait_for raises asyncio.TimeoutError, which is only the builtin from Python 3.11;
# the GPU AMI ships Python 3.10.
WAIT_TIMEOUT: type[Exception] = (
    asyncio.TimeoutError
)  # rough estimate, used only to pick a context-length tier


def _estimate_tokens(request: ChatRequest) -> int:
    chars = len(request.system) + sum(len(m["content"]) for m in request.messages)
    return chars // CHARS_PER_TOKEN + request.max_tokens


def _rate(tokens: int, seconds: float) -> float:
    return round(tokens / seconds, 1) if seconds > 0 else 0.0


class ChatService:
    def __init__(
        self,
        settings: Settings,
        router: Router,
        index: PrefixIndex,
        vllm: VllmBackend,
        bedrock: BedrockBackend | None,
        stats: Stats,
        metrics: CloudWatchMetrics | None = None,
    ) -> None:
        self._settings = settings
        self._router = router
        self._index = index
        self._vllm = vllm
        self._bedrock = bedrock
        self._stats = stats
        self._metrics = metrics
        model = settings.model
        self._throughput = Throughput(model.prefill_tps, model.decode_tps)

    async def compare(
        self, requests: Mapping[str, ChatRequest], workload: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Run every requested target concurrently and interleave their events."""
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        runners = {SELFHOST: self.selfhost, BEDROCK: self.bedrock}

        async def pump(target: str, request: ChatRequest) -> None:
            try:
                async for event in runners[target](request, workload):
                    await queue.put(event)
            except Exception:  # never leave a target without a final event
                LOGGER.exception("%s failed unexpectedly", target)
                self._stats.record_error(target)
                await queue.put({"target": target, "type": "error", "message": "internal error"})
            finally:
                await queue.put(None)

        tasks = [asyncio.create_task(pump(t, r)) for t, r in requests.items()]
        remaining = len(tasks)
        try:
            while remaining:
                try:
                    event = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
                except WAIT_TIMEOUT:  # keeps CloudFront's 60 s origin read timeout from firing
                    yield {"type": "ping"}
                    continue
                if event is None:
                    remaining -= 1
                else:
                    yield event
        finally:
            for task in tasks:
                task.cancel()

    async def selfhost(self, request: ChatRequest, workload: str) -> AsyncIterator[dict[str, Any]]:
        model = self._settings.model
        hashes = block_hashes(
            openai_messages(request), model=model.key, tokenizer_version=model.tokenizer_version
        )
        tried: set[str] = set()
        while True:
            try:
                decision = self._router.route(hashes, _estimate_tokens(request), frozenset(tried))
            except Overloaded:
                self._stats.record_error(SELFHOST)
                yield {
                    "target": SELFHOST,
                    "type": "error",
                    "message": "The self-hosted model is busy or still loading; try again shortly.",
                }
                return
            except NoReplicaForContext as exc:
                self._stats.record_error(SELFHOST)
                yield {"target": SELFHOST, "type": "error", "message": str(exc)}
                return
            name = decision.replica.name
            streamed = False
            try:
                with self._router.lease(name):
                    busy_start = self._router.busy_integral()
                    async for event in self._vllm.stream(decision.replica.url, request):
                        if isinstance(event, Delta):
                            streamed = True
                            yield {"target": SELFHOST, "type": "delta", "text": event.text}
                        elif isinstance(event, Reasoning):
                            streamed = True
                            yield {"target": SELFHOST, "type": "reasoning", "text": event.text}
                        elif isinstance(event, Final):
                            busy = self._router.busy_integral() - busy_start
                            yield self._selfhost_done(event, decision, busy)
                self._index.record(hashes, name, workload)
                return
            except ReplicaUnavailable as exc:
                self._router.mark_unhealthy(name)
                tried.add(name)
                if streamed:
                    self._stats.record_error(SELFHOST)
                    yield {"target": SELFHOST, "type": "error", "message": str(exc)}
                    return
                LOGGER.warning("replica %s unavailable, retrying elsewhere: %s", name, exc)
            except BackendError as exc:
                self._stats.record_error(SELFHOST)
                yield {"target": SELFHOST, "type": "error", "message": str(exc)}
                return

    def _selfhost_done(self, final: Final, decision: Any, busy_integral: float) -> dict[str, Any]:
        settings = self._settings
        usage = final.usage
        concurrency = (
            max(1.0, busy_integral / final.elapsed_seconds) if final.elapsed_seconds else 1.0
        )
        actual = selfhost_actual_cost(
            final.elapsed_seconds, concurrency, settings.instance_usd_per_hour
        )
        if settings.model.engine == "llama.cpp":
            # CPU slots gain little from batching, so the machine time this answer measurably
            # took is its capacity; only vLLM's batching makes the throughput model apply.
            capacity = actual / settings.target_utilization
        else:
            capacity = selfhost_capacity_cost(
                usage, self._throughput, settings.instance_usd_per_hour, settings.target_utilization
            )
        metrics = self._common_metrics(final) | {
            "cost_usd": actual,
            "capacity_cost_usd": capacity,
            "bedrock_equivalent_usd": bedrock_cost(
                usage, settings.model.input_per_million, settings.model.output_per_million
            ),
            "avg_concurrency": round(concurrency, 2),
            "replica": decision.replica.name,
            "prefix_blocks_matched": decision.matched_blocks,
            "prefix_blocks_total": decision.total_blocks,
        }
        self._record(SELFHOST, final, actual, capacity)
        return {"target": SELFHOST, "type": "done", "metrics": metrics}

    async def bedrock(self, request: ChatRequest, workload: str) -> AsyncIterator[dict[str, Any]]:
        del workload  # Bedrock on-demand has no workload-specific retention
        if self._bedrock is None:
            yield {"target": BEDROCK, "type": "error", "message": "Bedrock is not configured"}
            return
        try:
            async for event in self._bedrock.stream(request):
                if isinstance(event, Delta):
                    yield {"target": BEDROCK, "type": "delta", "text": event.text}
                elif isinstance(event, Reasoning):
                    yield {"target": BEDROCK, "type": "reasoning", "text": event.text}
                elif isinstance(event, Final):
                    model = self._settings.model
                    cost = bedrock_cost(
                        event.usage, model.input_per_million, model.output_per_million
                    )
                    self._record(BEDROCK, event, cost, cost)
                    metrics = self._common_metrics(event) | {
                        "cost_usd": cost,
                        "model_id": model.bedrock_model_id,
                    }
                    yield {"target": BEDROCK, "type": "done", "metrics": metrics}
        except BackendError as exc:
            self._stats.record_error(BEDROCK)
            yield {"target": BEDROCK, "type": "error", "message": str(exc)}

    @staticmethod
    def _common_metrics(final: Final) -> dict[str, Any]:
        usage = final.usage
        generation = max(0.0, final.elapsed_seconds - final.ttft_seconds)
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": usage.cached_tokens,
            "ttft_ms": round(final.ttft_seconds * 1000, 1),
            "elapsed_ms": round(final.elapsed_seconds * 1000, 1),
            "output_tokens_per_second": _rate(usage.output_tokens, generation),
        }

    def _record(self, target: str, final: Final, cost: float, capacity_cost: float) -> None:
        usage = final.usage
        self._stats.record(
            target,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            cost_usd=cost,
            capacity_cost_usd=capacity_cost,
            ttft_seconds=final.ttft_seconds,
            elapsed_seconds=final.elapsed_seconds,
        )
        if self._metrics is None:
            return
        generation = max(1e-6, final.elapsed_seconds - final.ttft_seconds)
        output = max(1, usage.output_tokens)
        self._metrics.put(
            target,
            self._settings.model.key,
            {
                "TimeToFirstToken": (final.ttft_seconds * 1000, "Milliseconds"),
                "InterTokenLatency": (generation * 1000 / output, "Milliseconds"),
                "OutputTokensPerSecond": (usage.output_tokens / generation, "Count/Second"),
                "InputTokens": (usage.input_tokens, "Count"),
                "CachedInputTokens": (usage.cached_tokens, "Count"),
                "CostUSD": (cost, "None"),
                "CostPerMillionTokensUSD": (
                    cost * 1_000_000 / max(1, usage.input_tokens + usage.output_tokens),
                    "None",
                ),
            },
        )
