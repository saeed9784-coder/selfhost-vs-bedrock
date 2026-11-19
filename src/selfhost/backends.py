"""Streaming clients for the two backends, normalized to the same events.

Both yield ``Delta`` events while text arrives and finish with one ``Final`` carrying token
usage and timings. Errors are raised as ``BackendError`` (or ``ReplicaUnavailable`` when a
vLLM replica could not be reached before any token was produced, so the caller may retry).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from selfhost.pricing import Usage

LOGGER = logging.getLogger("selfhost.backends")
_DONE = object()
REASONING_HEADROOM = 1024
MAX_REASONING_BUDGET = 8192


@dataclass(frozen=True)
class Delta:
    text: str


@dataclass(frozen=True)
class Final:
    usage: Usage
    ttft_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class Reasoning:
    """Thinking text from a reasoning model; billed as output tokens, shown separately."""

    text: str


Event = Delta | Reasoning | Final


def token_budget(answer_tokens: int, reasoning: bool) -> int:
    """Reasoning models spend output tokens thinking first; without headroom they can use the
    whole budget before writing any answer. Both sides get the same budget; it is billed."""
    if not reasoning:
        return answer_tokens
    return min(MAX_REASONING_BUDGET, answer_tokens + max(REASONING_HEADROOM, answer_tokens))


class BackendError(Exception):
    """The backend failed; the message is safe to show to the user."""


class ReplicaUnavailable(BackendError):
    """The replica could not be reached; nothing was streamed yet."""


@dataclass(frozen=True)
class ChatRequest:
    system: str
    messages: tuple[dict[str, str], ...]
    max_tokens: int
    temperature: float


def openai_messages(request: ChatRequest) -> list[dict[str, str]]:
    head = [{"role": "system", "content": request.system}] if request.system else []
    return head + [dict(m) for m in request.messages]


class VllmBackend:
    """vLLM's OpenAI-compatible server (``--enable-prompt-tokens-details`` for cached tokens)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        served_model: str,
        extra_body: dict[str, Any] | None = None,
        reasoning: bool = False,
    ) -> None:
        self._client = client
        self._model = served_model
        self._extra = dict(extra_body or {})
        self._reasoning = reasoning

    async def stream(self, base_url: str, request: ChatRequest) -> AsyncIterator[Event]:
        payload = {
            "model": self._model,
            "messages": openai_messages(request),
            "max_tokens": token_budget(request.max_tokens, self._reasoning),
            "temperature": request.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._extra,
        }
        start = time.perf_counter()
        first_token: float | None = None
        usage: Usage | None = None
        try:
            async with self._client.stream(
                "POST", f"{base_url}/v1/chat/completions", json=payload
            ) as response:
                if response.status_code >= 500:
                    raise ReplicaUnavailable(f"replica returned HTTP {response.status_code}")
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")[:300]
                    raise BackendError(f"vLLM returned HTTP {response.status_code}: {body}")
                async for line in response.aiter_lines():
                    chunk = _sse_json(line)
                    if chunk is None:
                        continue
                    text, thinking = _openai_delta(chunk)
                    if thinking:
                        first_token = first_token or time.perf_counter()
                        yield Reasoning(thinking)
                    if text:
                        first_token = first_token or time.perf_counter()
                        yield Delta(text)
                    if chunk.get("usage"):
                        usage = _openai_usage(chunk["usage"], chunk.get("timings"))
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ReplicaUnavailable(f"cannot reach vLLM replica: {exc}") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"vLLM stream failed: {exc}") from exc
        end = time.perf_counter()
        yield Final(
            usage=usage or Usage(0, 0),
            ttft_seconds=(first_token or end) - start,
            elapsed_seconds=end - start,
        )


def _sse_json(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _openai_delta(chunk: dict[str, Any]) -> tuple[str, str]:
    """(answer text, reasoning text); llama.cpp and vLLM send thinking as reasoning_content."""
    choices = chunk.get("choices") or []
    if not choices:
        return "", ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or "", delta.get("reasoning_content") or ""


def _openai_usage(raw: dict[str, Any], timings: dict[str, Any] | None = None) -> Usage:
    """vLLM reports cached prompt tokens in usage; llama.cpp reports them as timings.cache_n."""
    details = raw.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or (timings or {}).get("cache_n") or 0
    return Usage(
        input_tokens=int(raw.get("prompt_tokens") or 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cached_tokens=int(cached),
    )


def bedrock_messages(messages: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    return [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]


class BedrockBackend:
    """Amazon Bedrock ConverseStream; boto3 is synchronous, so it runs on a worker thread."""

    def __init__(self, runtime_client: Any, model_id: str, reasoning: bool = False) -> None:
        self._client = runtime_client
        self._model_id = model_id
        self._reasoning = reasoning

    async def stream(self, request: ChatRequest) -> AsyncIterator[Event]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()
        stop = threading.Event()
        start = time.perf_counter()

        def put(item: object) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            try:
                kwargs: dict[str, Any] = {
                    "modelId": self._model_id,
                    "messages": bedrock_messages(request.messages),
                    "inferenceConfig": {
                        "maxTokens": token_budget(request.max_tokens, self._reasoning),
                        "temperature": request.temperature,
                    },
                }
                if request.system:
                    kwargs["system"] = [{"text": request.system}]
                response = self._client.converse_stream(**kwargs)
                for event in response["stream"]:
                    if stop.is_set():
                        break
                    put(event)
            except Exception as exc:  # boto/botocore errors of many types end up here
                put(exc)
            finally:
                put(_DONE)

        threading.Thread(target=worker, name="bedrock-stream", daemon=True).start()
        first_token: float | None = None
        usage = Usage(0, 0)
        try:
            while (item := await queue.get()) is not _DONE:
                if isinstance(item, Exception):
                    raise BackendError(f"Bedrock call failed: {_short_error(item)}")
                text, thinking, maybe_usage = _bedrock_event(item)
                if thinking:
                    first_token = first_token or time.perf_counter()
                    yield Reasoning(thinking)
                if text:
                    first_token = first_token or time.perf_counter()
                    yield Delta(text)
                usage = maybe_usage or usage
        finally:
            stop.set()
        end = time.perf_counter()
        yield Final(
            usage=usage, ttft_seconds=(first_token or end) - start, elapsed_seconds=end - start
        )


def _bedrock_event(event: Any) -> tuple[str, str, Usage | None]:
    """(answer text, reasoning text, usage) from one ConverseStream event."""
    if not isinstance(event, dict):
        return "", "", None
    delta = event.get("contentBlockDelta", {}).get("delta", {})
    raw = event.get("metadata", {}).get("usage")
    usage = None
    if raw:
        usage = Usage(
            input_tokens=int(raw.get("inputTokens") or 0),
            output_tokens=int(raw.get("outputTokens") or 0),
            cached_tokens=int(raw.get("cacheReadInputTokens") or 0),
        )
    thinking = (delta.get("reasoningContent") or {}).get("text") or ""
    return delta.get("text") or "", thinking, usage


def _short_error(exc: Exception) -> str:
    """Error code only for the client; the full message (it may name roles/accounts) is logged."""
    LOGGER.warning("Bedrock error: %s", exc)
    response = getattr(exc, "response", None)
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    return f"{code} (details in the gateway log)" if code else type(exc).__name__
