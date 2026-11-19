"""A stand-in for vLLM's OpenAI server, for running the gateway and chat UI without a GPU.

It streams a canned answer and imitates prefix caching: the cached token count is the longest
prompt prefix seen before. Timings are simulated, so its costs mean nothing; it exists to
exercise the gateway, the router and the UI.

    uvicorn selfhost.devserver:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

CHARS_PER_TOKEN = 4
BLOCK_TOKENS = 16
TOKEN_DELAY_SECONDS = 0.02
MAX_REMEMBERED = 500

app = FastAPI(title="vLLM stand-in")
_seen: list[str] = []
_running = 0


def _cached_tokens(prompt: str) -> int:
    best = 0
    for previous in _seen:
        limit = min(len(previous), len(prompt))
        length = next((i for i in range(limit) if previous[i] != prompt[i]), limit)
        best = max(best, length)
    tokens = best // CHARS_PER_TOKEN
    return tokens - tokens % BLOCK_TOKENS


def _answer(question: str) -> list[str]:
    text = (
        "This answer comes from the local vLLM stand-in, not a real model. "
        f'You asked: "{question[:120]}". On the deployed stack the same request is served by '
        "vLLM on the GPU node, and repeated conversation history is read from the KV cache "
        "instead of being processed again."
    )
    return [word + " " for word in text.split(" ")]


@app.post("/v1/chat/completions")
async def completions(request: Request) -> StreamingResponse:
    body = await request.json()
    messages = body.get("messages", [])
    prompt = json.dumps(messages, separators=(",", ":"))
    cached = _cached_tokens(prompt)
    _seen.append(prompt)
    del _seen[:-MAX_REMEMBERED]
    question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    words = _answer(question)[: int(body.get("max_tokens", 512))]

    async def stream() -> AsyncIterator[bytes]:
        global _running
        _running += 1
        try:
            prompt_tokens = len(prompt) // CHARS_PER_TOKEN
            await asyncio.sleep(0.05 + (prompt_tokens - cached) / 20000)  # simulated prefill
            for word in words:
                yield _chunk({"choices": [{"index": 0, "delta": {"content": word}}]})
                await asyncio.sleep(TOKEN_DELAY_SECONDS)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(words),
                "prompt_tokens_details": {"cached_tokens": cached},
            }
            yield _chunk({"choices": [], "usage": usage})
            yield b"data: [DONE]\n\n"
        finally:
            _running -= 1

    return StreamingResponse(stream(), media_type="text/event-stream")


def _chunk(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps({'created': int(time.time()), **payload})}\n\n".encode()


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        f"vllm:num_requests_running {_running}\nvllm:num_requests_waiting 0\n"
        "vllm:kv_cache_usage_perc 0.1\n"
    )
