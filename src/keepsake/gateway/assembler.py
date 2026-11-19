"""Prompt assembly in a fixed, cache-friendly order (FR3.2, FR3.8, FR4.1).

Order: system instructions -> shared documents sorted by id -> provider cache checkpoint (only
when the model supports prompt caching and the stable prefix reaches its minimum token count)
-> customer history -> the request. Keeping the stable part first and identical across calls is
what lets the provider cache serve it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

from keepsake.common.models import Message, ModelCapability

CHARS_PER_TOKEN = 4
CACHE_POINT: dict[str, Any] = {"cachePoint": {"type": "default"}}


@dataclass(frozen=True)
class SharedDocument:
    document_id: str
    version_id: str
    text: str

    @property
    def ref(self) -> str:
        return f"{self.document_id}@{self.version_id}"


@dataclass(frozen=True)
class AssembledPrompt:
    system: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, Any], ...]
    document_versions: tuple[str, ...]
    has_cache_point: bool

    def converse_request(self, model_id: str, max_tokens: int) -> dict[str, Any]:
        request: dict[str, Any] = {
            "modelId": model_id,
            "messages": [dict(m) for m in self.messages],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if self.system:
            request["system"] = [dict(block) for block in self.system]
        return request


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _document_block(document: SharedDocument) -> dict[str, Any]:
    header = (
        f'<document id="{escape(document.document_id)}" '
        f'version="{escape(document.version_id)}">'
    )
    return {"text": f"{header}\n{document.text}\n</document>"}


def _message(message: Message) -> dict[str, Any]:
    return {"role": message.role, "content": [{"text": message.text}]}


def should_place_cache_point(capability: ModelCapability, stable_prefix_tokens: int) -> bool:
    return capability.supports_prompt_cache and stable_prefix_tokens >= capability.min_cache_tokens


def assemble(
    *,
    system_prompt: str,
    documents: Sequence[SharedDocument],
    history: Sequence[Message],
    messages: Sequence[Message],
    capability: ModelCapability,
) -> AssembledPrompt:
    ordered = sorted(documents, key=lambda d: d.document_id)
    system: list[dict[str, Any]] = [{"text": system_prompt}] if system_prompt.strip() else []
    system.extend(_document_block(d) for d in ordered)
    stable_tokens = sum(estimate_tokens(block["text"]) for block in system)
    has_cache_point = bool(ordered) and should_place_cache_point(capability, stable_tokens)
    if has_cache_point:
        system.append(CACHE_POINT)
    return AssembledPrompt(
        system=tuple(system),
        messages=tuple(_message(m) for m in (*history, *messages)),
        document_versions=tuple(d.ref for d in ordered),
        has_cache_point=has_cache_point,
    )
