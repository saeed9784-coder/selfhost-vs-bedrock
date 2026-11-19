"""Deterministic sample invocation logs modelled on the SOW (FR1.6).

No real Groundhog Labs logs exist, so Phase 0 runs on generated traffic in the Bedrock
model-invocation-logging format:
- Helpdesk (DeepSeek V3.2): a 100,000-token support manual sent 50 times over 3 days.
- Reviewer (Claude): a 30,000-token repository snapshot sent across PR revisions; each revision
  changes one paragraph; 5 PRs, 4 revisions each, over 5 days.
- Atlas (Nova): 6 product documents of 5,000 tokens; 30 requests over 10 days, 2 documents each.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

CHARS_PER_TOKEN = 4
ACCOUNT = "111122223333"
HELPDESK_MODEL = "deepseek.v3.2"
REVIEWER_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
ATLAS_MODEL = "us.amazon.nova-pro-v1:0"
DEFAULT_START = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
_WORDS = [
    "account",
    "billing",
    "refund",
    "ticket",
    "escalation",
    "policy",
    "customer",
    "service",
    "response",
    "warranty",
    "shipping",
    "order",
    "invoice",
    "credit",
    "request",
    "agent",
    "priority",
    "tier",
    "support",
    "handbook",
    "procedure",
    "review",
    "approval",
    "release",
    "deploy",
    "module",
    "function",
    "class",
    "service",
    "handler",
    "config",
    "cache",
    "token",
    "model",
    "context",
    "document",
    "version",
    "product",
    "feature",
    "roadmap",
    "launch",
    "metric",
    "dashboard",
    "security",
    "privacy",
    "retention",
    "region",
    "storage",
    "network",
    "latency",
    "throughput",
    "incident",
    "runbook",
]


def _paragraph(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(words)).capitalize() + "."


def _document(rng: random.Random, tokens: int, paragraph_words: int = 120) -> list[str]:
    target_chars = tokens * CHARS_PER_TOKEN
    paragraphs: list[str] = []
    while sum(len(p) + 2 for p in paragraphs) < target_chars:
        paragraphs.append(_paragraph(rng, paragraph_words))
    return paragraphs


def _record(
    rng: random.Random,
    agent: str,
    model: str,
    when: datetime,
    system: str,
    context: str,
    question: str,
) -> dict[str, Any]:
    text = f"{context}\n\n{question}"
    return {
        "schemaType": "ModelInvocationLog",
        "schemaVersion": "1.0",
        "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "accountId": ACCOUNT,
        "identity": {"arn": f"arn:aws:sts::{ACCOUNT}:assumed-role/keepsake-agent-{agent}/run"},
        "region": "us-east-1",
        "requestId": str(uuid.UUID(int=rng.getrandbits(128))),
        "operation": "Converse",
        "modelId": model,
        "requestMetadata": {"agent": agent},
        "input": {
            "inputContentType": "application/json",
            "inputBodyJson": {
                "system": [{"text": system}],
                "messages": [{"role": "user", "content": [{"text": text}]}],
            },
            "inputTokenCount": (len(system) + len(text)) // CHARS_PER_TOKEN,
        },
        "output": {"outputContentType": "application/json", "outputTokenCount": 250},
    }


def helpdesk_records(
    rng: random.Random,
    start: datetime,
    manual_tokens: int = 100_000,
    calls: int = 50,
    days: int = 3,
) -> list[dict[str, Any]]:
    manual = "\n\n".join(_document(rng, manual_tokens))
    step = timedelta(days=days) / calls
    return [
        _record(
            rng,
            "helpdesk",
            HELPDESK_MODEL,
            start + step * i,
            "You answer Groundhog Labs support questions using the manual.",
            manual,
            f"Customer {i}: {_paragraph(rng, 25)}",
        )
        for i in range(calls)
    ]


def reviewer_records(rng: random.Random, start: datetime) -> list[dict[str, Any]]:
    snapshot = _document(rng, 30_000)
    records = []
    for pr in range(5):
        for revision in range(4):
            changed = rng.randrange(len(snapshot))
            snapshot = [*snapshot[:changed], _paragraph(rng, 120), *snapshot[changed + 1 :]]
            when = start + timedelta(days=pr, hours=revision * 3)
            records.append(
                _record(
                    rng,
                    "reviewer",
                    REVIEWER_MODEL,
                    when,
                    "You review pull requests against the repository snapshot.",
                    "\n\n".join(snapshot),
                    f"Review PR {pr} revision {revision}.",
                )
            )
    return records


def atlas_records(rng: random.Random, start: datetime) -> list[dict[str, Any]]:
    documents = ["\n\n".join(_document(rng, 5_000)) for _ in range(6)]
    step = timedelta(days=10) / 30
    return [
        _record(
            rng,
            "atlas",
            ATLAS_MODEL,
            start + step * i,
            "You are Atlas, the internal knowledge assistant.",
            "\n\n".join(rng.sample(documents, 2)),
            f"Question {i}: {_paragraph(rng, 20)}",
        )
        for i in range(30)
    ]


def generate(seed: int = 7, start: datetime = DEFAULT_START) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = [
        *helpdesk_records(rng, start),
        *reviewer_records(rng, start),
        *atlas_records(rng, start),
    ]
    return sorted(records, key=lambda r: r["timestamp"])


def to_lines(records: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(record) for record in records]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(to_lines(records)) + "\n", encoding="utf-8")
    return path
