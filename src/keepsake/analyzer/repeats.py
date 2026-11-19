"""Repeated-context detection per agent (FR1.1, FR1.2, FR1.3).

Each request's text is cut into segments: paragraphs (blank-line separated), with paragraphs
longer than WINDOW_CHARS cut into fixed windows so unbroken text is still compared. A segment
counts as repeated when the same agent sent an identical segment (same SHA-256) earlier. The
gap since that earlier send is bucketed into < 1 h, < 1 d, < 3 d, < 7 d and >= 7 d.

Characters are converted to tokens with each request's own `inputTokenCount`, so repeated
token figures stay consistent with what Bedrock billed.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from keepsake.analyzer.logs import InvocationRecord
from keepsake.common.errors import ValidationError

WINDOW_CHARS = 2000
MIN_SEGMENT_CHARS = 64
GAP_BUCKETS: tuple[tuple[str, float], ...] = (
    ("<1h", 3600),
    ("<1d", 86_400),
    ("<3d", 259_200),
    ("<7d", 604_800),
    (">=7d", float("inf")),
)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class RequestRepeat:
    request_id: str
    timestamp: datetime
    input_tokens: int
    repeated_tokens: int
    repeated_share: float


@dataclass(frozen=True)
class AgentRepeatStats:
    agent_id: str
    model_id: str
    request_count: int
    total_input_tokens: int
    repeated_tokens: int
    repeated_share: float
    mean_repeated_tokens: float
    max_repeated_tokens: int
    unique_repeated_tokens: int
    gap_tokens: tuple[tuple[str, int], ...]
    observation_hours: float
    requests: tuple[RequestRepeat, ...]

    def gap(self, bucket: str) -> int:
        return dict(self.gap_tokens)[bucket]


def segments(text: str) -> tuple[str, ...]:
    pieces: list[str] = []
    for paragraph in (p.strip() for p in _PARAGRAPH_BREAK.split(text)):
        for start in range(0, len(paragraph), WINDOW_CHARS):
            piece = paragraph[start : start + WINDOW_CHARS]
            if len(piece) >= MIN_SEGMENT_CHARS:
                pieces.append(piece)
    return tuple(pieces)


def gap_bucket(seconds: float) -> str:
    for name, upper in GAP_BUCKETS:
        if seconds < upper:
            return name
    return GAP_BUCKETS[-1][0]


@dataclass
class _AgentAccumulator:
    """Mutable working state for one agent's scan; converted to frozen stats at the end."""

    last_seen: dict[str, datetime]
    repeated_segment_chars: dict[str, int]
    gap_tokens: Counter[str]
    requests: list[RequestRepeat]
    models: Counter[str]


def _scan_record(acc: _AgentAccumulator, record: InvocationRecord) -> None:
    all_segments = [s for text in record.texts for s in segments(text)]
    total_chars = sum(len(t) for t in record.texts) or 1
    tokens_per_char = record.input_tokens / total_chars
    repeated_chars = 0
    for segment in all_segments:
        digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
        previous = acc.last_seen.get(digest)
        if previous is not None:
            repeated_chars += len(segment)
            acc.repeated_segment_chars[digest] = len(segment)
            gap_seconds = (record.timestamp - previous).total_seconds()
            acc.gap_tokens[gap_bucket(gap_seconds)] += round(len(segment) * tokens_per_char)
        acc.last_seen[digest] = record.timestamp
    repeated_tokens = min(record.input_tokens, round(repeated_chars * tokens_per_char))
    share = repeated_tokens / record.input_tokens if record.input_tokens else 0.0
    acc.requests.append(
        RequestRepeat(
            record.request_id, record.timestamp, record.input_tokens, repeated_tokens, share
        )
    )
    acc.models[record.model_id] += 1


def _finish(agent_id: str, acc: _AgentAccumulator, tokens_per_char: float) -> AgentRepeatStats:
    total = sum(r.input_tokens for r in acc.requests)
    repeated = sum(r.repeated_tokens for r in acc.requests)
    with_repeats = [r.repeated_tokens for r in acc.requests if r.repeated_tokens]
    first, last = acc.requests[0].timestamp, acc.requests[-1].timestamp
    return AgentRepeatStats(
        agent_id=agent_id,
        model_id=acc.models.most_common(1)[0][0],
        request_count=len(acc.requests),
        total_input_tokens=total,
        repeated_tokens=repeated,
        repeated_share=repeated / total if total else 0.0,
        mean_repeated_tokens=sum(with_repeats) / len(with_repeats) if with_repeats else 0.0,
        max_repeated_tokens=max(with_repeats, default=0),
        unique_repeated_tokens=round(sum(acc.repeated_segment_chars.values()) * tokens_per_char),
        gap_tokens=tuple((name, acc.gap_tokens.get(name, 0)) for name, _ in GAP_BUCKETS),
        observation_hours=(last - first).total_seconds() / 3600,
        requests=tuple(acc.requests),
    )


def analyze(records: Iterable[InvocationRecord]) -> tuple[AgentRepeatStats, ...]:
    """Per-agent and per-model repeated-context statistics."""
    by_agent_model: dict[tuple[str, str], list[InvocationRecord]] = defaultdict(list)
    for record in records:
        by_agent_model[(record.agent_id, record.model_id)].append(record)
    results = []
    for agent_id, model_id in sorted(by_agent_model):
        ordered = sorted(by_agent_model[(agent_id, model_id)], key=lambda r: r.timestamp)
        acc = _AgentAccumulator({}, {}, Counter(), [], Counter())
        for record in ordered:
            _scan_record(acc, record)
        total_chars = sum(len(t) for r in ordered for t in r.texts) or 1
        tokens_per_char = sum(r.input_tokens for r in ordered) / total_chars
        results.append(_finish(agent_id, acc, tokens_per_char))
    return tuple(results)


def analyze_by_model(records: Iterable[InvocationRecord]) -> tuple[AgentRepeatStats, ...]:
    """Measure common prompt-prefix chunks across agents for each model in one account."""
    records = tuple(records)
    account_ids = {record.account_id for record in records}
    if len(account_ids) > 1:
        raise ValidationError(
            "model-level reuse analysis must contain logs from one AWS account at a time"
        )
    by_model: dict[str, list[InvocationRecord]] = defaultdict(list)
    for record in records:
        by_model[record.model_id].append(record)
    results = []
    for model_id in sorted(by_model):
        ordered = sorted(by_model[model_id], key=lambda r: r.timestamp)
        acc = _AgentAccumulator({}, {}, Counter(), [], Counter())
        for record in ordered:
            prompt = "\n".join(record.texts)
            chars = len(prompt) or 1
            tokens_per_char = record.input_tokens / chars
            repeated_chars = 0
            rolling_prefix = hashlib.sha256()
            for start in range(0, len(prompt), WINDOW_CHARS):
                chunk = prompt[start : start + WINDOW_CHARS]
                if len(chunk) < MIN_SEGMENT_CHARS:
                    break
                rolling_prefix.update(chunk.encode("utf-8"))
                digest = rolling_prefix.hexdigest()
                previous = acc.last_seen.get(digest)
                if previous is not None:
                    repeated_chars += len(chunk)
                    acc.repeated_segment_chars[digest] = len(chunk)
                    seconds = (record.timestamp - previous).total_seconds()
                    acc.gap_tokens[gap_bucket(seconds)] += round(len(chunk) * tokens_per_char)
                acc.last_seen[digest] = record.timestamp
            repeated_tokens = min(record.input_tokens, round(repeated_chars * tokens_per_char))
            share = repeated_tokens / record.input_tokens if record.input_tokens else 0.0
            acc.requests.append(
                RequestRepeat(
                    record.request_id,
                    record.timestamp,
                    record.input_tokens,
                    repeated_tokens,
                    share,
                )
            )
            acc.models[record.model_id] += 1
        total_chars = sum(len("\n".join(record.texts)) for record in ordered) or 1
        tokens_per_char = sum(record.input_tokens for record in ordered) / total_chars
        results.append(_finish("all-agents", acc, tokens_per_char))
    return tuple(results)
