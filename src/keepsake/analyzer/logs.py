"""Read Bedrock model-invocation-logging records (FR1.1).

Bedrock writes JSON records (optionally gzipped, one record per line) to S3. Each record has
`timestamp`, `requestId`, `modelId`, `identity.arn`, `input.inputBodyJson` and
`input.inputTokenCount`. Both Converse bodies (`system` list, `messages[].content[].text`) and
InvokeModel Anthropic bodies (`system` string, `messages[].content` string or blocks) are read.

The agent is `requestMetadata.agent` when present, else the IAM role name of the caller.
Malformed lines are counted and skipped (never silently): see `ParseResult.skipped`.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from keepsake.common.aws import AWS_ERRORS, translate_aws_error
from keepsake.common.clock import parse_iso
from keepsake.common.config import role_name_from_arn
from keepsake.common.errors import AccessDenied, ValidationError

LOGGER = logging.getLogger(__name__)
MAX_LOG_OBJECTS = 10_000


@dataclass(frozen=True)
class InvocationRecord:
    request_id: str
    agent_id: str
    account_id: str
    model_id: str
    timestamp: datetime
    texts: tuple[str, ...]
    input_tokens: int


@dataclass(frozen=True)
class ParseResult:
    records: tuple[InvocationRecord, ...]
    skipped: int


def _texts_from_content(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
    return []


def extract_texts(body: dict[str, Any]) -> tuple[str, ...]:
    """Every text string in a Converse or Anthropic InvokeModel request body, in order."""
    texts = _texts_from_content(body.get("system"))
    for message in body.get("messages", []) or []:
        if isinstance(message, dict):
            texts.extend(_texts_from_content(message.get("content")))
    return tuple(t for t in texts if t.strip())


def _agent_of(record: dict[str, Any]) -> str:
    metadata = record.get("requestMetadata") or {}
    if isinstance(metadata, dict) and metadata.get("agent"):
        return str(metadata["agent"])
    arn = (record.get("identity") or {}).get("arn", "")
    try:
        return role_name_from_arn(arn)
    except AccessDenied:
        return arn or "unknown"


def parse_record(record: dict[str, Any]) -> InvocationRecord:
    """Raise KeyError/ValueError/TypeError on a malformed record."""
    body = record["input"].get("inputBodyJson") or {}
    texts = extract_texts(body)
    token_count = record["input"].get("inputTokenCount")
    tokens = int(token_count) if token_count is not None else sum(len(t) for t in texts) // 4
    identity_arn = str((record.get("identity") or {}).get("arn", ""))
    arn_parts = identity_arn.split(":")
    account_id = arn_parts[4] if len(arn_parts) > 4 and arn_parts[4].isdigit() else "unknown"
    return InvocationRecord(
        request_id=str(record["requestId"]),
        agent_id=_agent_of(record),
        account_id=account_id,
        model_id=str(record["modelId"]),
        timestamp=parse_iso(str(record["timestamp"])),
        texts=texts,
        input_tokens=tokens,
    )


def parse_lines(lines: Iterable[str]) -> ParseResult:
    records: list[InvocationRecord] = []
    skipped = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(parse_record(json.loads(line)))
        except (KeyError, ValueError, TypeError, AttributeError):
            skipped += 1
    if skipped:
        LOGGER.warning("skipped malformed invocation-log lines", extra={"skipped": skipped})
    return ParseResult(records=tuple(sorted(records, key=lambda r: r.timestamp)), skipped=skipped)


def _decode(data: bytes, name: str) -> Iterator[str]:
    raw = gzip.decompress(data) if name.endswith(".gz") else data
    yield from raw.decode("utf-8").splitlines()


def load_from_paths(paths: Iterable[Path]) -> ParseResult:
    lines: list[str] = []
    for path in paths:
        try:
            lines.extend(_decode(path.read_bytes(), path.name))
        except OSError as exc:
            raise FileNotFoundError(f"cannot read invocation log {path}: {exc}") from exc
    return parse_lines(lines)


def load_from_s3(s3: Any, bucket: str, prefix: str) -> ParseResult:
    """Read every object under `prefix` (bounded by MAX_LOG_OBJECTS)."""
    lines: list[str] = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
            if len(keys) > MAX_LOG_OBJECTS:
                raise ValidationError(
                    f"log prefix contains more than {MAX_LOG_OBJECTS} objects; narrow the prefix "
                    "to the intended observation window"
                )
        for key in keys:
            data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            lines.extend(_decode(data, key))
    except AWS_ERRORS as exc:
        raise translate_aws_error(exc, "read invocation logs") from exc
    return parse_lines(lines)
