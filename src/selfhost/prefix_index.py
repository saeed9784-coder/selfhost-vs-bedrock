"""Cache-aware routing metadata: which replica most likely holds a prompt's prefix in KV cache.

The gateway cannot see vLLM's token blocks, so it hashes the prompt in fixed-size character
blocks, each hash chained to the one before (like vLLM's own prefix hashing). Keys include the
model and tokenizer version, so a model upgrade never reuses stale entries. Entries expire after
the retention set for their workload class (agent or chat).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

BLOCK_CHARS = 1024  # about 256 tokens; coarse on purpose, the router only needs a hint
MAX_ENTRIES = 200_000


@dataclass(frozen=True)
class PrefixEntry:
    replica: str
    expires_at: float


def block_hashes(
    messages: Sequence[dict[str, str]], *, model: str, tokenizer_version: str
) -> tuple[str, ...]:
    """Chained hashes over the serialized conversation, one per full block."""
    text = json.dumps(list(messages), ensure_ascii=False, separators=(",", ":"))
    parent = hashlib.sha256(f"{model}|{tokenizer_version}".encode()).hexdigest()
    hashes = []
    for start in range(0, len(text) - BLOCK_CHARS + 1, BLOCK_CHARS):
        block = text[start : start + BLOCK_CHARS]
        parent = hashlib.sha256(f"{parent}|{block}".encode()).hexdigest()
        hashes.append(parent)
    return tuple(hashes)


class PrefixIndex:
    """In-memory prefix-hash to replica map with per-workload TTL and bounded size."""

    def __init__(
        self, retention_hours: dict[str, float], clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._retention = dict(retention_hours)
        self._clock = clock
        self._entries: dict[str, PrefixEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def matched_blocks(self, hashes: Sequence[str], replica: str) -> int:
        """Number of leading blocks this replica is believed to hold."""
        now = self._clock()
        count = 0
        for digest in hashes:
            entry = self._entries.get(digest)
            if entry is None or entry.replica != replica or entry.expires_at <= now:
                break
            count += 1
        return count

    def record(self, hashes: Sequence[str], replica: str, workload: str) -> None:
        ttl = self._retention.get(workload, self._retention.get("chat", 24.0)) * 3600
        expires_at = self._clock() + ttl
        for digest in hashes:
            self._entries[digest] = PrefixEntry(replica=replica, expires_at=expires_at)
        if len(self._entries) > MAX_ENTRIES:
            self.evict_expired()
        while len(self._entries) > MAX_ENTRIES:  # still full: drop oldest-inserted first
            self._entries.pop(next(iter(self._entries)))

    def evict_expired(self) -> int:
        now = self._clock()
        expired = [k for k, v in self._entries.items() if v.expires_at <= now]
        for key in expired:
            del self._entries[key]
        return len(expired)
