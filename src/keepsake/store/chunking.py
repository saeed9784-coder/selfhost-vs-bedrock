"""Paragraph chunking with a SHA-256 per chunk (FR2.7).

A document is stored as ordered paragraph sections, so a one-paragraph edit changes only that
paragraph's hash; unchanged chunks keep their hashes across versions.
"""

from __future__ import annotations

import hashlib
import re

from keepsake.common.models import Chunk

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_into_chunks(text: str) -> tuple[Chunk, ...]:
    """Split on blank lines; strip each paragraph; drop empty ones."""
    normalized = text.replace("\r\n", "\n")
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(normalized)]
    return tuple(
        Chunk(index=index, text=paragraph, sha256=chunk_hash(paragraph))
        for index, paragraph in enumerate(p for p in paragraphs if p)
    )


def join_chunks(chunks: tuple[Chunk, ...]) -> str:
    return "\n\n".join(chunk.text for chunk in sorted(chunks, key=lambda c: c.index))


def changed_chunk_indexes(old: tuple[str, ...], new: tuple[str, ...]) -> tuple[int, ...]:
    """Positions whose hash differs between two versions' ordered chunk-hash lists.

    Positions present in only one version (added or removed at the end) count as changed.
    """
    length = max(len(old), len(new))
    return tuple(
        index
        for index in range(length)
        if index >= len(old) or index >= len(new) or old[index] != new[index]
    )
