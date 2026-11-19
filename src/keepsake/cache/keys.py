"""Deterministic response-reuse keys (FR4.2, FR4.5).

The key is a SHA-256 over a canonical JSON of: agent, customer, model id, model version, the
system prompt, the sorted `document@version` list and the normalized conversation. Changing any
of them, in particular the model id or version or any document version, produces a different
key, so an entry for an old model or document version can never be served.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from keepsake.common.models import Message


def normalize_text(text: str) -> str:
    """Collapse runs of whitespace and trim, so formatting-only differences still match."""
    return " ".join(text.split())


def compute_reuse_key(
    *,
    agent_id: str,
    customer_id: str,
    model_id: str,
    model_version: str,
    system_prompt: str,
    document_versions: Iterable[str],
    messages: Iterable[Message],
) -> str:
    canonical = {
        "agent": agent_id,
        "customer": customer_id,
        "model_id": model_id,
        "model_version": model_version,
        "system": normalize_text(system_prompt),
        "documents": sorted(document_versions),
        "messages": [[m.role, normalize_text(m.text)] for m in messages],
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
