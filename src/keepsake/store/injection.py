"""Pattern-based prompt-injection scan run on every upload (FR2.8).

A flagged document is quarantined until a staff member approves it. This is a cheap first line
of defence, not a guarantee: it catches common instruction-override, exfiltration, tool-call and
hidden-Unicode patterns. Findings carry a rule id and a short sanitized excerpt.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from keepsake.common.models import Chunk

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore-instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b"
            r".{0,20}\b(instructions?|prompts?|rules?|directions?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role-override",
        re.compile(
            r"(\byou are now\b|\bact as (?:an? )?(?:system|developer|admin)\b"
            r"|^\s*(?:#+\s*)?system\s*:"
            r"|<\s*/?\s*system\s*>|\bnew instructions\s*:)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "reveal-prompt",
        re.compile(
            r"\b(reveal|print|show|repeat|output)\b.{0,30}\b(system prompt|hidden instructions|"
            r"your instructions)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "exfiltration",
        re.compile(
            r"(\b(send|post|upload|forward|exfiltrate)\b.{0,60}\b(https?://|webhook|email to)\b"
            r"|\bcurl\s+-|!\[[^\]]*\]\(https?://[^)]*\?[^)]*=)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "tool-invocation",
        re.compile(
            r"(<\s*tool_call\s*>|\"function_call\"\s*:"
            r"|\b(call|invoke|run|execute) the [\w-]+ tool\b)",
            re.IGNORECASE,
        ),
    ),
)
_HIDDEN_CHARACTERS = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]")
_EXCERPT_CHARS = 80


@dataclass(frozen=True)
class InjectionFinding:
    rule: str
    chunk_index: int
    excerpt: str

    def describe(self) -> str:
        return f"{self.rule} (chunk {self.chunk_index}): {self.excerpt}"


def _sanitize(text: str) -> str:
    visible = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in text)
    collapsed = " ".join(visible.split())
    return collapsed[:_EXCERPT_CHARS]


def scan_chunk(chunk: Chunk) -> tuple[InjectionFinding, ...]:
    findings = [
        InjectionFinding(rule=rule, chunk_index=chunk.index, excerpt=_sanitize(match.group(0)))
        for rule, pattern in _RULES
        if (match := pattern.search(chunk.text))
    ]
    if _HIDDEN_CHARACTERS.search(chunk.text):
        findings.append(
            InjectionFinding(
                rule="hidden-unicode",
                chunk_index=chunk.index,
                excerpt="invisible or bidirectional control characters",
            )
        )
    return tuple(findings)


def scan_chunks(chunks: tuple[Chunk, ...]) -> tuple[InjectionFinding, ...]:
    return tuple(finding for chunk in chunks for finding in scan_chunk(chunk))
