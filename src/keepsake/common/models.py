"""Frozen domain models with validation at construction (FR2.1, FR3.1, FR4.3).

Every model validates its own invariants in `__post_init__`, so an invalid value can never
exist inside the system once it has passed a boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Literal

from keepsake.common.errors import ValidationError

CLASSIFICATIONS = frozenset(
    {"policy", "runbook", "source-code", "contract", "product-doc", "customer-data"}
)
DOCUMENT_STATUSES = frozenset({"active", "retired", "quarantined"})
CACHE_OUTCOMES = frozenset({"provider_hit", "response_reuse", "miss"})
MESSAGE_ROLES = frozenset({"user", "assistant"})

MIN_RETENTION_HOURS = 1
MAX_RETENTION_HOURS = 168  # 7 days (FR4.3)
MAX_DOCUMENT_CHARS = 2_000_000
MAX_MESSAGE_CHARS = 200_000
MAX_MESSAGES = 100
MAX_DOCUMENTS_PER_REQUEST = 20
MAX_TITLE_CHARS = 200
MAX_SYSTEM_PROMPT_CHARS = 20_000

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")

MessageRole = Literal["user", "assistant"]


def require_id(value: Any, field_name: str) -> str:
    """Return `value` if it is a safe identifier (letters, digits, . _ : @ -; max 128)."""
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValidationError(f"{field_name} must be 1-128 characters of [A-Za-z0-9._:@-]")
    return value


def require_text(value: Any, field_name: str, max_chars: int, *, allow_empty: bool = False) -> str:
    """Return `value` if it is a string within `max_chars` (and non-blank unless allowed)."""
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise ValidationError(f"{field_name} must not be empty")
    if len(value) > max_chars:
        raise ValidationError(f"{field_name} exceeds {max_chars} characters")
    return value


def require_classification(value: Any) -> str:
    if value not in CLASSIFICATIONS:
        raise ValidationError(f"classification must be one of {sorted(CLASSIFICATIONS)}")
    return str(value)


def require_retention_hours(value: Any) -> int:
    """Retention must be an integer number of hours in [1, 168] (FR4.3)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError("retention_hours must be an integer")
    if not MIN_RETENTION_HOURS <= value <= MAX_RETENTION_HOURS:
        raise ValidationError(
            f"retention_hours must be between {MIN_RETENTION_HOURS} and {MAX_RETENTION_HOURS}"
        )
    return value


def _require_non_negative(value: float, field_name: str) -> None:
    if value < 0:
        raise ValidationError(f"{field_name} must not be negative")


@dataclass(frozen=True)
class Chunk:
    """One ordered paragraph section of a document with its SHA-256 (FR2.7)."""

    index: int
    text: str
    sha256: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValidationError("chunk index must not be negative")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValidationError("chunk sha256 must be 64 lowercase hex characters")


@dataclass(frozen=True)
class Document:
    """Document metadata and its version pointers (FR2.1, FR2.2)."""

    document_id: str
    title: str
    owner_team: str
    classification: str
    status: str
    latest_version_number: int
    current_version_id: str | None = None
    pending_version_id: str | None = None

    def __post_init__(self) -> None:
        require_id(self.document_id, "document_id")
        require_text(self.title, "title", MAX_TITLE_CHARS)
        require_id(self.owner_team, "owner_team")
        require_classification(self.classification)
        if self.status not in DOCUMENT_STATUSES:
            raise ValidationError(f"status must be one of {sorted(DOCUMENT_STATUSES)}")


@dataclass(frozen=True)
class DocumentVersion:
    """One immutable version of a document (FR2.1)."""

    document_id: str
    version_id: str
    owner_team: str
    classification: str
    status: str
    author: str
    created_at: str
    s3_key: str
    chunk_hashes: tuple[str, ...]
    s3_version_id: str | None = None
    injection_findings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.document_id, "document_id")
        require_id(self.version_id, "version_id")
        require_id(self.owner_team, "owner_team")
        require_classification(self.classification)
        require_text(self.author, "author", 256)
        if self.status not in DOCUMENT_STATUSES:
            raise ValidationError(f"status must be one of {sorted(DOCUMENT_STATUSES)}")

    @property
    def ref(self) -> str:
        """`document_id@version_id`, the form recorded on every request (FR2.2)."""
        return f"{self.document_id}@{self.version_id}"


@dataclass(frozen=True)
class AgentConfig:
    """Per-agent configuration held in SSM Parameter Store (FR3.5, FR4.3, FR5.5)."""

    agent_id: str
    team: str
    model_id: str
    retention_hours: int
    monthly_budget_usd: float
    allowed_classifications: frozenset[str]
    system_prompt: str = ""
    shadow_model_id: str | None = None

    def __post_init__(self) -> None:
        require_id(self.agent_id, "agent_id")
        require_id(self.team, "team")
        require_id(self.model_id, "model_id")
        if self.shadow_model_id is not None:
            require_id(self.shadow_model_id, "shadow_model_id")
        require_retention_hours(self.retention_hours)
        _require_non_negative(self.monthly_budget_usd, "monthly_budget_usd")
        for classification in self.allowed_classifications:
            require_classification(classification)
        require_text(self.system_prompt, "system_prompt", MAX_SYSTEM_PROMPT_CHARS, allow_empty=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        if not isinstance(data, dict):
            raise ValidationError("agent config must be a JSON object")
        try:
            return cls(
                agent_id=data["agent_id"],
                team=data["team"],
                model_id=data["model_id"],
                retention_hours=data["retention_hours"],
                monthly_budget_usd=float(data.get("monthly_budget_usd", 0.0)),
                allowed_classifications=frozenset(data.get("allowed_classifications", [])),
                system_prompt=data.get("system_prompt", ""),
                shadow_model_id=data.get("shadow_model_id"),
            )
        except KeyError as exc:
            raise ValidationError(f"agent config is missing {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"agent config is invalid: {exc}") from exc


@dataclass(frozen=True)
class ModelCapability:
    """Caching rules for one Bedrock model (FR3.8)."""

    model_id: str
    family: str
    model_version: str
    supports_prompt_cache: bool
    min_cache_tokens: int
    cache_ttl_seconds: int

    def __post_init__(self) -> None:
        require_id(self.model_id, "model_id")
        require_id(self.model_version, "model_version")
        if self.family not in {"anthropic", "amazon-nova", "deepseek"}:
            raise ValidationError(f"unsupported model family {self.family!r}")
        _require_non_negative(self.min_cache_tokens, "min_cache_tokens")
        _require_non_negative(self.cache_ttl_seconds, "cache_ttl_seconds")


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens for one model (FR1.4, FR5.1)."""

    model_id: str
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float
    cache_write_per_million: float

    def __post_init__(self) -> None:
        for name in (
            "input_per_million",
            "output_per_million",
            "cache_read_per_million",
            "cache_write_per_million",
        ):
            _require_non_negative(getattr(self, name), name)


@dataclass(frozen=True)
class Message:
    role: MessageRole
    text: str

    def __post_init__(self) -> None:
        if self.role not in MESSAGE_ROLES:
            raise ValidationError("message role must be 'user' or 'assistant'")
        require_text(self.text, "message text", MAX_MESSAGE_CHARS)


@dataclass(frozen=True)
class InvokeRequest:
    """What an agent sends to the gateway (FR3.1)."""

    agent_id: str
    customer_id: str
    document_ids: tuple[str, ...]
    messages: tuple[Message, ...]
    history: tuple[Message, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.agent_id, "agent_id")
        require_id(self.customer_id, "customer_id")
        if len(self.document_ids) > MAX_DOCUMENTS_PER_REQUEST:
            raise ValidationError(f"at most {MAX_DOCUMENTS_PER_REQUEST} documents per request")
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValidationError("document_ids must not repeat")
        for document_id in self.document_ids:
            require_id(document_id, "document_id")
        if not self.messages:
            raise ValidationError("messages must not be empty")
        if len(self.messages) + len(self.history) > MAX_MESSAGES:
            raise ValidationError(f"at most {MAX_MESSAGES} messages including history")
        conversation = (*self.history, *self.messages)
        if conversation[0].role != "user" or conversation[-1].role != "user":
            raise ValidationError("the conversation must start and end with a user message")
        for previous, current in pairwise(conversation):
            if previous.role == current.role:
                raise ValidationError("message roles must alternate between user and assistant")


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            _require_non_negative(getattr(self, name), name)

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass(frozen=True)
class InvokeResult:
    request_id: str
    answer: str
    model_id: str
    model_version: str
    cache_outcome: str
    document_versions: tuple[str, ...]
    usage: TokenUsage
    latency_ms: int
    degraded: bool = False

    def __post_init__(self) -> None:
        if self.cache_outcome not in CACHE_OUTCOMES:
            raise ValidationError(f"cache_outcome must be one of {sorted(CACHE_OUTCOMES)}")


@dataclass(frozen=True)
class UsageRecord:
    """One gateway request's usage and cost (FR5.1)."""

    request_id: str
    timestamp: str
    agent_id: str
    team: str
    customer_id: str
    model_id: str
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    full_price_tokens: int
    output_tokens: int
    cost_usd: float
    saving_usd: float
    storage_cost_usd: float
    latency_ms: int
    cache_outcome: str
    degraded: bool = False

    def __post_init__(self) -> None:
        if self.cache_outcome not in CACHE_OUTCOMES:
            raise ValidationError(f"cache_outcome must be one of {sorted(CACHE_OUTCOMES)}")
        _require_non_negative(self.latency_ms, "latency_ms")


@dataclass(frozen=True)
class CacheEntry:
    """A stored response for response reuse (FR4.2, FR4.4, FR4.5)."""

    team: str
    customer_id: str
    agent_id: str
    cache_key: str
    model_id: str
    model_version: str
    document_versions: tuple[str, ...]
    response_text: str
    usage: TokenUsage
    created_at: int
    expires_at: int

    def __post_init__(self) -> None:
        require_id(self.team, "team")
        require_id(self.customer_id, "customer_id")
        require_id(self.agent_id, "agent_id")
        if not re.fullmatch(r"[0-9a-f]{64}", self.cache_key):
            raise ValidationError("cache_key must be a SHA-256 hex digest")
        if self.expires_at <= self.created_at:
            raise ValidationError("expires_at must be after created_at")


@dataclass(frozen=True)
class AuditEvent:
    """An append-only audit record (FR6.1). `subject` is the lookup key, e.g. `req#<id>`."""

    subject: str
    event_id: str
    event_type: str
    actor: str
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject or "#" not in self.subject:
            raise ValidationError("audit subject must look like '<kind>#<id>'")
        require_id(self.event_id, "event_id")
        require_text(self.event_type, "event_type", 64)
        require_text(self.actor, "actor", 512)
