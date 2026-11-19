"""Domain errors. Handlers map each type to one HTTP status (see common/http.py).

Recoverable vs fatal:
- DependencyUnavailable is recoverable: callers may retry or fall back (the gateway falls back to
  a full-price model call when the store or cache is down, FR3.9).
- ValidationError, AccessDenied, NotFound, ModelBlocked, Conflict and ConfigurationError are
  fatal for the request: retrying the same input will fail the same way.
"""

from __future__ import annotations


class KeepsakeError(Exception):
    """Base class for every Keepsake domain error. `message` is safe to show to callers."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ValidationError(KeepsakeError):
    """Input failed validation at a system boundary (HTTP 400)."""


class AccessDenied(KeepsakeError):
    """The caller may not perform this action or read this data (HTTP 403)."""


class ModelBlocked(AccessDenied):
    """The configured model is blocked for this customer's group (HTTP 403, audited, FR3.7)."""

    def __init__(self, message: str, *, model_id: str, customer_group: str) -> None:
        super().__init__(message)
        self.model_id = model_id
        self.customer_group = customer_group


class NotFound(KeepsakeError):
    """The requested resource does not exist (HTTP 404)."""


class Conflict(KeepsakeError):
    """The request conflicts with current state, e.g. a concurrent update (HTTP 409)."""


class DependencyUnavailable(KeepsakeError):
    """An AWS dependency failed or timed out; recoverable by retry or fallback (HTTP 503)."""


class ConfigurationError(KeepsakeError):
    """Deployment configuration is missing or invalid; fatal until fixed (HTTP 503)."""
