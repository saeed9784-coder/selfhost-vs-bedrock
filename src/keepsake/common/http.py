"""HTTP helpers for API Gateway (REST, proxy integration) Lambda handlers.

Every response uses one envelope: `{"success": bool, "data": ..., "error": str | null}`.
Domain errors map to 400/403/404/409/503; anything unexpected is logged with its stack trace and
returned as a generic 500 so no internal detail reaches the caller.
"""

from __future__ import annotations

import base64
import binascii
import functools
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from keepsake.common.errors import (
    AccessDenied,
    ConfigurationError,
    Conflict,
    DependencyUnavailable,
    KeepsakeError,
    NotFound,
    ValidationError,
)
from keepsake.store.service import StaffIdentity
from keepsake.usage.metrics import MetricsEmitter

LOGGER = logging.getLogger(__name__)
MAX_BODY_BYTES = 6_000_000  # API Gateway payload limit is 10 MB; Lambda sync payload is 6 MB
_STATUS: tuple[tuple[type[KeepsakeError], int], ...] = (
    (ValidationError, 400),
    (AccessDenied, 403),
    (NotFound, 404),
    (Conflict, 409),
    (DependencyUnavailable, 503),
    (ConfigurationError, 503),
)
Event = dict[str, Any]
Response = dict[str, Any]


def status_for(error: KeepsakeError) -> int:
    for error_type, status in _STATUS:
        if isinstance(error, error_type):
            return status
    return 500


def _headers() -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    }
    origin = os.environ.get("KEEPSAKE_ALLOWED_ORIGIN", "")
    if origin:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
    return headers


def respond(status: int, data: Any = None, error: str | None = None) -> Response:
    body = {"success": error is None, "data": data, "error": error}
    return {"statusCode": status, "headers": _headers(), "body": json.dumps(body, default=str)}


def ok(data: Any, status: int = 200) -> Response:
    return respond(status, data=data)


def parse_body(event: Event) -> dict[str, Any]:
    raw = event.get("body")
    if raw is None or raw == "":
        raise ValidationError("request body is required")
    if event.get("isBase64Encoded"):
        try:
            raw = base64.b64decode(raw).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValidationError("request body is not valid base64 UTF-8") from exc
    if len(raw.encode("utf-8")) > MAX_BODY_BYTES:
        raise ValidationError("request body is too large")
    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise ValidationError("request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise ValidationError("request body must be a JSON object")
    return body


def path_param(event: Event, name: str) -> str:
    value = (event.get("pathParameters") or {}).get(name)
    if not value:
        raise ValidationError(f"path parameter {name} is required")
    return str(value)


def query_params(event: Event) -> dict[str, str]:
    return {str(k): str(v) for k, v in (event.get("queryStringParameters") or {}).items()}


def caller_arn(event: Event) -> str:
    """The SigV4 caller ARN that API Gateway IAM authorization verified."""
    arn = ((event.get("requestContext") or {}).get("identity") or {}).get("userArn")
    if not arn:
        raise AccessDenied("IAM-signed request required")
    return str(arn)


def _parse_groups(raw: Any) -> frozenset[str]:
    if isinstance(raw, list):
        return frozenset(str(g) for g in raw)
    text = str(raw or "").strip().strip("[]")
    return frozenset(g for g in text.replace(",", " ").split() if g)


def staff_identity(event: Event) -> StaffIdentity:
    """The Cognito user API Gateway's Cognito authorizer verified (claims are trusted here)."""
    claims = ((event.get("requestContext") or {}).get("authorizer") or {}).get("claims")
    if not isinstance(claims, dict):
        raise AccessDenied("signed-in staff user required")
    username = claims.get("email") or claims.get("cognito:username") or claims.get("sub")
    if not username:
        raise AccessDenied("signed-in staff user required")
    return StaffIdentity(username=str(username), groups=_parse_groups(claims.get("cognito:groups")))


def api_handler(
    component: str,
    metrics: Callable[[], MetricsEmitter] = MetricsEmitter,
) -> Callable[[Callable[[Event, Any], Response]], Callable[[Event, Any], Response]]:
    """Error mapping plus request/error metrics (NFR7) around a Lambda handler."""

    def decorate(func: Callable[[Event, Any], Response]) -> Callable[[Event, Any], Response]:
        @functools.wraps(func)
        def wrapper(event: Event, context: Any) -> Response:
            is_error = True
            try:
                response = func(event, context)
                is_error = response["statusCode"] >= 500
                return response
            except KeepsakeError as exc:
                status = status_for(exc)
                is_error = status >= 500
                log = LOGGER.error if is_error else LOGGER.info
                log(
                    "request failed",
                    extra={
                        "component": component,
                        "status": status,
                        "error_type": type(exc).__name__,
                    },
                )
                return respond(status, error=exc.message)
            except Exception:
                LOGGER.exception("unhandled error", extra={"component": component})
                return respond(500, error="internal error")
            finally:
                metrics().component_request(component, is_error=is_error)

        return wrapper

    return decorate
