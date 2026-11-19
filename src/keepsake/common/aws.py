"""AWS client construction and error translation at the integration boundary."""

from __future__ import annotations

import logging
import os
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from keepsake.common.errors import Conflict, DependencyUnavailable, NotFound

LOGGER = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"
# Short timeouts keep gateway overhead bounded (NFR1); standard retries handle throttling.
AWS_CLIENT_CONFIG = Config(
    connect_timeout=2,
    read_timeout=5,
    retries={"mode": "standard", "max_attempts": 3},
)

_NOT_FOUND_CODES = frozenset({"NoSuchKey", "ParameterNotFound", "ResourceNotFoundException", "404"})
_CONFLICT_CODES = frozenset({"ConditionalCheckFailedException", "TransactionCanceledException"})


def region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION


def client(service: str, *, config: Config | None = None) -> Any:
    """A boto3 client in the Keepsake region with bounded timeouts."""
    return boto3.client(service, region_name=region(), config=config or AWS_CLIENT_CONFIG)


def dynamodb_table(name: str) -> Any:
    """A boto3 DynamoDB Table resource in the Keepsake region."""
    resource = boto3.resource("dynamodb", region_name=region(), config=AWS_CLIENT_CONFIG)
    return resource.Table(name)


def error_code(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        return str(exc.response.get("Error", {}).get("Code", "Unknown"))
    return type(exc).__name__


def translate_aws_error(exc: Exception, operation: str) -> Exception:
    """Map a botocore failure to a domain error and log it with context (never swallowed)."""
    code = error_code(exc)
    if code in _NOT_FOUND_CODES:
        return NotFound(f"{operation}: not found")
    if code in _CONFLICT_CODES:
        return Conflict(f"{operation}: the resource changed concurrently; retry the request")
    LOGGER.error("AWS call failed", extra={"operation": operation, "error_code": code})
    return DependencyUnavailable(f"{operation} is temporarily unavailable")


AWS_ERRORS = (ClientError, BotoCoreError)
