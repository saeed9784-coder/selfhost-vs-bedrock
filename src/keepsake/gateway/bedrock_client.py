"""Bedrock Converse calls with bounded retries on throttling (FR3.3, FR4.1, FR5.1).

Recoverable errors (throttling, service unavailable, model not ready, internal error) are retried
with exponential backoff up to `max_attempts`, then surface as DependencyUnavailable (HTTP 503).
Request errors are fatal: ValidationException -> ValidationError; access or missing-model errors
-> ConfigurationError (model access not enabled or wrong id in configuration).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from botocore.config import Config

from keepsake.common.aws import AWS_ERRORS, client, error_code
from keepsake.common.errors import ConfigurationError, DependencyUnavailable, ValidationError
from keepsake.common.models import TokenUsage
from keepsake.gateway.assembler import AssembledPrompt

LOGGER = logging.getLogger(__name__)
RETRYABLE_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelNotReadyException",
        "InternalServerException",
        "ModelTimeoutException",
    }
)
CONFIGURATION_CODES = frozenset({"AccessDeniedException", "ResourceNotFoundException"})
DEFAULT_MAX_TOKENS = 1024
# The gateway retries itself, so botocore's own retries are disabled for this client.
BEDROCK_CLIENT_CONFIG = Config(
    connect_timeout=3, read_timeout=120, retries={"mode": "standard", "total_max_attempts": 1}
)


def bedrock_runtime_client() -> Any:
    return client("bedrock-runtime", config=BEDROCK_CLIENT_CONFIG)


@dataclass(frozen=True)
class ModelResponse:
    model_id: str
    text: str
    usage: TokenUsage
    latency_ms: int
    stop_reason: str


def parse_response(model_id: str, response: dict[str, Any], latency_ms: int) -> ModelResponse:
    content = response.get("output", {}).get("message", {}).get("content", [])
    text = "".join(block.get("text", "") for block in content if isinstance(block, dict))
    usage = response.get("usage", {})
    return ModelResponse(
        model_id=model_id,
        text=text,
        usage=TokenUsage(
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            cache_read_tokens=int(usage.get("cacheReadInputTokens", 0)),
            cache_write_tokens=int(usage.get("cacheWriteInputTokens", 0)),
        ),
        latency_ms=latency_ms,
        stop_reason=str(response.get("stopReason", "")),
    )


class BedrockGateway:
    def __init__(
        self,
        bedrock_client: Any,
        *,
        max_attempts: int = 3,
        base_delay_s: float = 0.2,
        sleep: Callable[[float], None] = time.sleep,
        perf_counter: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._client = bedrock_client
        self._max_attempts = max_attempts
        self._base_delay = base_delay_s
        self._sleep = sleep
        self._perf_counter = perf_counter

    def converse(
        self, model_id: str, prompt: AssembledPrompt, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> ModelResponse:
        request = prompt.converse_request(model_id, max_tokens)
        for attempt in range(1, self._max_attempts + 1):
            started = self._perf_counter()
            try:
                response = self._client.converse(**request)
            except AWS_ERRORS as exc:
                self._handle_failure(exc, model_id, attempt)
                continue
            latency_ms = int((self._perf_counter() - started) * 1000)
            return parse_response(model_id, response, latency_ms)
        raise DependencyUnavailable("the model is busy; retry later")  # pragma: no cover

    def _handle_failure(self, exc: Exception, model_id: str, attempt: int) -> None:
        """Sleep before a retry, or raise the mapped domain error."""
        code = error_code(exc)
        if code in RETRYABLE_CODES and attempt < self._max_attempts:
            delay = self._base_delay * (2 ** (attempt - 1))
            LOGGER.warning(
                "Bedrock call throttled; retrying",
                extra={"model": model_id, "attempt": attempt, "code": code},
            )
            self._sleep(delay)
            return
        LOGGER.error(
            "Bedrock call failed", extra={"model": model_id, "code": code, "attempt": attempt}
        )
        if code == "ValidationException":
            raise ValidationError("the model rejected the request") from exc
        if code in CONFIGURATION_CODES:
            raise ConfigurationError(f"model {model_id} is not available to Keepsake") from exc
        raise DependencyUnavailable("the model is temporarily unavailable; retry later") from exc
