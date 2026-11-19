"""SSM-backed runtime configuration with a short in-memory TTL (FR3.5, FR3.7, FR3.8, NFR5).

Parameters (under the prefix, default `/keepsake`):
- `agents/registry`: JSON `{"<IAM role name>": "<agent_id>"}`; maps caller identity to an agent.
- `agents/<agent_id>/config`: JSON agent configuration (model id, retention, budget, ...).
- `models/blocks`: JSON `{"groups": {"<group>": {"customers": [...], "blocked_models": [...]}}}`.
- `models/capabilities`, `models/prices`: optional overrides of the files in `config/`.

Values are cached for `ttl_seconds` (30 s), so a configuration change, such as switching an
agent's model, takes effect within one TTL on every warm Lambda with no deployment (NFR5: 60 s).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from keepsake.common.aws import AWS_ERRORS, error_code, translate_aws_error
from keepsake.common.errors import (
    AccessDenied,
    ConfigurationError,
    DependencyUnavailable,
    NotFound,
    ValidationError,
)
from keepsake.common.models import AgentConfig, ModelCapability, ModelPrice

LOGGER = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 30.0
_ROLE_ARN = re.compile(r"^arn:aws[a-z-]*:iam::\d{12}:role/(?:[\w+=,.@-]+/)*([\w+=,.@-]{1,64})$")
_ASSUMED_ROLE_ARN = re.compile(
    r"^arn:aws[a-z-]*:sts::\d{12}:assumed-role/([\w+=,.@-]{1,64})/[\w+=,.@-]{1,64}$"
)


def default_config_dir() -> Path:
    """`KEEPSAKE_CONFIG_DIR`, else the repository's `config/` directory."""
    configured = os.environ.get("KEEPSAKE_CONFIG_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "config"


def role_name_from_arn(caller_arn: str) -> str:
    """The IAM role name from an IAM role ARN or an STS assumed-role ARN."""
    for pattern in (_ASSUMED_ROLE_ARN, _ROLE_ARN):
        match = pattern.fullmatch(caller_arn or "")
        if match:
            return match.group(1)
    raise AccessDenied("caller identity is not an IAM role")


def parse_capabilities(data: dict[str, Any]) -> dict[str, ModelCapability]:
    try:
        return {
            model_id: ModelCapability(
                model_id=model_id,
                family=spec["family"],
                model_version=spec["model_version"],
                supports_prompt_cache=bool(spec["supports_prompt_cache"]),
                min_cache_tokens=int(spec.get("min_cache_tokens", 0)),
                cache_ttl_seconds=int(spec.get("cache_ttl_seconds", 0)),
            )
            for model_id, spec in data["models"].items()
        }
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ConfigurationError(f"model capability table is invalid: {exc}") from exc


def parse_prices(data: dict[str, Any]) -> tuple[dict[str, ModelPrice], float]:
    """Per-model prices and the storage price in USD per GB-month."""
    try:
        prices = {
            model_id: ModelPrice(model_id=model_id, **{k: float(v) for k, v in spec.items()})
            for model_id, spec in data["models"].items()
        }
        return prices, float(data["storage_usd_per_gb_month"])
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ConfigurationError(f"price table is invalid: {exc}") from exc


class ConfigProvider:
    """Reads configuration from SSM with a TTL cache; files are the fallback for model tables."""

    def __init__(
        self,
        *,
        ssm: Any,
        prefix: str = "/keepsake",
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        config_dir: Path | None = None,
    ) -> None:
        self._ssm = ssm
        self._prefix = prefix.rstrip("/")
        self._ttl = ttl_seconds
        self._monotonic = monotonic
        self._config_dir = config_dir or default_config_dir()
        self._cache: dict[str, tuple[float, Any]] = {}

    def resolve_agent(self, caller_arn: str) -> str:
        """The agent id registered for the caller's IAM role; unknown callers are denied."""
        role_name = role_name_from_arn(caller_arn)
        registry = self._json_parameter("agents/registry")
        agent_id = registry.get(role_name) if isinstance(registry, dict) else None
        if not agent_id:
            raise AccessDenied("caller is not a registered Keepsake agent")
        return str(agent_id)

    def agent_ids(self) -> tuple[str, ...]:
        registry = self._json_parameter("agents/registry")
        return tuple(sorted(set(registry.values()))) if isinstance(registry, dict) else ()

    def agent_config(self, agent_id: str) -> AgentConfig:
        try:
            data = self._json_parameter(f"agents/{agent_id}/config")
        except NotFound as exc:
            raise AccessDenied(f"agent {agent_id} has no configuration") from exc
        try:
            return AgentConfig.from_dict(data)
        except ValidationError as exc:
            raise ConfigurationError(f"agent {agent_id} configuration is invalid") from exc

    def model_blocks(self) -> dict[str, Any]:
        try:
            data = self._json_parameter("models/blocks")
        except NotFound:
            return {"groups": {}}
        if not isinstance(data, dict) or not isinstance(data.get("groups", {}), dict):
            raise ConfigurationError('model block list must be {"groups": {...}}')
        return data

    def capabilities(self) -> dict[str, ModelCapability]:
        return parse_capabilities(
            self._table_with_override("models/capabilities", "model_capabilities.json")
        )

    def prices(self) -> tuple[dict[str, ModelPrice], float]:
        return parse_prices(self._table_with_override("models/prices", "prices.json"))

    def _table_with_override(self, parameter: str, filename: str) -> dict[str, Any]:
        """SSM override if present and readable; otherwise the shipped file (recoverable)."""
        try:
            return self._json_parameter(parameter)
        except NotFound:
            pass
        except DependencyUnavailable as exc:
            LOGGER.warning(
                "config override unavailable, using shipped file",
                extra={"parameter": parameter, "error": exc.message},
            )
        return self._read_file(filename)

    def _read_file(self, filename: str) -> dict[str, Any]:
        path = self._config_dir / filename
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigurationError(f"{filename} not found and no SSM override set") from exc
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"{filename} is unreadable: {exc}") from exc

    def _json_parameter(self, name: str) -> Any:
        full_name = f"{self._prefix}/{name}"
        cached = self._cache.get(full_name)
        now = self._monotonic()
        if cached and now - cached[0] < self._ttl:
            return cached[1]
        try:
            raw = self._ssm.get_parameter(Name=full_name)["Parameter"]["Value"]
        except AWS_ERRORS as exc:
            if error_code(exc) == "ParameterNotFound":
                raise NotFound(f"parameter {full_name} not found") from exc
            raise translate_aws_error(exc, f"read parameter {full_name}") from exc
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise ConfigurationError(f"parameter {full_name} is not valid JSON") from exc
        self._cache[full_name] = (now, value)
        return value
