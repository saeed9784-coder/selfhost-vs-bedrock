"""Deployment settings read from environment variables set by the CDK stack."""

from __future__ import annotations

import os
from dataclasses import dataclass

from keepsake.common.errors import ConfigurationError


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigurationError(f"environment variable {name} is not set")
    return value


@dataclass(frozen=True)
class Settings:
    documents_table: str
    cache_table: str
    usage_table: str
    audit_table: str
    documents_bucket: str
    reports_bucket: str
    kms_key_id: str
    alerts_topic_arn: str
    ssm_prefix: str = "/keepsake"
    build_run_cost_usd: float = 500.0
    allowed_origin: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        """Read every setting; missing values are fatal (ConfigurationError)."""
        try:
            build_cost = float(os.environ.get("KEEPSAKE_BUILD_RUN_COST_USD", "500"))
        except ValueError as exc:
            raise ConfigurationError("KEEPSAKE_BUILD_RUN_COST_USD must be a number") from exc
        return cls(
            documents_table=_required("KEEPSAKE_DOCUMENTS_TABLE"),
            cache_table=_required("KEEPSAKE_CACHE_TABLE"),
            usage_table=_required("KEEPSAKE_USAGE_TABLE"),
            audit_table=_required("KEEPSAKE_AUDIT_TABLE"),
            documents_bucket=_required("KEEPSAKE_DOCUMENTS_BUCKET"),
            reports_bucket=_required("KEEPSAKE_REPORTS_BUCKET"),
            kms_key_id=_required("KEEPSAKE_KMS_KEY_ID"),
            alerts_topic_arn=_required("KEEPSAKE_ALERTS_TOPIC_ARN"),
            ssm_prefix=os.environ.get("KEEPSAKE_SSM_PREFIX", "/keepsake").rstrip("/"),
            build_run_cost_usd=build_cost,
            allowed_origin=os.environ.get("KEEPSAKE_ALLOWED_ORIGIN", ""),
        )
