"""Phase 0 log analyzer, isolated from the later Keepsake application prototype.

Event: ``{"source": "sample"}`` analyzes generated sample traffic; ``{"prefix": "AWSLogs/"}``
reads invocation logs from the one source bucket configured by the Phase 0 stack. Reports go to
the stack-owned encrypted reports bucket. This function never invokes a model or writes customer
prompt text into the report.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from keepsake.analyzer import sample_logs
from keepsake.analyzer.logs import load_from_s3, parse_lines
from keepsake.analyzer.pipeline import run_analysis
from keepsake.analyzer.report import write_report
from keepsake.common.clock import utc_now
from keepsake.common.aws import client
from keepsake.common.config import parse_prices
from keepsake.common.errors import ConfigurationError, ValidationError

LOGGER = logging.getLogger("keepsake.phase0")

MAX_PREFIX_CHARS = 512


def _parse_source(s3: Any, event: dict[str, Any]) -> Any:
    if event.get("source") == "sample":
        return parse_lines(sample_logs.to_lines(sample_logs.generate()))
    configured_prefix = os.environ.get("KEEPSAKE_SOURCE_INVOCATION_LOGS_PREFIX", "AWSLogs/")
    prefix = event.get("prefix", configured_prefix)
    if (
        not isinstance(prefix, str)
        or not prefix
        or len(prefix) > MAX_PREFIX_CHARS
        or ".." in prefix
        or not prefix.startswith(configured_prefix)
    ):
        raise ValidationError(
            "event must contain source='sample' or a prefix within the configured log prefix"
        )
    bucket = os.environ.get("KEEPSAKE_SOURCE_INVOCATION_LOGS_BUCKET", "")
    if not bucket:
        raise ConfigurationError("KEEPSAKE_SOURCE_INVOCATION_LOGS_BUCKET is not set")
    return load_from_s3(s3, bucket, prefix)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    del context
    s3 = client("s3")
    parsed = _parse_source(s3, event or {})
    try:
        price_data = json.loads(os.environ["KEEPSAKE_PHASE0_PRICES_JSON"])
        build_cost = float(os.environ.get("KEEPSAKE_BUILD_RUN_COST_USD", "500"))
    except (KeyError, ValueError) as exc:
        raise ConfigurationError("Phase 0 price table or operating cost is invalid") from exc
    prices, storage_price = parse_prices(price_data)
    report = run_analysis(parsed, prices, storage_price, build_cost, utc_now())
    json_key, md_key = write_report(
        s3,
        os.environ["KEEPSAKE_REPORTS_BUCKET"],
        "phase0",
        report,
        os.environ["KEEPSAKE_KMS_KEY_ID"],
    )
    LOGGER.info("Phase 0 report written: verdict=%s key=%s", report.verdict, json_key)
    return {"verdict": report.verdict, "report_json": json_key, "report_markdown": md_key}
