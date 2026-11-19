"""Customer erasure across cache entries and request logs, with a deletion report (FR6.2).

What holds customer data:
- Cache entries (response text and, through the reuse key, the customer's conversation). The
  gateway does not persist customer history separately: history arrives with each request and
  is only retained inside cache entries, so deleting the entries removes it.
- Usage records (request logs: customer id, tokens, cost).
- The audit log stores only a SHA-256 `customer_ref`, never the raw id, so it stays append-only.

The report lists what was removed and is written to the reports bucket (SSE-KMS) and the audit
log. If a step fails, the error surfaces (503) and the request can be retried safely: each step
is idempotent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from keepsake.cache.repository import CacheRepository
from keepsake.common.audit import AuditLog, customer_ref
from keepsake.common.aws import AWS_ERRORS, translate_aws_error
from keepsake.common.clock import Clock, iso, utc_now
from keepsake.common.models import require_id
from keepsake.usage.repository import UsageRepository

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeletionReport:
    customer_ref: str
    requested_by: str
    completed_at: str
    cache_entries_removed: int
    cache_keys_removed: tuple[str, ...]
    usage_records_removed: int
    usage_request_ids_removed: tuple[str, ...]
    customer_history_items_removed: int
    report_key: str


class ErasureService:
    def __init__(
        self,
        *,
        cache: CacheRepository,
        usage: UsageRepository,
        audit: AuditLog,
        s3: Any,
        reports_bucket: str,
        kms_key_id: str,
        now: Clock = utc_now,
    ) -> None:
        self._cache = cache
        self._usage = usage
        self._audit = audit
        self._s3 = s3
        self._bucket = reports_bucket
        self._kms_key_id = kms_key_id
        self._now = now

    def erase_customer(self, customer_id: str, requested_by: str) -> DeletionReport:
        require_id(customer_id, "customer_id")
        ref = customer_ref(customer_id)
        entries = self._cache.list_for_customer(customer_id)
        removed_entries = self._cache.delete_by_customer(customer_id)
        request_ids = self._usage.delete_by_customer(customer_id)
        completed = iso(self._now())
        report = DeletionReport(
            customer_ref=ref,
            requested_by=requested_by,
            completed_at=completed,
            cache_entries_removed=removed_entries,
            cache_keys_removed=tuple(sorted(e.cache_key for e in entries)),
            usage_records_removed=len(request_ids),
            usage_request_ids_removed=tuple(sorted(request_ids)),
            customer_history_items_removed=removed_entries,
            report_key=f"erasure-reports/{ref}/{completed}.json",
        )
        self._write_report(report)
        self._audit.record(
            f"erasure#{ref}",
            "customer.erased",
            requested_by,
            {
                "report_key": report.report_key,
                "cache_entries_removed": removed_entries,
                "usage_records_removed": report.usage_records_removed,
            },
        )
        LOGGER.info(
            "customer erased",
            extra={
                "customer_ref": ref,
                "cache_entries": removed_entries,
                "usage_records": report.usage_records_removed,
            },
        )
        return report

    def _write_report(self, report: DeletionReport) -> None:
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=report.report_key,
                ContentType="application/json",
                Body=json.dumps(asdict(report), indent=2).encode("utf-8"),
                ServerSideEncryption="aws:kms",
                SSEKMSKeyId=self._kms_key_id,
            )
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "write deletion report") from exc
