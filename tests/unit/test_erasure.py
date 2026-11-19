"""Customer erasure tests (FR6.2)."""

from __future__ import annotations

import hashlib
import json

import pytest
from botocore.exceptions import ClientError

from keepsake.cache.repository import CacheRepository
from keepsake.common.audit import AuditLog, customer_ref
from keepsake.common.errors import DependencyUnavailable, ValidationError
from keepsake.common.models import CacheEntry, TokenUsage, UsageRecord
from keepsake.governance.erasure import ErasureService
from keepsake.usage.repository import UsageRepository
from tests.conftest import REPORTS_BUCKET, AwsResources


def _entry(customer: str) -> CacheEntry:
    return CacheEntry(
        team="helpdesk",
        customer_id=customer,
        agent_id="helpdesk",
        cache_key=hashlib.sha256(customer.encode()).hexdigest(),
        model_id="deepseek.v3.2",
        model_version="v3.2",
        document_versions=("policy-manual@000001",),
        response_text="answer",
        usage=TokenUsage(input_tokens=10),
        created_at=1,
        expires_at=4_000_000_000,
    )


def _usage(customer: str, request_id: str) -> UsageRecord:
    return UsageRecord(
        request_id=request_id,
        timestamp="2026-11-10T12:00:00.000Z",
        agent_id="helpdesk",
        team="helpdesk",
        customer_id=customer,
        model_id="deepseek.v3.2",
        input_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=0,
        full_price_tokens=10,
        output_tokens=5,
        cost_usd=0.1,
        saving_usd=0.0,
        storage_cost_usd=0.0,
        latency_ms=50,
        cache_outcome="miss",
    )


def _service(aws: AwsResources, fixed_now: object) -> ErasureService:
    return ErasureService(
        cache=CacheRepository(table=aws.tables["cache"]),
        usage=UsageRepository(table=aws.tables["usage"]),
        audit=AuditLog(table=aws.tables["audit"], now=fixed_now),
        s3=aws.s3,
        reports_bucket=REPORTS_BUCKET,
        kms_key_id=aws.kms_key_id,
        now=fixed_now,
    )


def test_erasure_removes_only_that_customers_data_and_writes_a_report(
    aws: AwsResources,
    fixed_now: object,
) -> None:
    cache = CacheRepository(table=aws.tables["cache"])
    usage = UsageRepository(table=aws.tables["usage"])
    for customer in ("cust-1", "cust-2"):
        cache.put(_entry(customer))
    usage.put(_usage("cust-1", "r1"))
    usage.put(_usage("cust-1", "r2"))
    usage.put(_usage("cust-2", "r3"))

    report = _service(aws, fixed_now).erase_customer("cust-1", requested_by="dpo@example.com")

    assert (report.cache_entries_removed, report.usage_records_removed) == (1, 2)
    assert report.usage_request_ids_removed == ("r1", "r2")
    assert cache.list_for_customer("cust-1") == ()
    assert len(cache.list_for_customer("cust-2")) == 1
    stored = json.loads(
        aws.s3.get_object(Bucket=REPORTS_BUCKET, Key=report.report_key)["Body"].read()
    )
    assert stored["customer_ref"] == customer_ref("cust-1")
    assert "cust-1" not in json.dumps(stored)
    audit = AuditLog(table=aws.tables["audit"]).lookup(f"erasure#{customer_ref('cust-1')}")
    assert audit[0].event_type == "customer.erased"


def test_erasure_rejects_invalid_customer_ids(aws: AwsResources, fixed_now: object) -> None:
    with pytest.raises(ValidationError):
        _service(aws, fixed_now).erase_customer("../../etc", requested_by="dpo")


def test_erasure_surfaces_dependency_failures(
    aws: AwsResources,
    fixed_now: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(**_: object) -> None:
        raise ClientError({"Error": {"Code": "InternalServerError", "Message": "x"}}, "Query")

    monkeypatch.setattr(aws.tables["cache"], "query", broken)
    with pytest.raises(DependencyUnavailable):
        _service(aws, fixed_now).erase_customer("cust-1", requested_by="dpo")
