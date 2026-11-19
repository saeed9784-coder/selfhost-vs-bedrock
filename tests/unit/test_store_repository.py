"""Document repository tests against moto (FR2.1, FR2.2, FR2.3)."""

from __future__ import annotations

import hashlib

import pytest

from keepsake.common.errors import Conflict, NotFound
from keepsake.common.models import Chunk, Document, DocumentVersion
from keepsake.store.repository import DocumentRepository, object_key, version_id_for
from tests.conftest import DOCUMENTS_BUCKET, AwsResources


def _chunks(*texts: str) -> tuple[Chunk, ...]:
    return tuple(
        Chunk(index=i, text=t, sha256=hashlib.sha256(t.encode()).hexdigest())
        for i, t in enumerate(texts)
    )


def _repo(aws: AwsResources) -> DocumentRepository:
    return DocumentRepository(
        s3=aws.s3, table=aws.tables["documents"], bucket=DOCUMENTS_BUCKET, kms_key_id=aws.kms_key_id
    )


def _write(repo: DocumentRepository, number: int, *texts: str) -> DocumentVersion:
    version_id = version_id_for(number)
    chunks = _chunks(*texts)
    version = DocumentVersion(
        document_id="policy-manual",
        version_id=version_id,
        owner_team="company",
        classification="policy",
        status="active",
        author="ana@example.com",
        created_at=f"2026-11-10T12:0{number}:00.000Z",
        s3_key=object_key("policy-manual", version_id),
        chunk_hashes=tuple(c.sha256 for c in chunks),
    )
    document = Document(
        document_id="policy-manual",
        title="Policy manual",
        owner_team="company",
        classification="policy",
        status="active",
        latest_version_number=number,
        current_version_id=version_id,
    )
    return repo.write_version(document, version, chunks, expected_latest=number - 1)


def test_version_history_keeps_every_version_and_current_points_to_latest(
    aws: AwsResources,
) -> None:
    repo = _repo(aws)

    _write(repo, 1, "Refunds within 30 days.")
    v2 = _write(repo, 2, "Refunds within 45 days.")

    history = repo.list_versions("policy-manual")
    document = repo.get_document("policy-manual")
    assert [v.version_id for v in history] == ["000001", "000002"]
    assert document.current_version_id == "000002"
    assert repo.load_chunks(v2)[0].text == "Refunds within 45 days."
    assert repo.load_chunks(history[0])[0].text == "Refunds within 30 days."


def test_document_body_is_written_with_sse_kms_and_the_customer_managed_key(
    aws: AwsResources,
) -> None:
    version = _write(_repo(aws), 1, "Escalate after 48 hours.")

    head = aws.s3.head_object(Bucket=DOCUMENTS_BUCKET, Key=version.s3_key)

    assert head["ServerSideEncryption"] == "aws:kms"
    assert aws.kms_key_id in head["SSEKMSKeyId"]


def test_concurrent_write_with_a_stale_version_number_conflicts(aws: AwsResources) -> None:
    repo = _repo(aws)
    _write(repo, 1, "first")

    with pytest.raises(Conflict):
        _write(repo, 1, "stale writer")


def test_missing_document_and_version_raise_not_found(aws: AwsResources) -> None:
    repo = _repo(aws)

    with pytest.raises(NotFound):
        repo.get_document("does-not-exist")
    with pytest.raises(NotFound):
        repo.get_version("does-not-exist", "000001")
