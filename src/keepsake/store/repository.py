"""Document store data access: S3 (versioned, SSE-KMS) bodies and DynamoDB metadata.

Layout (FR2.1, FR2.2, FR2.3, FR2.7):
- S3 `documents/<document_id>/<version_id>.json`: ordered chunks with SHA-256 each, written with
  SSE-KMS under the customer-managed key. The bucket is versioned, so nothing is overwritten.
- DynamoDB `pk=doc#<id>, sk=document`: the pointer item (current / pending version, status).
- DynamoDB `pk=doc#<id>, sk=v#<version_id>`: one item per immutable version.

Business rules (what becomes current, quarantine, retirement) live in store/service.py; this
module only reads and writes, and translates AWS failures into domain errors.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from boto3.dynamodb.conditions import Key

from keepsake.common.aws import AWS_ERRORS, translate_aws_error
from keepsake.common.errors import NotFound
from keepsake.common.models import Chunk, Document, DocumentVersion

POINTER_SK = "document"
VERSION_PREFIX = "v#"


def version_id_for(number: int) -> str:
    return f"{number:06d}"


def object_key(document_id: str, version_id: str) -> str:
    return f"documents/{document_id}/{version_id}.json"


def _typed(item: dict[str, Any]) -> dict[str, Any]:
    """Drop None values. The Table resource's client serializes native Python types itself."""
    return {k: v for k, v in item.items() if v is not None}


def _pointer_item(document: Document) -> dict[str, Any]:
    return {
        "pk": f"doc#{document.document_id}",
        "sk": POINTER_SK,
        "document_id": document.document_id,
        "title": document.title,
        "owner_team": document.owner_team,
        "classification": document.classification,
        "status": document.status,
        "latest_version_number": document.latest_version_number,
        "current_version_id": document.current_version_id,
        "pending_version_id": document.pending_version_id,
    }


def _version_item(version: DocumentVersion) -> dict[str, Any]:
    return {
        "pk": f"doc#{version.document_id}",
        "sk": f"{VERSION_PREFIX}{version.version_id}",
        "document_id": version.document_id,
        "version_id": version.version_id,
        "owner_team": version.owner_team,
        "classification": version.classification,
        "status": version.status,
        "author": version.author,
        "created_at": version.created_at,
        "s3_key": version.s3_key,
        "s3_version_id": version.s3_version_id,
        "chunk_hashes": list(version.chunk_hashes),
        "injection_findings": list(version.injection_findings),
    }


def _document_from_item(item: dict[str, Any]) -> Document:
    return Document(
        document_id=item["document_id"],
        title=item["title"],
        owner_team=item["owner_team"],
        classification=item["classification"],
        status=item["status"],
        latest_version_number=int(item["latest_version_number"]),
        current_version_id=item.get("current_version_id"),
        pending_version_id=item.get("pending_version_id"),
    )


def _version_from_item(item: dict[str, Any]) -> DocumentVersion:
    return DocumentVersion(
        document_id=item["document_id"],
        version_id=item["version_id"],
        owner_team=item["owner_team"],
        classification=item["classification"],
        status=item["status"],
        author=item["author"],
        created_at=item["created_at"],
        s3_key=item["s3_key"],
        s3_version_id=item.get("s3_version_id"),
        chunk_hashes=tuple(item.get("chunk_hashes", [])),
        injection_findings=tuple(item.get("injection_findings", [])),
    )


def _pointer_condition(expected_latest: int) -> tuple[str, dict[str, Any]]:
    if expected_latest == 0:
        return "attribute_not_exists(pk)", {}
    return "latest_version_number = :expected", {":expected": expected_latest}


class DocumentRepository:
    """Reads and writes documents; every AWS failure surfaces as a domain error."""

    def __init__(self, *, s3: Any, table: Any, bucket: str, kms_key_id: str) -> None:
        self._s3 = s3
        self._table = table
        self._bucket = bucket
        self._kms_key_id = kms_key_id

    def find_document(self, document_id: str) -> Document | None:
        try:
            response = self._table.get_item(Key={"pk": f"doc#{document_id}", "sk": POINTER_SK})
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "read document metadata") from exc
        item = response.get("Item")
        return _document_from_item(item) if item else None

    def get_document(self, document_id: str) -> Document:
        document = self.find_document(document_id)
        if document is None:
            raise NotFound(f"document {document_id} not found")
        return document

    def list_documents(self) -> tuple[Document, ...]:
        """All pointer items (POC scale; a scan with a filter on the pointer sort key)."""
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {"FilterExpression": Key("sk").eq(POINTER_SK)}
        try:
            while True:
                page = self._table.scan(**kwargs)
                items.extend(page.get("Items", []))
                if "LastEvaluatedKey" not in page:
                    break
                kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "list documents") from exc
        return tuple(sorted((_document_from_item(i) for i in items), key=lambda d: d.document_id))

    def get_version(self, document_id: str, version_id: str) -> DocumentVersion:
        key = {"pk": f"doc#{document_id}", "sk": f"{VERSION_PREFIX}{version_id}"}
        try:
            item = self._table.get_item(Key=key).get("Item")
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "read document version") from exc
        if not item:
            raise NotFound(f"document {document_id} version {version_id} not found")
        return _version_from_item(item)

    def list_versions(self, document_id: str) -> tuple[DocumentVersion, ...]:
        condition = Key("pk").eq(f"doc#{document_id}") & Key("sk").begins_with(VERSION_PREFIX)
        try:
            items = self._table.query(KeyConditionExpression=condition).get("Items", [])
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "list document versions") from exc
        return tuple(_version_from_item(item) for item in items)

    def write_version(
        self,
        document: Document,
        version: DocumentVersion,
        chunks: tuple[Chunk, ...],
        expected_latest: int,
    ) -> DocumentVersion:
        """Write the S3 body (SSE-KMS), then the version and pointer items in one transaction.

        `expected_latest` is the version number the caller read; a concurrent writer makes the
        transaction fail with Conflict (optimistic concurrency).
        """
        s3_version_id = self._put_body(version, chunks)
        stored = replace(version, s3_version_id=s3_version_id)
        self._transact(
            [
                {
                    "Put": {
                        "TableName": self._table.name,
                        "Item": _typed(_version_item(stored)),
                        "ConditionExpression": "attribute_not_exists(pk)",
                    }
                },
                {"Put": self._pointer_put(document, expected_latest)},
            ],
            "write document version",
        )
        return stored

    def update_document(
        self,
        document: Document,
        expected_latest: int,
        version_status: tuple[str, str] | None = None,
    ) -> None:
        """Replace the pointer item and optionally one version's status, atomically."""
        actions: list[dict[str, Any]] = [{"Put": self._pointer_put(document, expected_latest)}]
        if version_status:
            version_id, status = version_status
            actions.append(
                {
                    "Update": {
                        "TableName": self._table.name,
                        "Key": _typed(
                            {
                                "pk": f"doc#{document.document_id}",
                                "sk": f"{VERSION_PREFIX}{version_id}",
                            }
                        ),
                        "UpdateExpression": "SET #s = :s",
                        "ConditionExpression": "attribute_exists(pk)",
                        "ExpressionAttributeNames": {"#s": "status"},
                        "ExpressionAttributeValues": {":s": status},
                    }
                }
            )
        self._transact(actions, "update document")

    def load_chunks(self, version: DocumentVersion) -> tuple[Chunk, ...]:
        """Read exactly the stored S3 object version of a document version."""
        kwargs: dict[str, Any] = {"Bucket": self._bucket, "Key": version.s3_key}
        if version.s3_version_id:
            kwargs["VersionId"] = version.s3_version_id
        try:
            body = self._s3.get_object(**kwargs)["Body"].read()
            data = json.loads(body)
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "read document body") from exc
        except (ValueError, KeyError) as exc:
            raise NotFound(f"document body for {version.ref} is unreadable") from exc
        return tuple(
            Chunk(index=c["index"], text=c["text"], sha256=c["sha256"]) for c in data["chunks"]
        )

    def _put_body(self, version: DocumentVersion, chunks: tuple[Chunk, ...]) -> str | None:
        body = json.dumps(
            {
                "document_id": version.document_id,
                "version_id": version.version_id,
                "chunks": [{"index": c.index, "sha256": c.sha256, "text": c.text} for c in chunks],
            }
        ).encode("utf-8")
        try:
            response = self._s3.put_object(
                Bucket=self._bucket,
                Key=version.s3_key,
                Body=body,
                ContentType="application/json",
                ServerSideEncryption="aws:kms",
                SSEKMSKeyId=self._kms_key_id,
            )
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "write document body") from exc
        return response.get("VersionId")

    def _pointer_put(self, document: Document, expected_latest: int) -> dict[str, Any]:
        condition, values = _pointer_condition(expected_latest)
        put: dict[str, Any] = {
            "TableName": self._table.name,
            "Item": _typed(_pointer_item(document)),
            "ConditionExpression": condition,
        }
        return {**put, "ExpressionAttributeValues": values} if values else put

    def _transact(self, actions: list[dict[str, Any]], operation: str) -> None:
        try:
            self._table.meta.client.transact_write_items(TransactItems=actions)
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, operation) from exc
