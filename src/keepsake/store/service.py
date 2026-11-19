"""Document store business rules (FR2.1, FR2.2, FR2.4, FR2.5, FR2.7, FR2.8, FR4.6, FR6.1).

- Every upload is chunked (FR2.7) and scanned for prompt injection (FR2.8). A clean version
  becomes current; a flagged version is quarantined as `pending_version_id` and the previous
  current version keeps being served until a doc-admin approves the new one.
- When the current version changes or the document is retired, cache entries that used the old
  version are invalidated (FR4.6). Invalidation failure is logged, not fatal: reuse keys contain
  the document version, so a stale entry can never be served for the new version anyway.
- Every change is written to the audit log with who, when and which version (FR6.1).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

from keepsake.common.audit import AuditLog
from keepsake.common.clock import Clock, iso, utc_now
from keepsake.common.errors import AccessDenied, Conflict, KeepsakeError, NotFound, ValidationError
from keepsake.common.models import (
    MAX_DOCUMENT_CHARS,
    MAX_TITLE_CHARS,
    AgentConfig,
    Document,
    DocumentVersion,
    require_classification,
    require_id,
    require_text,
)
from keepsake.store.access import (
    check_agent_can_read,
    check_staff_can_approve,
    check_staff_can_manage,
    check_staff_can_read,
)
from keepsake.store.chunking import changed_chunk_indexes, join_chunks, split_into_chunks
from keepsake.store.injection import scan_chunks
from keepsake.store.repository import DocumentRepository, object_key, version_id_for

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StaffIdentity:
    """A signed-in staff member (Cognito username and groups)."""

    username: str
    groups: frozenset[str]


@dataclass(frozen=True)
class NewDocument:
    document_id: str
    title: str
    owner_team: str
    classification: str
    text: str

    def __post_init__(self) -> None:
        require_id(self.document_id, "document_id")
        require_text(self.title, "title", MAX_TITLE_CHARS)
        require_id(self.owner_team, "owner_team")
        require_classification(self.classification)
        require_text(self.text, "text", MAX_DOCUMENT_CHARS)


@dataclass(frozen=True)
class LoadedVersion:
    version: DocumentVersion
    text: str


class DocumentService:
    def __init__(
        self,
        *,
        repository: DocumentRepository,
        audit: AuditLog,
        invalidate: Callable[[str], int],
        now: Clock = utc_now,
    ) -> None:
        self._repo = repository
        self._audit = audit
        self._invalidate = invalidate
        self._now = now

    # ---- staff operations (FR2.5) -------------------------------------------------------
    def create(self, staff: StaffIdentity, new: NewDocument) -> DocumentVersion:
        check_staff_can_manage(staff.groups, new.owner_team)
        if self._repo.find_document(new.document_id) is not None:
            raise Conflict(f"document {new.document_id} already exists; update it instead")
        shell = Document(
            document_id=new.document_id,
            title=new.title,
            owner_team=new.owner_team,
            classification=new.classification,
            status="quarantined",
            latest_version_number=0,
        )
        return self._add_version(staff, shell, new.text, is_new=True)

    def update(self, staff: StaffIdentity, document_id: str, text: str) -> DocumentVersion:
        require_text(text, "text", MAX_DOCUMENT_CHARS)
        document = self._repo.get_document(document_id)
        check_staff_can_manage(staff.groups, document.owner_team)
        if document.status == "retired":
            raise Conflict(f"document {document_id} is retired")
        return self._add_version(staff, document, text, is_new=False)

    def approve(self, staff: StaffIdentity, document_id: str, version_id: str) -> Document:
        check_staff_can_approve(staff.groups)
        document = self._repo.get_document(document_id)
        if document.pending_version_id != version_id:
            raise Conflict(f"version {version_id} is not awaiting approval")
        after = replace(
            document, current_version_id=version_id, pending_version_id=None, status="active"
        )
        self._repo.update_document(after, document.latest_version_number, (version_id, "active"))
        self._record(document_id, "document.approved", staff, {"version_id": version_id})
        self._invalidate_old(document, document.current_version_id)
        return after

    def retire(self, staff: StaffIdentity, document_id: str) -> Document:
        document = self._repo.get_document(document_id)
        check_staff_can_manage(staff.groups, document.owner_team)
        if document.status == "retired":
            return document
        after = replace(document, status="retired")
        self._repo.update_document(after, document.latest_version_number)
        self._record(
            document_id, "document.retired", staff, {"version_id": document.current_version_id}
        )
        self._invalidate_old(document, document.current_version_id)
        return after

    def history(self, staff: StaffIdentity, document_id: str) -> tuple[DocumentVersion, ...]:
        check_staff_can_read(staff.groups, self._repo.get_document(document_id))
        return self._repo.list_versions(document_id)

    def get_current(self, staff: StaffIdentity, document_id: str) -> LoadedVersion:
        document = self._repo.get_document(document_id)
        check_staff_can_read(staff.groups, document)
        if document.current_version_id is None:
            raise NotFound(f"document {document_id} has no approved version")
        return self._load(document_id, document.current_version_id)

    def list_documents(self, staff: StaffIdentity) -> tuple[Document, ...]:
        visible = []
        for document in self._repo.list_documents():
            try:
                check_staff_can_read(staff.groups, document)
            except AccessDenied:
                continue
            visible.append(document)
        return tuple(visible)

    def get_version(self, document_id: str, version_id: str) -> DocumentVersion:
        return self._repo.get_version(document_id, version_id)

    # ---- agent read path (FR2.2, FR2.4, FR2.8) ------------------------------------------
    def current_for_agent(self, agent: AgentConfig, document_id: str) -> LoadedVersion:
        """The current version for an agent; quarantined and retired documents are not served."""
        document = self._repo.get_document(document_id)
        if document.status == "retired":
            raise NotFound(f"document {document_id} is retired")
        if document.current_version_id is None:
            raise AccessDenied(f"document {document_id} is quarantined pending approval")
        version = self._repo.get_version(document_id, document.current_version_id)
        check_agent_can_read(agent, version)
        return LoadedVersion(version, join_chunks(self._repo.load_chunks(version)))

    # ---- internals ------------------------------------------------------------------------
    def _add_version(
        self, staff: StaffIdentity, before: Document, text: str, *, is_new: bool
    ) -> DocumentVersion:
        chunks = split_into_chunks(text)
        if not chunks:
            raise ValidationError("document text has no content")
        findings = scan_chunks(chunks)
        number = before.latest_version_number + 1
        version_id = version_id_for(number)
        status = "quarantined" if findings else "active"
        version = DocumentVersion(
            document_id=before.document_id,
            version_id=version_id,
            owner_team=before.owner_team,
            classification=before.classification,
            status=status,
            author=staff.username,
            created_at=iso(self._now()),
            s3_key=object_key(before.document_id, version_id),
            chunk_hashes=tuple(c.sha256 for c in chunks),
            injection_findings=tuple(f.describe() for f in findings),
        )
        after = self._document_after(before, version_id, status, number)
        stored = self._repo.write_version(after, version, chunks, before.latest_version_number)
        self._record(
            before.document_id,
            "document.version_created",
            staff,
            {
                "version_id": version_id,
                "status": status,
                "created": is_new,
                "injection_findings": list(version.injection_findings),
                "changed_chunks": self._changed_chunks(before, version),
            },
        )
        if status == "active":
            self._invalidate_old(before, before.current_version_id)
        return stored

    @staticmethod
    def _document_after(before: Document, version_id: str, status: str, number: int) -> Document:
        if status == "active":
            return replace(
                before,
                status="active",
                current_version_id=version_id,
                pending_version_id=None,
                latest_version_number=number,
            )
        keeps_serving = before.current_version_id is not None
        return replace(
            before,
            status="active" if keeps_serving else "quarantined",
            pending_version_id=version_id,
            latest_version_number=number,
        )

    def _changed_chunks(self, before: Document, version: DocumentVersion) -> list[int]:
        if before.current_version_id is None:
            return list(range(len(version.chunk_hashes)))
        previous = self._repo.get_version(before.document_id, before.current_version_id)
        return list(changed_chunk_indexes(previous.chunk_hashes, version.chunk_hashes))

    def _load(self, document_id: str, version_id: str) -> LoadedVersion:
        version = self._repo.get_version(document_id, version_id)
        return LoadedVersion(version, join_chunks(self._repo.load_chunks(version)))

    def _record(
        self, document_id: str, event: str, staff: StaffIdentity, details: dict[str, object]
    ) -> None:
        self._audit.record(f"doc#{document_id}", event, staff.username, details)

    def _invalidate_old(self, document: Document, old_version_id: str | None) -> None:
        if old_version_id is None:
            return
        ref = f"{document.document_id}@{old_version_id}"
        try:
            removed = self._invalidate(ref)
            LOGGER.info("invalidated cache entries", extra={"document": ref, "removed": removed})
        except KeepsakeError as exc:
            LOGGER.error(
                "cache invalidation failed; entries expire by TTL",
                extra={"document": ref, "error": exc.message},
            )
