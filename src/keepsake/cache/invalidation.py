"""Invalidate reuse entries when a document version is superseded (FR4.6)."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from keepsake.cache.repository import CacheRepository
from keepsake.common.errors import ValidationError

LOGGER = logging.getLogger(__name__)
_DOCUMENT_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}@[0-9]{6}$")


def make_invalidator(cache: CacheRepository) -> Callable[[str], int]:
    """A callable the document service uses: `doc@version` -> number of entries removed."""

    def invalidate(document_version: str) -> int:
        return invalidate_document_version(cache, document_version)

    return invalidate


def invalidate_document_version(cache: CacheRepository, document_version: str) -> int:
    if not _DOCUMENT_REF.fullmatch(document_version):
        raise ValidationError("document version must look like <document_id>@<6-digit version>")
    removed = cache.delete_by_document_version(document_version)
    LOGGER.info(
        "cache invalidated", extra={"document_version": document_version, "removed": removed}
    )
    return removed
