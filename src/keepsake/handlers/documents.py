"""Staff document API (FR2.5, FR2.8); Cognito-authorized.

GET    /documents                      list documents the caller may read
POST   /documents                      create {document_id, title, owner_team, classification, text}
GET    /documents/{id}                 current version metadata and text
PUT    /documents/{id}                 new version {text}
DELETE /documents/{id}                 retire
GET    /documents/{id}/versions        version history
POST   /documents/{id}/approve         approve a quarantined version {version_id} (doc-admin)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from keepsake.common.errors import NotFound
from keepsake.common.http import (
    Event,
    Response,
    api_handler,
    ok,
    parse_body,
    path_param,
    staff_identity,
)
from keepsake.common.models import require_id
from keepsake.handlers.dependencies import container
from keepsake.handlers.observability import LOGGER
from keepsake.store.service import DocumentService, NewDocument, StaffIdentity

Route = Callable[[DocumentService, StaffIdentity, Event], Response]


def _list(service: DocumentService, staff: StaffIdentity, event: Event) -> Response:
    return ok({"documents": [asdict(d) for d in service.list_documents(staff)]})


def _create(service: DocumentService, staff: StaffIdentity, event: Event) -> Response:
    body = parse_body(event)
    new = NewDocument(
        document_id=body.get("document_id"),
        title=body.get("title"),
        owner_team=body.get("owner_team"),
        classification=body.get("classification"),
        text=body.get("text"),
    )
    return ok({"version": asdict(service.create(staff, new))}, status=201)


def _document_id(event: Event) -> str:
    return require_id(path_param(event, "id"), "document id")


def _get(service: DocumentService, staff: StaffIdentity, event: Event) -> Response:
    loaded = service.get_current(staff, _document_id(event))
    return ok({"version": asdict(loaded.version), "text": loaded.text})


def _update(service: DocumentService, staff: StaffIdentity, event: Event) -> Response:
    version = service.update(staff, _document_id(event), parse_body(event).get("text"))
    return ok({"version": asdict(version)})


def _retire(service: DocumentService, staff: StaffIdentity, event: Event) -> Response:
    return ok({"document": asdict(service.retire(staff, _document_id(event)))})


def _history(service: DocumentService, staff: StaffIdentity, event: Event) -> Response:
    versions = service.history(staff, _document_id(event))
    return ok({"versions": [asdict(v) for v in versions]})


def _approve(service: DocumentService, staff: StaffIdentity, event: Event) -> Response:
    version_id = require_id(parse_body(event).get("version_id"), "version_id")
    return ok({"document": asdict(service.approve(staff, _document_id(event), version_id))})


ROUTES: dict[tuple[str, str], Route] = {
    ("GET", "/documents"): _list,
    ("POST", "/documents"): _create,
    ("GET", "/documents/{id}"): _get,
    ("PUT", "/documents/{id}"): _update,
    ("DELETE", "/documents/{id}"): _retire,
    ("GET", "/documents/{id}/versions"): _history,
    ("POST", "/documents/{id}/approve"): _approve,
}


@LOGGER.inject_lambda_context
@api_handler("documents")
def handler(event: Event, context: Any) -> Response:
    route = ROUTES.get((str(event.get("httpMethod")), str(event.get("resource"))))
    if route is None:
        raise NotFound("route not found")
    staff = staff_identity(event)
    return route(container().documents, staff, event)
