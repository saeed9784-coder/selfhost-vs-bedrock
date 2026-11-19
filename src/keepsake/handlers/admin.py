"""Administrative API (FR3.7, FR4.7, FR6.1, FR6.2); Cognito-authorized, `doc-admin` only
(audit lookup also allows `finance`).

DELETE /cache?customer=<id> | ?agent=<id> | ?all=true   delete reuse entries, returns the count
POST   /erasure/{customer}                              erase a customer, returns the report
PUT    /models/blocks                                   replace the model block list
GET    /audit/{requestId}                               audit events + document versions/editors
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from keepsake.common.aws import AWS_ERRORS, translate_aws_error
from keepsake.common.errors import AccessDenied, KeepsakeError, NotFound, ValidationError
from keepsake.common.http import (
    Event,
    Response,
    api_handler,
    ok,
    parse_body,
    path_param,
    query_params,
    staff_identity,
)
from keepsake.common.models import require_id
from keepsake.handlers.dependencies import Container, container
from keepsake.handlers.observability import LOGGER
from keepsake.store.access import DOC_ADMIN_GROUP, FINANCE_GROUP
from keepsake.store.service import StaffIdentity

Route = Callable[[Container, StaffIdentity, Event], Response]
MAX_BLOCK_GROUPS = 50
MAX_GROUP_ENTRIES = 500


def _require_group(staff: StaffIdentity, *allowed: str) -> None:
    if not staff.groups.intersection(allowed):
        raise AccessDenied(f"requires one of the groups: {', '.join(allowed)}")


def _delete_cache(deps: Container, staff: StaffIdentity, event: Event) -> Response:
    _require_group(staff, DOC_ADMIN_GROUP)
    params = query_params(event)
    selectors = [k for k in ("customer", "agent", "all") if params.get(k)]
    if len(selectors) != 1:
        raise ValidationError("give exactly one of customer, agent or all=true")
    selector = selectors[0]
    if selector == "all":
        if params["all"] != "true":
            raise ValidationError("all must be 'true'")
        deleted = deps.cache.delete_all()
    elif selector == "customer":
        deleted = deps.cache.delete_by_customer(require_id(params["customer"], "customer"))
    else:
        deleted = deps.cache.delete_by_agent(require_id(params["agent"], "agent"))
    deps.audit.record(
        f"cache#{selector}",
        "cache.deleted",
        staff.username,
        {"selector": selector, "deleted": deleted},
    )
    return ok({"deleted": deleted, "selector": selector})


def _erase(deps: Container, staff: StaffIdentity, event: Event) -> Response:
    _require_group(staff, DOC_ADMIN_GROUP)
    report = deps.erasure.erase_customer(path_param(event, "customer"), staff.username)
    return ok(asdict(report))


def validate_blocks(body: dict[str, Any]) -> dict[str, Any]:
    groups = body.get("groups")
    if not isinstance(groups, dict) or len(groups) > MAX_BLOCK_GROUPS:
        raise ValidationError(f"groups must be an object with at most {MAX_BLOCK_GROUPS} entries")
    clean: dict[str, Any] = {}
    for name, spec in groups.items():
        require_id(name, "group name")
        if not isinstance(spec, dict):
            raise ValidationError(f"group {name} must be an object")
        customers, models = spec.get("customers", []), spec.get("blocked_models", [])
        if (
            not isinstance(customers, list)
            or not isinstance(models, list)
            or len(customers) > MAX_GROUP_ENTRIES
            or len(models) > MAX_GROUP_ENTRIES
        ):
            raise ValidationError(f"group {name} needs customers and blocked_models lists")
        clean[name] = {
            "customers": [c if c == "*" else require_id(c, "customer") for c in customers],
            "blocked_models": [require_id(m, "model id") for m in models],
        }
    return {"groups": clean}


def _put_blocks(deps: Container, staff: StaffIdentity, event: Event) -> Response:
    _require_group(staff, DOC_ADMIN_GROUP)
    blocks = validate_blocks(parse_body(event))
    name = f"{deps.settings.ssm_prefix}/models/blocks"
    try:
        deps.ssm.put_parameter(Name=name, Value=json.dumps(blocks), Type="String", Overwrite=True)
    except AWS_ERRORS as exc:
        raise translate_aws_error(exc, "update model blocks") from exc
    deps.audit.record("config#model-blocks", "models.blocks_updated", staff.username, blocks)
    return ok(blocks)


def _document_editors(deps: Container, refs: list[str]) -> list[dict[str, Any]]:
    editors = []
    for ref in refs:
        document_id, _, version_id = ref.partition("@")
        try:
            version = deps.documents.get_version(document_id, version_id)
            editors.append(
                {
                    "document_version": ref,
                    "author": version.author,
                    "created_at": version.created_at,
                }
            )
        except KeepsakeError as exc:
            LOGGER.warning("audit enrichment failed", extra={"ref": ref, "error": exc.message})
            editors.append({"document_version": ref, "author": None, "created_at": None})
    return editors


def _audit(deps: Container, staff: StaffIdentity, event: Event) -> Response:
    _require_group(staff, DOC_ADMIN_GROUP, FINANCE_GROUP)
    request_id = require_id(path_param(event, "requestId"), "requestId")
    events = deps.audit.lookup_request(request_id)
    if not events:
        raise NotFound(f"no audit events for request {request_id}")
    refs = sorted({ref for e in events for ref in e.details.get("document_versions", [])})
    return ok(
        {
            "request_id": request_id,
            "events": [asdict(e) for e in events],
            "documents": _document_editors(deps, refs),
        }
    )


ROUTES: dict[tuple[str, str], Route] = {
    ("DELETE", "/cache"): _delete_cache,
    ("POST", "/erasure/{customer}"): _erase,
    ("PUT", "/models/blocks"): _put_blocks,
    ("GET", "/audit/{requestId}"): _audit,
}


@LOGGER.inject_lambda_context
@api_handler("admin")
def handler(event: Event, context: Any) -> Response:
    route = ROUTES.get((str(event.get("httpMethod")), str(event.get("resource"))))
    if route is None:
        raise NotFound("route not found")
    return route(container(), staff_identity(event), event)
