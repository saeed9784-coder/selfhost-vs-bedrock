"""GET /usage?scope=team|company&team=&from=&to=&group_by=agent|team&bucket_hours=24 (FR5.3, FR5.4).

Team leads (`team-<name>`) only ever receive their own team's records; `finance` may request the
company scope or any team. The scope is decided from the verified Cognito groups, never from
anything the caller can set freely.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from keepsake.common.clock import iso, parse_iso, utc_now
from keepsake.common.errors import ValidationError
from keepsake.common.http import Event, Response, api_handler, ok, query_params, staff_identity
from keepsake.common.models import require_id
from keepsake.handlers.dependencies import container
from keepsake.handlers.observability import LOGGER
from keepsake.usage.aggregation import hit_rate_over_time, summarize, totals
from keepsake.usage.scope import resolve_usage_scope

MAX_BUCKET_HOURS = 24 * 31


def _time(value: str | None, default: datetime) -> str:
    if not value:
        return iso(default)
    try:
        return iso(parse_iso(value if "T" in value else f"{value}T00:00:00Z"))
    except ValueError as exc:
        raise ValidationError("from/to must be ISO-8601 dates or timestamps") from exc


def _bucket_hours(value: str | None) -> int:
    try:
        hours = int(value) if value else 24
    except ValueError as exc:
        raise ValidationError("bucket_hours must be an integer") from exc
    if not 1 <= hours <= MAX_BUCKET_HOURS:
        raise ValidationError(f"bucket_hours must be between 1 and {MAX_BUCKET_HOURS}")
    return hours


@LOGGER.inject_lambda_context
@api_handler("usage")
def handler(event: Event, context: Any) -> Response:
    staff = staff_identity(event)
    params = query_params(event)
    team = require_id(params["team"], "team") if params.get("team") else None
    scope = resolve_usage_scope(staff.groups, params.get("scope", "team"), team)
    now = utc_now()
    start = _time(params.get("from"), now.replace(day=1, hour=0, minute=0, second=0, microsecond=0))
    end = _time(params.get("to"), now)
    if start > end:
        raise ValidationError("from must not be after to")
    repo = container().usage
    records = (
        repo.query_company(start, end)
        if scope.team is None
        else repo.query_team(scope.team, start, end)
    )
    return ok(
        {
            "scope": "company" if scope.team is None else "team",
            "team": scope.team,
            "from": start,
            "to": end,
            "totals": asdict(totals(records)),
            "groups": [asdict(s) for s in summarize(records, params.get("group_by", "agent"))],
            "hit_rate_over_time": [
                asdict(p)
                for p in hit_rate_over_time(records, _bucket_hours(params.get("bucket_hours")))
            ],
        }
    )
