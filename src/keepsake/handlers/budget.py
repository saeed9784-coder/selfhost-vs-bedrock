"""Daily scheduled budget and storage-versus-savings check (FR5.5, FR5.6).

Triggered by an EventBridge schedule. Failures raise so the invocation is marked failed, counted
by the Lambda error metric and alarmed on; the next daily run re-checks from scratch.
"""

from __future__ import annotations

from typing import Any

from keepsake.common.clock import iso, utc_now
from keepsake.handlers.dependencies import container
from keepsake.handlers.observability import LOGGER
from keepsake.usage.budget import check_agent, publish_alerts


@LOGGER.inject_lambda_context
def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    deps = container()
    now = utc_now()
    start = iso(now.replace(day=1, hour=0, minute=0, second=0, microsecond=0))
    end = iso(now)
    alerts = []
    is_error = True
    try:
        for agent_id in deps.config.agent_ids():
            agent = deps.config.agent_config(agent_id)
            records = deps.usage.query_team(agent.team, start, end)
            alerts.extend(check_agent(agent, records))
        published = publish_alerts(deps.sns, deps.settings.alerts_topic_arn, alerts)
        is_error = False
    finally:
        deps.metrics.component_request("budget", is_error=is_error)
    return {"alerts": published, "kinds": sorted({a.kind for a in alerts})}
