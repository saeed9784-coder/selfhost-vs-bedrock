"""Monthly budget and storage-versus-savings alerts via SNS (FR5.5, FR5.6).

Run daily (EventBridge schedule). For each configured agent, month-to-date usage is compared
with the agent's monthly budget (alert at >= 80 %) and its cache storage cost with its savings
(alert when storage exceeds savings).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from keepsake.common.aws import AWS_ERRORS, translate_aws_error
from keepsake.common.models import AgentConfig, UsageRecord

LOGGER = logging.getLogger(__name__)
BUDGET_ALERT_THRESHOLD = 0.8


@dataclass(frozen=True)
class Alert:
    kind: str  # "budget" | "storage-exceeds-savings"
    agent_id: str
    team: str
    message: str
    spend_usd: float
    limit_usd: float


def check_agent(agent: AgentConfig, month_records: Iterable[UsageRecord]) -> tuple[Alert, ...]:
    records = [r for r in month_records if r.agent_id == agent.agent_id]
    spend = sum(r.cost_usd + r.storage_cost_usd for r in records)
    storage = sum(r.storage_cost_usd for r in records)
    saved = sum(r.saving_usd for r in records)
    alerts: list[Alert] = []
    budget = agent.monthly_budget_usd
    if budget > 0 and spend >= BUDGET_ALERT_THRESHOLD * budget:
        alerts.append(
            Alert(
                kind="budget",
                agent_id=agent.agent_id,
                team=agent.team,
                spend_usd=spend,
                limit_usd=budget,
                message=f"{agent.agent_id} has spent ${spend:,.2f} of its ${budget:,.2f} monthly "
                f"budget ({spend / budget:.0%})",
            )
        )
    if storage > 0 and storage > saved:
        alerts.append(
            Alert(
                kind="storage-exceeds-savings",
                agent_id=agent.agent_id,
                team=agent.team,
                spend_usd=storage,
                limit_usd=saved,
                message=f"{agent.agent_id} cache storage cost ${storage:,.4f} exceeds its savings "
                f"${saved:,.4f} this month",
            )
        )
    return tuple(alerts)


def publish_alerts(sns: Any, topic_arn: str, alerts: Iterable[Alert]) -> int:
    """Publish each alert; raises DependencyUnavailable if SNS fails (the schedule retries)."""
    count = 0
    for alert in alerts:
        try:
            sns.publish(
                TopicArn=topic_arn,
                Subject=f"Keepsake {alert.kind} alert: {alert.agent_id}"[:100],
                Message=json.dumps(
                    {
                        "kind": alert.kind,
                        "agent_id": alert.agent_id,
                        "team": alert.team,
                        "message": alert.message,
                        "value_usd": round(alert.spend_usd, 4),
                        "limit_usd": round(alert.limit_usd, 4),
                    }
                ),
            )
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "publish alert") from exc
        count += 1
    LOGGER.info("alerts published", extra={"count": count})
    return count
