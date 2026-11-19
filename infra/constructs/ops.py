"""Operations: SSM configuration, SNS alerts, CloudWatch dashboard and alarms, schedules
(FR3.5, FR3.7, FR3.8, FR5.2, FR5.5, NFR5, NFR7).

Agent configuration lives in SSM as JSON. Switching an agent's model is one parameter change
(`/keepsake/agents/<agent>/config`), live within the 30-second config cache (NFR5); rollback is
the same change reversed. The capability and price tables are seeded from `config/*.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aws_cdk import Duration
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subscriptions
from aws_cdk import aws_ssm as ssm
from constructs import Construct

from infra.constructs.auth import AGENTS, agent_role_name

SSM_PREFIX = "/keepsake"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
NAMESPACE = "Keepsake"
COMPONENTS = ("gateway", "documents", "usage", "admin", "analyzer", "budget")
ERROR_RATE_THRESHOLD_PERCENT = 5

DEFAULT_AGENT_CONFIGS: dict[str, dict[str, Any]] = {
    "reviewer": {
        "team": "reviewer",
        "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "retention_hours": 72,
        "monthly_budget_usd": 500.0,
        "allowed_classifications": ["policy", "runbook", "source-code", "product-doc"],
        "system_prompt": "You review pull requests for Groundhog Labs.",
    },
    "helpdesk": {
        "team": "helpdesk",
        "model_id": "deepseek.v3.2",
        "retention_hours": 72,
        "monthly_budget_usd": 300.0,
        "allowed_classifications": [
            "policy",
            "runbook",
            "product-doc",
            "contract",
            "customer-data",
        ],
        "system_prompt": "You answer customer support questions for Groundhog Labs.",
    },
    "atlas": {
        "team": "atlas",
        "model_id": "us.amazon.nova-pro-v1:0",
        "retention_hours": 24,
        "monthly_budget_usd": 300.0,
        "allowed_classifications": ["policy", "runbook", "product-doc", "contract"],
        "system_prompt": "You are Atlas, the Groundhog Labs internal knowledge assistant.",
    },
}


def load_config_file(name: str) -> dict[str, Any]:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


class OpsConstruct(Construct):
    def __init__(
        self, scope: Construct, construct_id: str, *, key: kms.IKey, alarm_email: str = ""
    ) -> None:
        super().__init__(scope, construct_id)
        self.parameters = self._parameters()
        self.topic = sns.Topic(
            self, "Alerts", display_name="Keepsake alerts", master_key=key, enforce_ssl=True
        )
        if alarm_email:
            self.topic.add_subscription(subscriptions.EmailSubscription(alarm_email))

    def _parameter(self, name: str, value: dict[str, Any], description: str) -> ssm.StringParameter:
        construct_id = name.strip("/").replace("/", "-")
        return ssm.StringParameter(
            self,
            construct_id,
            parameter_name=f"{SSM_PREFIX}/{name}",
            string_value=json.dumps(value, separators=(",", ":")),
            description=description,
            tier=ssm.ParameterTier.STANDARD,
        )

    def _parameters(self) -> dict[str, ssm.StringParameter]:
        params = {
            "registry": self._parameter(
                "agents/registry",
                {agent_role_name(a): a for a in AGENTS},
                "IAM role name -> Keepsake agent id",
            ),
            "blocks": self._parameter(
                "models/blocks", {"groups": {}}, "Models blocked per customer group (FR3.7)"
            ),
            "capabilities": self._parameter(
                "models/capabilities",
                load_config_file("model_capabilities.json"),
                "Model caching capabilities (FR3.8)",
            ),
            "prices": self._parameter(
                "models/prices",
                load_config_file("prices.json"),
                "Model and storage prices (FR1.4, FR5.1)",
            ),
        }
        for agent, config in DEFAULT_AGENT_CONFIGS.items():
            params[f"agent-{agent}"] = self._parameter(
                f"agents/{agent}/config",
                {"agent_id": agent, "shadow_model_id": None, **config},
                f"Configuration for agent {agent}: model, retention, budget (FR3.5)",
            )
        return params

    def add_monitoring(
        self,
        functions: dict[str, lambda_.IFunction],
        schedule_targets: dict[str, lambda_.IFunction],
    ) -> None:
        action = cw_actions.SnsAction(self.topic)
        for component in COMPONENTS:
            self._error_rate_alarm(component).add_alarm_action(action)
        degraded = cloudwatch.Alarm(
            self,
            "DegradedAlarm",
            metric=cloudwatch.Metric(
                namespace=NAMESPACE,
                metric_name="Degraded",
                dimensions_map={"component": "gateway"},
                statistic="Sum",
                period=Duration.minutes(5),
            ),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="Gateway served requests in degraded mode (store or cache down)",
        )
        degraded.add_alarm_action(action)
        for name, fn in functions.items():
            fn.metric_errors(period=Duration.minutes(5)).create_alarm(
                self,
                f"LambdaErrors-{name}",
                threshold=1,
                evaluation_periods=1,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ).add_alarm_action(action)
        self._dashboard()
        events.Rule(
            self,
            "DailyBudgetCheck",
            schedule=events.Schedule.cron(minute="0", hour="6"),
            targets=[targets.LambdaFunction(schedule_targets["budget"])],
        )
        events.Rule(
            self,
            "WeeklyPhase0",
            schedule=events.Schedule.cron(minute="0", hour="5", week_day="MON"),
            targets=[
                targets.LambdaFunction(
                    schedule_targets["analyzer"],
                    event=events.RuleTargetInput.from_object({"prefix": "AWSLogs/"}),
                )
            ],
        )

    def _component_metric(self, component: str, name: str) -> cloudwatch.Metric:
        return cloudwatch.Metric(
            namespace=NAMESPACE,
            metric_name=name,
            dimensions_map={"component": component},
            statistic="Sum",
            period=Duration.minutes(5),
        )

    def _error_rate_alarm(self, component: str) -> cloudwatch.Alarm:
        rate = cloudwatch.MathExpression(
            expression="IF(requests > 0, 100 * errors / requests, 0)",
            using_metrics={
                "errors": self._component_metric(component, "Errors"),
                "requests": self._component_metric(component, "Requests"),
            },
            period=Duration.minutes(5),
            label=f"{component} error rate (%)",
        )
        return cloudwatch.Alarm(
            self,
            f"ErrorRate-{component}",
            metric=rate,
            threshold=ERROR_RATE_THRESHOLD_PERCENT,
            evaluation_periods=2,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description=f"{component} error rate above {ERROR_RATE_THRESHOLD_PERCENT}%",
        )

    def _agent_metric(self, agent: str, name: str, statistic: str = "Sum") -> cloudwatch.Metric:
        return cloudwatch.Metric(
            namespace=NAMESPACE,
            metric_name=name,
            dimensions_map={"agent": agent},
            statistic=statistic,
            period=Duration.hours(1),
            label=agent,
        )

    def _dashboard(self) -> None:
        dashboard = cloudwatch.Dashboard(self, "Dashboard", dashboard_name="Keepsake")

        def per_agent(title: str, metric: str, statistic: str = "Sum") -> cloudwatch.GraphWidget:
            return cloudwatch.GraphWidget(
                title=title,
                width=12,
                left=[self._agent_metric(a, metric, statistic) for a in AGENTS],
            )

        dashboard.add_widgets(
            per_agent("Requests per agent", "GatewayRequests"),
            per_agent("Cache hits per agent", "CacheHits"),
        )
        dashboard.add_widgets(
            per_agent("Input tokens per agent", "InputTokens"),
            per_agent("Cached tokens per agent", "CachedTokens"),
        )
        dashboard.add_widgets(
            per_agent("Cost (USD) per agent", "CostUSD"),
            per_agent("Saving (USD) per agent", "SavingUSD"),
        )
        dashboard.add_widgets(
            per_agent("p99 latency (ms)", "Latency", "p99"),
            cloudwatch.GraphWidget(
                title="Errors and degraded calls",
                width=12,
                left=[
                    *(self._component_metric(c, "Errors") for c in COMPONENTS),
                    self._component_metric("gateway", "Degraded"),
                ],
            ),
        )
