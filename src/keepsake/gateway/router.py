"""Resolve the model for an agent and enforce model blocks (FR3.3, FR3.4, FR3.5, FR3.7).

The agent never names a model: the model comes from the agent's SSM configuration, so switching
models is a configuration change and rollback is the reverse change. A model listed in
`models/blocks` for the customer's group is rejected immediately (ModelBlocked, audited by the
gateway service).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from keepsake.common.config import ConfigProvider
from keepsake.common.errors import ConfigurationError, ModelBlocked
from keepsake.common.models import AgentConfig, ModelCapability

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Route:
    agent: AgentConfig
    capability: ModelCapability
    shadow: ModelCapability | None = None


class Router:
    def __init__(self, config: ConfigProvider) -> None:
        self._config = config

    def route(self, agent_id: str, customer_id: str) -> Route:
        agent = self._config.agent_config(agent_id)
        capabilities = self._config.capabilities()
        capability = self._capability(capabilities, agent.model_id)
        group = self.blocking_group(agent.model_id, customer_id)
        if group is not None:
            raise ModelBlocked(
                f"model {agent.model_id} is blocked for this customer",
                model_id=agent.model_id,
                customer_group=group,
            )
        return Route(
            agent=agent,
            capability=capability,
            shadow=self._shadow(agent, customer_id, capabilities),
        )

    def blocking_group(self, model_id: str, customer_id: str) -> str | None:
        """The first customer group that contains the customer and blocks the model."""
        for group, spec in sorted(self._config.model_blocks().get("groups", {}).items()):
            if not isinstance(spec, dict):
                raise ConfigurationError(f"model block group {group} is invalid")
            customers = spec.get("customers", [])
            in_group = "*" in customers or customer_id in customers
            if in_group and model_id in spec.get("blocked_models", []):
                return str(group)
        return None

    def _shadow(
        self, agent: AgentConfig, customer_id: str, capabilities: dict[str, ModelCapability]
    ) -> ModelCapability | None:
        if not agent.shadow_model_id:
            return None
        if self.blocking_group(agent.shadow_model_id, customer_id) is not None:
            LOGGER.warning(
                "shadow model blocked for customer group; shadow call skipped",
                extra={"agent": agent.agent_id, "model": agent.shadow_model_id},
            )
            return None
        return self._capability(capabilities, agent.shadow_model_id)

    @staticmethod
    def _capability(capabilities: dict[str, ModelCapability], model_id: str) -> ModelCapability:
        capability = capabilities.get(model_id)
        if capability is None:
            raise ConfigurationError(f"model {model_id} is not in the capability table")
        return capability
