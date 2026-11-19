"""SSM-backed configuration tests (FR3.5, FR3.8, NFR5)."""

from __future__ import annotations

import json

import pytest

from keepsake.common.config import ConfigProvider, role_name_from_arn
from keepsake.common.errors import AccessDenied, ConfigurationError
from tests.conftest import AGENT_CONFIGS, NOVA, AwsResources


class FakeMonotonic:
    def __init__(self) -> None:
        self.value = 1000.0

    def __call__(self) -> float:
        return self.value


def test_changed_ssm_model_is_visible_after_the_ttl(aws: AwsResources) -> None:
    clock = FakeMonotonic()
    provider = ConfigProvider(ssm=aws.ssm, ttl_seconds=30, monotonic=clock)
    assert provider.agent_config("helpdesk").model_id == "deepseek.v3.2"

    switched = {**AGENT_CONFIGS["helpdesk"], "model_id": NOVA}
    aws.ssm.put_parameter(
        Name="/keepsake/agents/helpdesk/config",
        Type="String",
        Value=json.dumps(switched),
        Overwrite=True,
    )

    clock.value += 10
    assert provider.agent_config("helpdesk").model_id == "deepseek.v3.2"  # still cached
    clock.value += 25
    assert provider.agent_config("helpdesk").model_id == NOVA  # within 60 s (NFR5)


def test_caller_role_arns_resolve_to_registered_agents(aws: AwsResources) -> None:
    provider = ConfigProvider(ssm=aws.ssm)

    assumed = "arn:aws:sts::123456789012:assumed-role/keepsake-agent-helpdesk/session-1"
    assert provider.resolve_agent(assumed) == "helpdesk"
    with pytest.raises(AccessDenied):
        provider.resolve_agent("arn:aws:sts::123456789012:assumed-role/intruder/session")
    with pytest.raises(AccessDenied):
        role_name_from_arn("arn:aws:iam::123456789012:user/alice")


def test_capability_table_comes_from_file_and_ssm_overrides_it(aws: AwsResources) -> None:
    provider = ConfigProvider(ssm=aws.ssm, ttl_seconds=0)
    assert provider.capabilities()["deepseek.v3.2"].supports_prompt_cache is False

    override = {
        "models": {
            "deepseek.v3.2": {
                "family": "deepseek",
                "model_version": "v3.2-b",
                "supports_prompt_cache": True,
                "min_cache_tokens": 2048,
            }
        }
    }
    aws.ssm.put_parameter(
        Name="/keepsake/models/capabilities", Type="String", Value=json.dumps(override)
    )

    assert provider.capabilities()["deepseek.v3.2"].supports_prompt_cache is True


def test_invalid_agent_configuration_is_a_configuration_error(aws: AwsResources) -> None:
    bad = {**AGENT_CONFIGS["atlas"], "retention_hours": 500}
    aws.ssm.put_parameter(
        Name="/keepsake/agents/atlas/config", Type="String", Value=json.dumps(bad), Overwrite=True
    )

    with pytest.raises(ConfigurationError):
        ConfigProvider(ssm=aws.ssm).agent_config("atlas")
