"""Shared fixtures: fake AWS credentials, moto-backed resources and a Stubber-based fake Bedrock.

No real AWS account is touched: every AWS call goes to moto, and Bedrock Runtime calls go to a
botocore Stubber that returns queued Converse responses.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber
from moto import mock_aws

REGION = "us-east-1"
FIXED_NOW = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)

TABLE_NAMES = {
    "documents": "keepsake-test-documents",
    "cache": "keepsake-test-cache",
    "usage": "keepsake-test-usage",
    "audit": "keepsake-test-audit",
}
DOCUMENTS_BUCKET = "keepsake-test-documents"
REPORTS_BUCKET = "keepsake-test-reports"
SSM_PREFIX = "/keepsake"

CLAUDE = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
NOVA = "us.amazon.nova-pro-v1:0"
DEEPSEEK = "deepseek.v3.2"

AGENT_CONFIGS: dict[str, dict[str, Any]] = {
    "reviewer": {
        "agent_id": "reviewer",
        "team": "reviewer",
        "model_id": CLAUDE,
        "shadow_model_id": None,
        "retention_hours": 72,
        "monthly_budget_usd": 100.0,
        "allowed_classifications": ["policy", "runbook", "source-code", "product-doc"],
        "system_prompt": "You review pull requests for Groundhog Labs.",
    },
    "helpdesk": {
        "agent_id": "helpdesk",
        "team": "helpdesk",
        "model_id": DEEPSEEK,
        "shadow_model_id": None,
        "retention_hours": 72,
        "monthly_budget_usd": 50.0,
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
        "agent_id": "atlas",
        "team": "atlas",
        "model_id": NOVA,
        "shadow_model_id": None,
        "retention_hours": 24,
        "monthly_budget_usd": 80.0,
        "allowed_classifications": ["policy", "runbook", "product-doc", "contract"],
        "system_prompt": "You are Atlas, the internal knowledge assistant.",
    },
}
AGENT_REGISTRY = {
    "keepsake-agent-reviewer": "reviewer",
    "keepsake-agent-helpdesk": "helpdesk",
    "keepsake-agent-atlas": "atlas",
}


@pytest.fixture(autouse=True)
def fake_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials and fixed resource names for every test."""
    env = {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SECURITY_TOKEN": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": REGION,
        "AWS_REGION": REGION,
        "KEEPSAKE_DOCUMENTS_TABLE": TABLE_NAMES["documents"],
        "KEEPSAKE_CACHE_TABLE": TABLE_NAMES["cache"],
        "KEEPSAKE_USAGE_TABLE": TABLE_NAMES["usage"],
        "KEEPSAKE_AUDIT_TABLE": TABLE_NAMES["audit"],
        "KEEPSAKE_DOCUMENTS_BUCKET": DOCUMENTS_BUCKET,
        "KEEPSAKE_REPORTS_BUCKET": REPORTS_BUCKET,
        "KEEPSAKE_SSM_PREFIX": SSM_PREFIX,
        "KEEPSAKE_ALERTS_TOPIC_ARN": f"arn:aws:sns:{REGION}:123456789012:keepsake-alerts",
        "KEEPSAKE_KMS_KEY_ID": "unset",
        "KEEPSAKE_BUILD_RUN_COST_USD": "500",
        "KEEPSAKE_ALLOWED_ORIGIN": "https://keepsake.example.com",
        "POWERTOOLS_SERVICE_NAME": "keepsake-test",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _create_tables(ddb: Any, kms_key_id: str) -> dict[str, Any]:
    sse = {"Enabled": True, "SSEType": "KMS", "KMSMasterKeyId": kms_key_id}
    attr = [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
    ]
    keys = [{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}]

    def gsi(name: str, hash_key: str, range_key: str | None = None) -> dict[str, Any]:
        schema = [{"AttributeName": hash_key, "KeyType": "HASH"}]
        if range_key:
            schema.append({"AttributeName": range_key, "KeyType": "RANGE"})
        return {"IndexName": name, "KeySchema": schema, "Projection": {"ProjectionType": "ALL"}}

    def s_attr(name: str) -> dict[str, str]:
        return {"AttributeName": name, "AttributeType": "S"}

    tables: dict[str, Any] = {}
    tables["documents"] = ddb.create_table(
        TableName=TABLE_NAMES["documents"],
        KeySchema=keys,
        AttributeDefinitions=attr,
        BillingMode="PAY_PER_REQUEST",
        SSESpecification=sse,
    )
    tables["cache"] = ddb.create_table(
        TableName=TABLE_NAMES["cache"],
        KeySchema=keys,
        AttributeDefinitions=[*attr, s_attr("doc_ref"), s_attr("customer_id"), s_attr("agent_id")],
        GlobalSecondaryIndexes=[
            gsi("by-doc", "doc_ref", "sk"),
            gsi("by-customer", "customer_id", "sk"),
            gsi("by-agent", "agent_id", "sk"),
        ],
        BillingMode="PAY_PER_REQUEST",
        SSESpecification=sse,
    )
    tables["usage"] = ddb.create_table(
        TableName=TABLE_NAMES["usage"],
        KeySchema=keys,
        AttributeDefinitions=[*attr, s_attr("customer_id"), s_attr("agent_id")],
        GlobalSecondaryIndexes=[
            gsi("by-agent", "agent_id", "sk"),
            gsi("by-customer", "customer_id", "sk"),
        ],
        BillingMode="PAY_PER_REQUEST",
        SSESpecification=sse,
    )
    tables["audit"] = ddb.create_table(
        TableName=TABLE_NAMES["audit"],
        KeySchema=keys,
        AttributeDefinitions=attr,
        BillingMode="PAY_PER_REQUEST",
        SSESpecification=sse,
    )
    return tables


def _put_ssm_parameters(ssm: Any) -> None:
    ssm.put_parameter(
        Name=f"{SSM_PREFIX}/agents/registry", Type="String", Value=json.dumps(AGENT_REGISTRY)
    )
    for agent_id, config in AGENT_CONFIGS.items():
        ssm.put_parameter(
            Name=f"{SSM_PREFIX}/agents/{agent_id}/config", Type="String", Value=json.dumps(config)
        )
    ssm.put_parameter(
        Name=f"{SSM_PREFIX}/models/blocks", Type="String", Value=json.dumps({"groups": {}})
    )


@dataclass(frozen=True)
class AwsResources:
    """Handles to the moto-backed resources one test uses."""

    ddb: Any
    s3: Any
    ssm: Any
    sns: Any
    kms_key_id: str
    tables: dict[str, Any]
    topic_arn: str


@pytest.fixture(scope="session")
def _moto_session() -> Iterator[str]:
    """One moto session for the run (moto's per-session service start-up is slow); yields a
    KMS key id. Per-test state is created and destroyed by the `aws` fixture below."""
    saved = {k: os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")}
    os.environ.update(AWS_ACCESS_KEY_ID="testing", AWS_SECRET_ACCESS_KEY="testing")
    with mock_aws():
        kms = boto3.client("kms", region_name=REGION)
        yield kms.create_key(Description="keepsake-test")["KeyMetadata"]["KeyId"]
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _empty_bucket(s3: Any, bucket: str) -> None:
    versions = s3.list_object_versions(Bucket=bucket)
    for item in [*versions.get("Versions", []), *versions.get("DeleteMarkers", [])]:
        s3.delete_object(Bucket=bucket, Key=item["Key"], VersionId=item["VersionId"])
    s3.delete_bucket(Bucket=bucket)


def _teardown(ddb: Any, s3: Any, ssm: Any, sns: Any, topic_arn: str) -> None:
    for name in TABLE_NAMES.values():
        ddb.Table(name).delete()
    for bucket in (DOCUMENTS_BUCKET, REPORTS_BUCKET):
        _empty_bucket(s3, bucket)
    names = [
        p["Name"]
        for p in ssm.get_parameters_by_path(Path=SSM_PREFIX, Recursive=True).get("Parameters", [])
    ]
    for start in range(0, len(names), 10):
        ssm.delete_parameters(Names=names[start : start + 10])
    sns.delete_topic(TopicArn=topic_arn)


@pytest.fixture
def aws(_moto_session: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[AwsResources]:
    """Create every table, bucket, parameter and topic Keepsake uses; remove them afterwards."""
    key_id = _moto_session
    monkeypatch.setenv("KEEPSAKE_KMS_KEY_ID", key_id)
    s3 = boto3.client("s3", region_name=REGION)
    for bucket in (DOCUMENTS_BUCKET, REPORTS_BUCKET):
        s3.create_bucket(Bucket=bucket)
    s3.put_bucket_versioning(Bucket=DOCUMENTS_BUCKET, VersioningConfiguration={"Status": "Enabled"})
    ddb = boto3.resource("dynamodb", region_name=REGION)
    tables = _create_tables(ddb, key_id)
    ssm = boto3.client("ssm", region_name=REGION)
    _put_ssm_parameters(ssm)
    sns = boto3.client("sns", region_name=REGION)
    topic_arn = sns.create_topic(Name="keepsake-alerts")["TopicArn"]
    monkeypatch.setenv("KEEPSAKE_ALERTS_TOPIC_ARN", topic_arn)
    yield AwsResources(
        ddb=ddb, s3=s3, ssm=ssm, sns=sns, kms_key_id=key_id, tables=tables, topic_arn=topic_arn
    )
    _teardown(ddb, s3, ssm, sns, topic_arn)


def converse_response(
    text: str = "ok",
    *,
    input_tokens: int = 1200,
    output_tokens: int = 50,
    cache_read: int = 0,
    cache_write: int = 0,
) -> dict[str, Any]:
    """A Converse response with the usage fields Keepsake reads."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens + cache_read + cache_write,
            "cacheReadInputTokens": cache_read,
            "cacheWriteInputTokens": cache_write,
        },
        "metrics": {"latencyMs": 120},
    }


class FakeBedrock:
    """A real bedrock-runtime client behind a botocore Stubber that records each request."""

    def __init__(self) -> None:
        self.client = boto3.client("bedrock-runtime", region_name=REGION)
        self.stubber = Stubber(self.client)
        self.requests: list[dict[str, Any]] = []
        self.client.meta.events.register(
            "before-parameter-build.bedrock-runtime.Converse", self._record
        )
        self.stubber.activate()

    def _record(self, params: dict[str, Any], **_: Any) -> None:
        self.requests.append(json.loads(json.dumps(params)))

    def queue(self, response: dict[str, Any] | None = None) -> None:
        self.stubber.add_response("converse", response or converse_response())

    def queue_error(self, code: str, status: int = 400) -> None:
        self.stubber.add_client_error(
            "converse", service_error_code=code, service_message=code, http_status_code=status
        )

    @property
    def model_ids(self) -> list[str]:
        return [request["modelId"] for request in self.requests]


@pytest.fixture
def fake_bedrock() -> Iterator[FakeBedrock]:
    fake = FakeBedrock()
    yield fake
    fake.stubber.deactivate()


@pytest.fixture
def fixed_now() -> Any:
    """A controllable clock: call it for the current time; set `.value` to move time."""

    class Clock:
        def __init__(self) -> None:
            self.value = FIXED_NOW

        def __call__(self) -> datetime:
            return self.value

    return Clock()


SAMPLE_DOCUMENTS = {
    "policy-manual": (
        "Refund policy. Customers may request a refund within 30 days of purchase.\n\n"
        "Escalation policy. Tickets open longer than 48 hours escalate to tier two.\n\n"
        "Privacy policy. Never share customer data with another customer."
    ),
    "runbook": "Restart procedure.\n\nStep one: drain traffic.\n\nStep two: restart the service.",
    "contract": "Master services agreement between Groundhog Labs and Example Corp.",
    "source-snapshot": "def handler(event):\n    return event\n\nclass Service:\n    pass",
}
