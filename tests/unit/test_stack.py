"""CDK stack assertions and cdk-nag gate (NFR4, NFR6, NFR8, FR2.3, FR4.4).

Synthesis needs aws-cdk-lib (and the Node.js runtime jsii uses) but no AWS credentials and no
CDK CLI. The stack is synthesized once per test module because jsii synthesis is slow.
"""

from __future__ import annotations

from typing import Any

import pytest
from aws_cdk.assertions import Annotations, Match, Template

from app import KEEPSAKE_REGION, build_app
from infra.keepsake_stack import KeepsakeStack


@pytest.fixture(scope="module")
def synthesized() -> tuple[KeepsakeStack, Template]:
    app, stack = build_app()
    app.synth()
    return stack, Template.from_stack(stack)


def _resources(template: Template, resource_type: str) -> dict[str, Any]:
    return template.find_resources(resource_type)


def test_stack_is_pinned_to_us_east_1(synthesized: tuple[KeepsakeStack, Template]) -> None:
    stack, _ = synthesized

    assert stack.region == KEEPSAKE_REGION == "us-east-1"


def test_customer_managed_key_with_rotation(synthesized: tuple[KeepsakeStack, Template]) -> None:
    _, template = synthesized

    template.resource_count_is("AWS::KMS::Key", 1)
    template.has_resource_properties("AWS::KMS::Key", {"EnableKeyRotation": True})


def test_document_bucket_is_versioned_kms_encrypted_private_and_tls_only(
    synthesized: tuple[KeepsakeStack, Template],
) -> None:
    _, template = synthesized
    key_ref = Match.object_like({"Fn::GetAtt": [Match.string_like_regexp("DataKey"), "Arn"]})

    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "VersioningConfiguration": {"Status": "Enabled"},
            "BucketEncryption": {
                "ServerSideEncryptionConfiguration": [
                    Match.object_like(
                        {
                            "ServerSideEncryptionByDefault": {
                                "SSEAlgorithm": "aws:kms",
                                "KMSMasterKeyID": key_ref,
                            }
                        }
                    )
                ]
            },
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            },
        },
    )
    template.has_resource_properties(
        "AWS::S3::BucketPolicy",
        {
            "PolicyDocument": {
                "Statement": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Effect": "Deny",
                                "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                            }
                        )
                    ]
                )
            }
        },
    )


def test_all_four_tables_use_the_cmk_with_point_in_time_recovery(
    synthesized: tuple[KeepsakeStack, Template],
) -> None:
    _, template = synthesized

    tables = _resources(template, "AWS::DynamoDB::Table")

    assert len(tables) == 4
    for table in tables.values():
        properties = table["Properties"]
        assert properties["SSESpecification"]["SSEType"] == "KMS"
        assert "KMSMasterKeyId" in properties["SSESpecification"]
        assert properties["PointInTimeRecoverySpecification"]["PointInTimeRecoveryEnabled"] is True
    cache = [
        t
        for t in tables.values()
        if t["Properties"].get("TimeToLiveSpecification", {}).get("AttributeName") == "expires_at"
    ]
    assert len(cache) == 1


def test_lambdas_are_python_313_arm64_and_reached_through_live_aliases(
    synthesized: tuple[KeepsakeStack, Template],
) -> None:
    _, template = synthesized

    functions = [
        f
        for f in _resources(template, "AWS::Lambda::Function").values()
        if str(f["Properties"].get("Handler", "")).startswith("keepsake.handlers.")
    ]
    aliases = _resources(template, "AWS::Lambda::Alias")

    assert len(functions) == 6
    assert all(f["Properties"]["Runtime"] == "python3.13" for f in functions)
    assert all(f["Properties"]["Architectures"] == ["arm64"] for f in functions)
    assert len([a for a in aliases.values() if a["Properties"]["Name"] == "live"]) == 6


def test_invoke_uses_iam_and_staff_routes_use_the_cognito_authorizer(
    synthesized: tuple[KeepsakeStack, Template],
) -> None:
    _, template = synthesized

    methods = [
        m["Properties"]
        for m in _resources(template, "AWS::ApiGateway::Method").values()
        if m["Properties"]["HttpMethod"] != "OPTIONS"
    ]

    template.resource_count_is("AWS::ApiGateway::Authorizer", 1)
    template.has_resource_properties("AWS::ApiGateway::Authorizer", {"Type": "COGNITO_USER_POOLS"})
    assert sum(1 for m in methods if m["AuthorizationType"] == "AWS_IAM") == 1
    assert (
        sum(1 for m in methods if m["AuthorizationType"] == "COGNITO_USER_POOLS")
        == len(methods) - 1
    )


def test_bedrock_access_is_limited_to_configured_models(
    synthesized: tuple[KeepsakeStack, Template],
) -> None:
    _, template = synthesized

    statements = [
        s
        for p in _resources(template, "AWS::IAM::Policy").values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
        if "bedrock:InvokeModel" in str(s["Action"])
    ]

    assert len(statements) == 1
    assert "*" not in statements[0]["Resource"]
    assert "deepseek.v3.2" in str(statements[0]["Resource"])


def test_cdk_nag_reports_no_unsuppressed_errors(
    synthesized: tuple[KeepsakeStack, Template],
) -> None:
    stack, _ = synthesized

    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )

    assert [f"{e.id}: {e.entry.data}" for e in errors] == []
