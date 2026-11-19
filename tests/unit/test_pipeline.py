"""CD pipeline tests: CDK Pipelines from GitHub via CodeConnections, one us-east-1 stage."""

from __future__ import annotations

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Annotations, Match, Template

from app import build_pipeline_app, cdk_json_context

CONNECTION_ARN = "arn:aws:codeconnections:us-east-1:111122223333:connection/abc-123"


def _context(**overrides: str) -> dict[str, str]:
    return {
        **cdk_json_context(),
        "keepsake:githubRepo": "groundhog/keepsake",
        "keepsake:connectionArn": CONNECTION_ARN,
        **overrides,
    }


@pytest.fixture(scope="module")
def pipeline() -> tuple[cdk.Stack, Template]:
    app = cdk.App(context=_context())
    app, stack = build_pipeline_app(app)
    app.synth()
    return stack, Template.from_stack(stack)


def test_pipeline_sources_github_through_codeconnections(
    pipeline: tuple[cdk.Stack, Template],
) -> None:
    _, template = pipeline

    template.has_resource_properties(
        "AWS::CodePipeline::Pipeline",
        {
            "Stages": Match.array_with(
                [
                    Match.object_like(
                        {
                            "Name": "Source",
                            "Actions": [
                                Match.object_like(
                                    {
                                        "ActionTypeId": Match.object_like(
                                            {"Provider": "CodeStarSourceConnection"}
                                        ),
                                        "Configuration": Match.object_like(
                                            {
                                                "ConnectionArn": CONNECTION_ARN,
                                                "FullRepositoryId": "groundhog/keepsake",
                                                "BranchName": "main",
                                            }
                                        ),
                                    }
                                )
                            ],
                        }
                    )
                ]
            ),
        },
    )


def test_pipeline_runs_tests_self_mutates_deploys_one_stage_and_smoke_tests(
    pipeline: tuple[cdk.Stack, Template],
) -> None:
    _, template = pipeline
    stages = next(iter(template.find_resources("AWS::CodePipeline::Pipeline").values()))[
        "Properties"
    ]["Stages"]
    names = [s["Name"] for s in stages]

    assert names == ["Source", "Build", "UpdatePipeline", "Assets", "Prod"]
    prod_actions = [a["Name"] for a in stages[-1]["Actions"]]
    assert any("SmokeTest" in name for name in prod_actions)
    synth_specs = [
        p["Properties"]["Source"]["BuildSpec"]
        for p in template.find_resources("AWS::CodeBuild::Project").values()
    ]
    assert any("pytest tests/unit" in spec for spec in synth_specs)


def test_pipeline_and_deployed_stage_stay_in_us_east_1(
    pipeline: tuple[cdk.Stack, Template],
) -> None:
    stack, _ = pipeline

    assert stack.region == "us-east-1"
    assert stack.stage.stack.region == "us-east-1"  # type: ignore[attr-defined]


def test_pipeline_passes_cdk_nag(pipeline: tuple[cdk.Stack, Template]) -> None:
    stack, _ = pipeline

    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )

    assert errors == []


def test_missing_repository_or_connection_is_rejected() -> None:
    app = cdk.App(context=_context(**{"keepsake:connectionArn": ""}))

    with pytest.raises(ValueError, match="keepsake:connectionArn"):
        build_pipeline_app(app)
