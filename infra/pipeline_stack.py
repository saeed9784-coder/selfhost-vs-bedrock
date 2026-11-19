"""Self-mutating pipeline for the Phase 0 evidence-gathering stack.

The pipeline deploys only the invocation-log analyzer. The customer-facing prototype and any
Bedrock serving-side cache integration remain out of this pipeline until their decision gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aws_cdk as cdk
from aws_cdk import Stack, Stage, pipelines
from aws_cdk import aws_codebuild as codebuild
from cdk_nag import AwsSolutionsChecks, NagSuppressions
from constructs import Construct

from infra.phase0_stack import SOURCE_BUCKET_CONTEXT, SOURCE_KMS_CONTEXT, Phase0Stack

SYNTH_COMMANDS = (
    "python -m pip install -r requirements.txt -r requirements-dev.txt",
    "npm install -g aws-cdk@2",
    "python -m pytest tests/unit -q -p no:cacheprovider",
    "cdk synth -c keepsake:mode=phase0",
)


@dataclass(frozen=True)
class PipelineSettings:
    repository: str  # "owner/repo"
    branch: str
    connection_arn: str
    source_bucket: str = ""
    source_prefix: str = "AWSLogs/"
    source_kms_key_arn: str = ""

    @staticmethod
    def from_context(node: Any) -> PipelineSettings:
        repository = node.try_get_context("keepsake:githubRepo") or ""
        branch = node.try_get_context("keepsake:githubBranch") or "main"
        connection_arn = node.try_get_context("keepsake:connectionArn") or ""
        source_bucket = node.try_get_context(SOURCE_BUCKET_CONTEXT) or ""
        if "/" not in repository or not connection_arn.startswith("arn:aws"):
            raise ValueError(
                "Set cdk.json context keepsake:githubRepo (owner/repo) and "
                "keepsake:connectionArn (an AWS CodeConnections ARN) before deploying the pipeline"
            )
        if not source_bucket:
            raise ValueError(
                f"Set cdk.json context {SOURCE_BUCKET_CONTEXT} to the existing Bedrock "
                "invocation-log bucket before deploying the Phase 0 pipeline"
            )
        return PipelineSettings(
            repository=repository,
            branch=branch,
            connection_arn=connection_arn,
            source_bucket=source_bucket,
            source_prefix=node.try_get_context("keepsake:sourceInvocationLogsPrefix") or "AWSLogs/",
            source_kms_key_arn=node.try_get_context(SOURCE_KMS_CONTEXT) or "",
        )


class Phase0Stage(Stage):
    """Phase 0 only; later stages require evidence and a separate approval."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        settings: PipelineSettings,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.stack = Phase0Stack(
            self,
            "KeepsakePhase0",
            source_bucket_name=settings.source_bucket,
            source_prefix=settings.source_prefix,
            source_kms_key_arn=settings.source_kms_key_arn,
        )
        cdk.Aspects.of(self).add(AwsSolutionsChecks(verbose=True))


class KeepsakePipelineStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *, settings: PipelineSettings, **kwargs: Any
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        source = pipelines.CodePipelineSource.connection(
            settings.repository,
            settings.branch,
            connection_arn=settings.connection_arn,
        )
        build_env = codebuild.BuildEnvironment(
            build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
            privileged=False,
        )
        self.pipeline = pipelines.CodePipeline(
            self,
            "Pipeline",
            pipeline_name="keepsake-phase0",
            self_mutation=True,
            cross_account_keys=False,
            code_build_defaults=pipelines.CodeBuildOptions(build_environment=build_env),
            synth=pipelines.ShellStep("Synth", input=source, commands=list(SYNTH_COMMANDS)),
        )
        self.stage = Phase0Stage(
            self,
            "Phase0",
            settings=settings,
            env=cdk.Environment(account=self.account, region=self.region),
        )
        self.pipeline.add_stage(self.stage)
        self.pipeline.build_pipeline()
        _suppress_pipeline_findings(self)


def _suppress_pipeline_findings(stack: Stack) -> None:
    """Findings inherent to CDK Pipelines' generated resources, each with its reason."""
    NagSuppressions.add_stack_suppressions(
        stack,
        [
            {
                "id": "AwsSolutions-IAM5",
                "reason": "CDK Pipelines generates role policies with wildcards scoped to its own "
                "artifact bucket, log groups and CDK bootstrap roles.",
            },
            {
                "id": "AwsSolutions-S1",
                "reason": "Pipeline artifact bucket holds transient build artifacts only.",
            },
            {
                "id": "AwsSolutions-CB4",
                "reason": "CodeBuild artifacts are encrypted with the pipeline's KMS-backed "
                "bucket; projects are generated by CDK Pipelines.",
            },
            {
                "id": "AwsSolutions-KMS5",
                "reason": "Key generated by CodePipeline for its artifact bucket; rotation is not "
                "configurable through CDK Pipelines.",
            },
        ],
    )
