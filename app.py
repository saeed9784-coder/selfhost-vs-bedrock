#!/usr/bin/env python3
"""CDK entry point; default deployment is evidence-first Phase 0 in us-east-1."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import aws_cdk as cdk
from cdk_nag import AwsSolutionsChecks

from infra.keepsake_stack import KeepsakeStack
from infra.phase0_stack import (
    PHASE0_COST_CONTEXT,
    SOURCE_BUCKET_CONTEXT,
    SOURCE_KMS_CONTEXT,
    SOURCE_PREFIX_CONTEXT,
    Phase0Stack,
)

KEEPSAKE_REGION = "us-east-1"
CDK_JSON = Path(__file__).resolve().parent / "cdk.json"
SRC_DIR = str(Path(__file__).resolve().parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)  # the self-hosted stack reads its price catalog from src/selfhost


def cdk_json_context() -> dict[str, Any]:
    """Context and feature flags from cdk.json, so tests synthesize exactly like `cdk synth`."""
    return dict(json.loads(CDK_JSON.read_text(encoding="utf-8")).get("context", {}))


def build_app(app: cdk.App | None = None) -> tuple[cdk.App, KeepsakeStack]:
    """Create the later-stage customer-facing Keepsake prototype stack."""
    app = app or cdk.App(context=cdk_json_context())
    region = app.node.try_get_context("keepsake:region") or KEEPSAKE_REGION
    if region != KEEPSAKE_REGION:
        raise ValueError(f"Keepsake must deploy to {KEEPSAKE_REGION} only, got {region!r}")
    env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=KEEPSAKE_REGION)
    stack = KeepsakeStack(app, "Keepsake", env=env)
    cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    return app, stack


def build_phase0_app(app: cdk.App | None = None) -> tuple[cdk.App, Phase0Stack]:
    """Create only the article's evidence-first Phase 0 analyzer stack."""
    app = app or cdk.App(context=cdk_json_context())
    region = app.node.try_get_context("keepsake:region") or KEEPSAKE_REGION
    if region != KEEPSAKE_REGION:
        raise ValueError(f"Keepsake Phase 0 is pinned to {KEEPSAKE_REGION}, got {region!r}")
    env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=KEEPSAKE_REGION)
    schedule_value = app.node.try_get_context("keepsake:phase0WeeklySchedule")
    stack = Phase0Stack(
        app,
        "KeepsakePhase0",
        env=env,
        source_bucket_name=app.node.try_get_context(SOURCE_BUCKET_CONTEXT),
        source_prefix=app.node.try_get_context(SOURCE_PREFIX_CONTEXT) or "AWSLogs/",
        source_kms_key_arn=app.node.try_get_context(SOURCE_KMS_CONTEXT) or "",
        build_run_cost_usd=str(app.node.try_get_context(PHASE0_COST_CONTEXT) or "500"),
        schedule_enabled=schedule_value is True
        or str(schedule_value).lower() in {"true", "1", "yes"},
    )
    cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    return app, stack


def build_pipeline_app(app: cdk.App | None = None) -> tuple[cdk.App, Any]:
    """CDK Pipeline for Stage 0 (`cdk deploy -c keepsake:mode=pipeline`)."""
    from infra.pipeline_stack import KeepsakePipelineStack, PipelineSettings

    app = app or cdk.App(context=cdk_json_context())
    env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=KEEPSAKE_REGION)
    stack = KeepsakePipelineStack(
        app, "KeepsakePipeline", env=env, settings=PipelineSettings.from_context(app.node)
    )
    cdk.Aspects.of(stack).add(AwsSolutionsChecks(verbose=True))
    return app, stack


def build_selfhost_app(app: cdk.App | None = None) -> tuple[cdk.App, Any]:
    """Minimal self-hosted inference slice (`cdk deploy -c keepsake:mode=selfhost`)."""
    from infra.selfhost_stack import SelfHostConfig, SelfHostedInferenceStack

    app = app or cdk.App(context=cdk_json_context())
    env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=KEEPSAKE_REGION)
    stack = SelfHostedInferenceStack(
        app, "SelfHostedInference", env=env, config=SelfHostConfig.from_context(app)
    )
    cdk.Aspects.of(stack).add(AwsSolutionsChecks(verbose=True))
    return app, stack


def build_selfhost_full_app(app: cdk.App | None = None) -> tuple[cdk.App, Any]:
    """Full-scale reference architecture; synthesize only (`-c keepsake:mode=selfhost-full`)."""
    from infra.selfhost_full_stack import SelfHostedInferenceFullStack

    app = app or cdk.App(context=cdk_json_context())
    env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=KEEPSAKE_REGION)
    stack = SelfHostedInferenceFullStack(app, "SelfHostedInferenceFull", env=env)
    cdk.Aspects.of(stack).add(AwsSolutionsChecks(verbose=True))
    return app, stack


if __name__ == "__main__":
    probe = cdk.App(context=cdk_json_context())
    mode = probe.node.try_get_context("keepsake:mode") or "phase0"
    if mode == "pipeline":
        keepsake_app, _ = build_pipeline_app(probe)
    elif mode == "phase0":
        keepsake_app, _ = build_phase0_app(probe)
    elif mode == "selfhost":
        keepsake_app, _ = build_selfhost_app(probe)
    elif mode == "selfhost-full":
        keepsake_app, _ = build_selfhost_full_app(probe)
    elif mode in {"customer-poc", "stack"}:
        keepsake_app, _ = build_app(probe)
    else:
        raise ValueError(
            f"Unknown keepsake:mode {mode!r}; choose 'phase0', 'pipeline', "
            "'customer-poc', 'selfhost' or 'selfhost-full'"
        )
    keepsake_app.synth()
