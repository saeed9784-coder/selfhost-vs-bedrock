"""CDK assertions for the deployable self-hosted inference stack, plus the cdk-nag gate."""

from __future__ import annotations

import json

import pytest
from aws_cdk import App
from aws_cdk.assertions import Annotations, Match, Template

from app import build_selfhost_app, build_selfhost_full_app, cdk_json_context
from infra.selfhost_stack import CLOUDFRONT_PREFIX_LIST_US_EAST_1, SelfHostedInferenceStack


@pytest.fixture(scope="module")
def synthesized() -> tuple[SelfHostedInferenceStack, Template]:
    app, stack = build_selfhost_app()
    app.synth()
    return stack, Template.from_stack(stack)


def test_no_unsuppressed_nag_errors(synthesized: tuple[SelfHostedInferenceStack, Template]) -> None:
    stack, _ = synthesized
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )
    assert errors == []


def test_cpu_poc_node_is_private_imdsv2_and_encrypted(
    synthesized: tuple[SelfHostedInferenceStack, Template],
) -> None:
    _, template = synthesized
    template.has_resource_properties(
        "AWS::EC2::LaunchTemplate",
        {
            "LaunchTemplateData": Match.object_like(
                {
                    "InstanceType": "r7a.4xlarge",
                    "Monitoring": {"Enabled": True},
                    "MetadataOptions": Match.object_like({"HttpTokens": "required"}),
                    "BlockDeviceMappings": [
                        Match.object_like(
                            {
                                "Ebs": Match.object_like(
                                    {
                                        "Encrypted": True,
                                        "VolumeType": "gp3",
                                        "VolumeSize": 200,
                                        "Throughput": 500,
                                    }
                                )
                            }
                        )
                    ],
                }
            )
        },
    )
    group = _node_group(template)
    assert (group["MinSize"], group["MaxSize"]) == ("1", "1")
    assert len(group["VPCZoneIdentifier"]) == 2  # both private AZs
    overrides = group["MixedInstancesPolicy"]["LaunchTemplate"]["Overrides"]
    assert [o["InstanceType"] for o in overrides] == [
        "r7a.4xlarge",
        "r7i.4xlarge",
        "r6a.4xlarge",
        "r6i.4xlarge",
    ]  # 16 vCPU, 128 GiB: room for the 101 GB MiniMax M2.5 build
    distribution = group["MixedInstancesPolicy"]["InstancesDistribution"]
    assert distribution["OnDemandPercentageAboveBaseCapacity"] == 0  # Spot only
    assert distribution["SpotAllocationStrategy"] == "price-capacity-optimized"
    assert group["CapacityRebalance"] is True
    assert "AssociatePublicIpAddress" not in json.dumps(
        template.find_resources("AWS::EC2::LaunchTemplate")
    )


def test_alb_only_accepts_cloudfront(
    synthesized: tuple[SelfHostedInferenceStack, Template],
) -> None:
    _, template = synthesized
    ingress = [
        rule
        for group in template.find_resources("AWS::EC2::SecurityGroup").values()
        for rule in group["Properties"].get("SecurityGroupIngress", [])
    ] + [
        r["Properties"] for r in template.find_resources("AWS::EC2::SecurityGroupIngress").values()
    ]
    assert ingress
    assert not [rule for rule in ingress if rule.get("CidrIp") == "0.0.0.0/0"]
    template.has_resource_properties(
        "AWS::EC2::SecurityGroupIngress",
        {"FromPort": 80, "ToPort": 80, "SourcePrefixListId": CLOUDFRONT_PREFIX_LIST_US_EAST_1},
    )
    template.has_resource_properties(
        "AWS::CloudFront::Distribution",
        {
            "DistributionConfig": Match.object_like(
                {
                    "DefaultCacheBehavior": Match.object_like(
                        {"ViewerProtocolPolicy": "redirect-to-https"}
                    ),
                    "Origins": [
                        Match.object_like(
                            {
                                "OriginCustomHeaders": [
                                    Match.object_like({"HeaderName": "X-Origin-Verify"})
                                ]
                            }
                        )
                    ],
                }
            )
        },
    )


def test_bedrock_permission_is_scoped_to_the_comparison_model(
    synthesized: tuple[SelfHostedInferenceStack, Template],
) -> None:
    _, template = synthesized
    rendered = json.dumps(template.find_resources("AWS::IAM::Policy"))
    assert "foundation-model/minimax.minimax-m2.5" in rendered  # same model as self-hosted
    assert "llama3-1-8b" not in rendered
    assert "SELFHOST_MODEL=minimax-m2.5" in _user_data(synthesized[1])
    assert "SELFHOST_BEDROCK_MODEL= " in _user_data(synthesized[1])  # no override
    assert '"bedrock:*"' not in rendered


def _node_group(template: Template) -> dict:
    group = next(iter(template.find_resources("AWS::AutoScaling::AutoScalingGroup").values()))
    return group["Properties"]


def _user_data(template: Template) -> str:
    lt = next(iter(template.find_resources("AWS::EC2::LaunchTemplate").values()))
    return json.dumps(lt["Properties"]["LaunchTemplateData"]["UserData"])


def test_cpu_user_data_runs_llamacpp_with_metrics_and_installs_cli_first(
    synthesized: tuple[SelfHostedInferenceStack, Template],
) -> None:
    _, template = synthesized
    script = _user_data(template)
    for expected in (
        "llama.cpp:server",
        "unsloth/MiniMax-M2.5-GGUF:UD-Q3_K_XL",
        "--jinja",
        "--metrics",
        "MEM_GIB",
        "SELFHOST_PURCHASE=spot",
        "selfhost-gateway",
    ):
        assert expected in script
    assert script.index("awscli-exe-linux") < script.index("aws s3 cp")
    assert "--gpus" not in script
    assert "apt-mark hold docker.io" in script
    # model copy in S3: links are not followed on upload and are recreated on restore
    assert "--no-follow-symlinks" in script
    assert "symlinks.tsv" in script and "ln -sfn" in script


def test_gpu_variant_runs_vllm_with_prefix_caching_on_the_dlami() -> None:
    app = App(
        context={
            **cdk_json_context(),
            "selfhost:model": "llama-3.1-8b",
            "selfhost:instanceType": "g6e.xlarge",
        }
    )
    app, stack = build_selfhost_app(app)
    app.synth()
    template = Template.from_stack(stack)
    script = _user_data(template)
    for expected in ("--enable-prefix-caching", "--enable-prompt-tokens-details", "--gpus all"):
        assert expected in script
    template.has_resource_properties(
        "AWS::EC2::LaunchTemplate",
        {"LaunchTemplateData": Match.object_like({"InstanceType": "g6e.xlarge"})},
    )


def test_gpu_only_model_on_cpu_is_rejected() -> None:
    app = App(context={**cdk_json_context(), "selfhost:model": "qwen3-32b"})
    with pytest.raises(ValueError, match="cannot run on cpu"):
        build_selfhost_app(app)


def test_unknown_model_is_rejected() -> None:
    app = App(context={**cdk_json_context(), "selfhost:model": "gpt-9"})
    with pytest.raises(ValueError, match="selfhost:model"):
        build_selfhost_app(app)


def test_full_reference_stack_synthesizes_both_gpu_pools() -> None:
    app, stack = build_selfhost_full_app()
    app.synth()
    template = Template.from_stack(stack)
    pools = {
        g["Properties"]["NodegroupName"]: g["Properties"]["InstanceTypes"]
        for g in template.find_resources("AWS::EKS::Nodegroup").values()
    }
    assert pools["prefill"] == ["p5.48xlarge"]
    assert pools["decode"] == ["p5.48xlarge"]
    template.resource_count_is("AWS::FSx::FileSystem", 1)


def test_availability_zone_can_be_chosen_when_one_has_no_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", "123456789012")  # concrete AZ names
    with pytest.raises(ValueError, match="not one of the VPC's AZs"):
        build_selfhost_app(App(context={**cdk_json_context(), "selfhost:availabilityZone": "x"}))
    app = App(context={**cdk_json_context(), "selfhost:availabilityZone": "dummy1b"})
    app, stack = build_selfhost_app(app)
    template = Template.from_stack(stack)
    subnets = _node_group(template)["VPCZoneIdentifier"]
    assert len(subnets) == 1
    subnet = template.find_resources("AWS::EC2::Subnet")[subnets[0]["Ref"]]
    assert subnet["Properties"]["AvailabilityZone"] == "dummy1b"


def test_same_model_comparison_scopes_bedrock_to_llama() -> None:
    app = App(
        context={
            **cdk_json_context(),
            "selfhost:model": "llama-3.1-8b",
            "selfhost:instanceType": "c7a.xlarge",
        }
    )
    app, stack = build_selfhost_app(app)
    rendered = json.dumps(Template.from_stack(stack).find_resources("AWS::IAM::Policy"))
    assert "foundation-model/meta.llama3-1-8b-instruct-v1:0" in rendered
    assert "inference-profile/us.meta.llama3-1-8b-instruct-v1:0" in rendered
    assert "minimax" not in rendered


def test_on_demand_purchase_option_and_bad_value() -> None:
    app = App(context={**cdk_json_context(), "selfhost:purchase": "on_demand"})
    app, stack = build_selfhost_app(app)
    group = _node_group(Template.from_stack(stack))
    assert (
        group["MixedInstancesPolicy"]["InstancesDistribution"][
            "OnDemandPercentageAboveBaseCapacity"
        ]
        == 100
    )
    with pytest.raises(ValueError, match="selfhost:purchase"):
        build_selfhost_app(App(context={**cdk_json_context(), "selfhost:purchase": "free"}))


def test_model_that_does_not_fit_in_memory_is_rejected() -> None:
    app = App(context={**cdk_json_context(), "selfhost:instanceType": "c7a.xlarge"})
    with pytest.raises(ValueError, match="needs about 112 GB"):
        build_selfhost_app(app)
