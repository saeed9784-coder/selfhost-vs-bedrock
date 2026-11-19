"""Documented cdk-nag suppressions (NFR8). Each one states why the finding is accepted for this POC.

Suppressions are applied per finding id at stack level only where the finding is about CDK-managed
helper resources or a scoped, explained exception; nothing here hides a data-protection gap.
"""

from __future__ import annotations

from aws_cdk import Stack
from cdk_nag import NagPackSuppression, NagSuppressions

STACK_SUPPRESSIONS = (
    NagPackSuppression(
        id="AwsSolutions-IAM4",
        reason="AWSLambdaBasicExecutionRole (Lambda logging) and the API Gateway CloudWatch role "
        "use AWS managed policies that only grant log delivery.",
        applies_to=[
            "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/"
            "AmazonAPIGatewayPushToCloudWatchLogs",
        ],
    ),
    NagPackSuppression(
        id="AwsSolutions-IAM5",
        reason="Wildcards are scoped: S3 object keys inside Keepsake buckets, KMS data-key "
        "actions on the Keepsake CMK, X-Ray, SSM parameters under /keepsake/*, Bedrock "
        "foundation-model ARNs in any US Region for geo inference profiles, and CDK "
        "BucketDeployment / auto-delete helpers.",
    ),
    NagPackSuppression(
        id="AwsSolutions-L1",
        reason="Python 3.13 is the runtime decided for this project (requirements Q5); "
        "CDK-managed helper functions use the runtime CDK pins.",
    ),
    NagPackSuppression(
        id="AwsSolutions-APIG3",
        reason="WAF tuning is out of scope for the POC (requirements: production hardening). "
        "Stage throttling and IAM / Cognito authorization are enforced.",
    ),
    NagPackSuppression(
        id="AwsSolutions-APIG4",
        reason="Only the CORS preflight OPTIONS methods are unauthenticated; every other method "
        "uses IAM or the Cognito authorizer.",
    ),
    NagPackSuppression(
        id="AwsSolutions-COG4",
        reason="Only CORS preflight OPTIONS methods and the IAM-authorized /invoke agent route "
        "do not use the Cognito authorizer.",
    ),
    NagPackSuppression(
        id="AwsSolutions-COG3",
        reason="Cognito threat protection (Plus plan) is a cost decision deferred to production; "
        "MFA is required and self sign-up is disabled.",
    ),
    NagPackSuppression(
        id="AwsSolutions-COG8",
        reason="The Cognito Plus feature plan (threat protection) is a cost decision deferred to "
        "production; MFA is required and self sign-up is disabled.",
    ),
    NagPackSuppression(
        id="AwsSolutions-CFR1",
        reason="No geo restriction: staff work from several countries; access needs Cognito login.",
    ),
    NagPackSuppression(
        id="AwsSolutions-CFR2",
        reason="WAF on CloudFront is out of scope for the POC; the site is static and all data "
        "access goes through the authenticated API.",
    ),
    NagPackSuppression(
        id="AwsSolutions-CFR4",
        reason="The POC uses the default *.cloudfront.net certificate, whose TLS policy cannot be "
        "raised; a custom domain with TLSv1.2_2021 is the production follow-up.",
    ),
    NagPackSuppression(
        id="AwsSolutions-S1",
        reason="The access-log bucket is the server-access-log destination itself.",
    ),
)


def apply_suppressions(stack: Stack) -> None:
    NagSuppressions.add_stack_suppressions(
        stack, list(STACK_SUPPRESSIONS), apply_to_nested_stacks=True
    )
