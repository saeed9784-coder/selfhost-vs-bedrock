"""Identity: Cognito for staff, one IAM role per agent (FR2.4, FR5.4, NFR3).

Staff sign in through the Cognito Hosted UI (authorization code + PKCE, MFA required). Groups:
`finance`, `doc-admin` and one `team-<name>` per team. Agents call the API with SigV4 using
their own IAM role, which may only invoke `POST /invoke`; the SSM agent registry maps each role
name to its agent id.
"""

from __future__ import annotations

from aws_cdk import Aws, Duration, RemovalPolicy
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_iam as iam
from constructs import Construct

STAFF_GROUPS = ("finance", "doc-admin", "team-reviewer", "team-helpdesk", "team-atlas")
AGENTS = ("reviewer", "helpdesk", "atlas")


def agent_role_name(agent_id: str) -> str:
    return f"keepsake-agent-{agent_id}"


class AuthConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str) -> None:
        super().__init__(scope, construct_id)
        self.user_pool = cognito.UserPool(
            self,
            "Staff",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            mfa=cognito.Mfa.REQUIRED,
            mfa_second_factor=cognito.MfaSecondFactor(otp=True, sms=False),
            password_policy=cognito.PasswordPolicy(
                min_length=12,
                require_digits=True,
                require_lowercase=True,
                require_uppercase=True,
                require_symbols=True,
                temp_password_validity=Duration.days(3),
            ),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=RemovalPolicy.RETAIN,
        )
        for group in STAFF_GROUPS:
            cognito.CfnUserPoolGroup(
                self, f"Group-{group}", user_pool_id=self.user_pool.user_pool_id, group_name=group
            )
        self.domain = self.user_pool.add_domain(
            "Domain",
            cognito_domain=cognito.CognitoDomainOptions(domain_prefix=f"keepsake-{Aws.ACCOUNT_ID}"),
        )
        self.agent_roles = {agent: self._agent_role(agent) for agent in AGENTS}
        self.client: cognito.UserPoolClient | None = None

    def _agent_role(self, agent_id: str) -> iam.Role:
        return iam.Role(
            self,
            f"AgentRole-{agent_id}",
            role_name=agent_role_name(agent_id),
            assumed_by=iam.AccountRootPrincipal(),
            description=f"Keepsake agent identity for {agent_id}; may only call POST /invoke",
            max_session_duration=Duration.hours(12),
        )

    def add_web_client(self, site_url: str) -> cognito.UserPoolClient:
        """The public PKCE client for the staff web pages (no client secret)."""
        self.client = self.user_pool.add_client(
            "WebClient",
            generate_secret=False,
            auth_flows=cognito.AuthFlow(user_srp=True),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[
                    cognito.OAuthScope.OPENID,
                    cognito.OAuthScope.EMAIL,
                    cognito.OAuthScope.PROFILE,
                ],
                callback_urls=[f"{site_url}/index.html"],
                logout_urls=[f"{site_url}/index.html"],
            ),
            prevent_user_existence_errors=True,
            id_token_validity=Duration.hours(1),
            access_token_validity=Duration.hours(1),
            refresh_token_validity=Duration.days(1),
        )
        return self.client

    def grant_invoke(self, invoke_method_arn: str) -> None:
        for role in self.agent_roles.values():
            role.add_to_policy(
                iam.PolicyStatement(actions=["execute-api:Invoke"], resources=[invoke_method_arn])
            )
