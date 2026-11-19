"""Access policy for documents (FR2.4, NFR3).

Agents (machine callers): an agent may read a document only when
1. the document belongs to the agent's team or to the shared `company` team, and
2. the document's classification is in the agent's `allowed_classifications` (SSM config).
The shipped configuration denies `contract` to Reviewer and `source-code` to Helpdesk.

Staff (Cognito groups): `doc-admin` manages every document; members of `team-<name>` manage
documents owned by `<name>`. Only `doc-admin` approves quarantined documents.
"""

from __future__ import annotations

from collections.abc import Iterable

from keepsake.common.errors import AccessDenied
from keepsake.common.models import AgentConfig, Document, DocumentVersion

SHARED_TEAM = "company"
DOC_ADMIN_GROUP = "doc-admin"
FINANCE_GROUP = "finance"
TEAM_GROUP_PREFIX = "team-"


def teams_of(groups: Iterable[str]) -> frozenset[str]:
    return frozenset(g[len(TEAM_GROUP_PREFIX) :] for g in groups if g.startswith(TEAM_GROUP_PREFIX))


def check_agent_can_read(agent: AgentConfig, version: DocumentVersion) -> None:
    """Raise AccessDenied unless the agent may read this document version."""
    if version.owner_team not in {agent.team, SHARED_TEAM}:
        raise AccessDenied(f"agent {agent.agent_id} may not read documents of another team")
    if version.classification not in agent.allowed_classifications:
        raise AccessDenied(
            f"agent {agent.agent_id} may not read {version.classification} documents"
        )


def check_staff_can_manage(groups: Iterable[str], owner_team: str) -> None:
    group_set = frozenset(groups)
    if DOC_ADMIN_GROUP in group_set:
        return
    if owner_team in teams_of(group_set):
        return
    raise AccessDenied("you may only manage documents owned by your team")


def check_staff_can_approve(groups: Iterable[str]) -> None:
    if DOC_ADMIN_GROUP not in frozenset(groups):
        raise AccessDenied("only doc-admin may approve quarantined documents")


def check_staff_can_read(groups: Iterable[str], document: Document) -> None:
    """Staff read: doc-admin, the owner team, or anyone for shared `company` documents."""
    if document.owner_team == SHARED_TEAM:
        return
    check_staff_can_manage(groups, document.owner_team)
