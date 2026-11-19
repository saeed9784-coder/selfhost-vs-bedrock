"""Who may see which usage figures (FR5.4, NFR3).

- `finance` sees company-wide totals and any team.
- A member of `team-<name>` (a team lead) sees only `<name>`; asking for another team or for the
  company scope is denied, so another team's figures are never returned by the API.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from keepsake.common.errors import AccessDenied, ValidationError
from keepsake.store.access import FINANCE_GROUP, teams_of


@dataclass(frozen=True)
class UsageScope:
    """`team` is None for the company-wide scope."""

    team: str | None


def resolve_usage_scope(groups: Iterable[str], scope: str, team: str | None) -> UsageScope:
    group_set = frozenset(groups)
    is_finance = FINANCE_GROUP in group_set
    member_of = teams_of(group_set)
    if scope == "company":
        if not is_finance:
            raise AccessDenied("only finance may view company-wide usage")
        return UsageScope(team=None)
    if scope != "team":
        raise ValidationError("scope must be 'team' or 'company'")
    if team is None:
        if len(member_of) != 1:
            raise ValidationError("team is required")
        team = next(iter(member_of))
    if not is_finance and team not in member_of:
        raise AccessDenied("you may only view your own team's usage")
    return UsageScope(team=team)
