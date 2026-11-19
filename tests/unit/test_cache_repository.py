"""Cache repository tests against moto (FR4.4, FR4.6, FR4.7, NFR3)."""

from __future__ import annotations

import hashlib

from keepsake.cache.repository import CacheRepository
from keepsake.common.models import CacheEntry, TokenUsage
from tests.conftest import AwsResources

NOW = 1_800_000_000


def _entry(
    customer: str = "cust-1",
    agent: str = "helpdesk",
    team: str = "helpdesk",
    docs: tuple[str, ...] = ("policy-manual@000001",),
    seed: str = "a",
) -> CacheEntry:
    return CacheEntry(
        team=team,
        customer_id=customer,
        agent_id=agent,
        cache_key=hashlib.sha256(f"{customer}{agent}{seed}".encode()).hexdigest(),
        model_id="deepseek.v3.2",
        model_version="deepseek-v3.2",
        document_versions=docs,
        response_text="Refunds take 5 days.",
        usage=TokenUsage(input_tokens=1000, output_tokens=20),
        created_at=NOW,
        expires_at=NOW + 3600,
    )


def _get(repo: CacheRepository, entry: CacheEntry, **overrides: str) -> CacheEntry | None:
    keys = {
        "team": entry.team,
        "customer_id": entry.customer_id,
        "agent_id": entry.agent_id,
        "cache_key": entry.cache_key,
    }
    keys.update(overrides)
    return repo.get(now_epoch=NOW + 10, **keys)


def test_put_then_get_round_trips_the_entry(aws: AwsResources) -> None:
    repo = CacheRepository(table=aws.tables["cache"])
    entry = _entry()

    repo.put(entry)

    assert _get(repo, entry) == entry


def test_another_customer_or_team_partition_never_reaches_the_entry(aws: AwsResources) -> None:
    repo = CacheRepository(table=aws.tables["cache"])
    entry = _entry()
    repo.put(entry)

    assert _get(repo, entry, customer_id="cust-2") is None
    assert _get(repo, entry, team="reviewer") is None


def test_expired_entry_is_a_miss(aws: AwsResources) -> None:
    repo = CacheRepository(table=aws.tables["cache"])
    entry = _entry()
    repo.put(entry)

    assert (
        repo.get(
            team=entry.team,
            customer_id=entry.customer_id,
            agent_id=entry.agent_id,
            cache_key=entry.cache_key,
            now_epoch=entry.expires_at,
        )
        is None
    )


def test_delete_by_customer_agent_and_all_return_entry_counts(aws: AwsResources) -> None:
    repo = CacheRepository(table=aws.tables["cache"])
    repo.put(_entry("cust-1", seed="a"))
    repo.put(_entry("cust-1", seed="b"))
    repo.put(_entry("cust-2", agent="reviewer", team="reviewer"))
    repo.put(_entry("cust-3", agent="atlas", team="atlas"))

    assert repo.delete_by_customer("cust-1") == 2
    assert repo.delete_by_agent("reviewer") == 1
    assert repo.delete_all() == 1
    assert repo.delete_all() == 0


def test_delete_by_document_version_removes_dependent_entries_only(aws: AwsResources) -> None:
    repo = CacheRepository(table=aws.tables["cache"])
    dependent = _entry(docs=("policy-manual@000001", "runbook@000003"), seed="a")
    unrelated = _entry(docs=("runbook@000003",), seed="b")
    repo.put(dependent)
    repo.put(unrelated)

    removed = repo.delete_by_document_version("policy-manual@000001")

    assert removed == 1
    assert _get(repo, dependent) is None
    assert _get(repo, unrelated) == unrelated
