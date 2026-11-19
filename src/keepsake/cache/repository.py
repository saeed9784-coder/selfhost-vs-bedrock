"""Response-reuse cache data access (FR4.2, FR4.4, FR4.6, FR4.7).

Item layout in one DynamoDB table (encrypted with the customer-managed KMS key, us-east-1):
- entry: `pk=team#<team>#customer#<customer>`, `sk=entry#<agent>#<cache_key>`. The partition is
  per team and customer, so a lookup for another team or customer can never reach an entry.
- reference: `pk=<same>`, `sk=ref#<doc@ver>#<agent>#<cache_key>`, `doc_ref=doc#<doc@ver>`; one per
  document version the entry used, indexed by the `by-doc` GSI for invalidation.
Both item kinds carry `customer_id` and `agent_id` for the `by-customer` / `by-agent` GSIs.
`expires_at` is the DynamoDB TTL attribute; reads also check it because TTL deletion is lazy.
"""

from __future__ import annotations

import json
from typing import Any

from boto3.dynamodb.conditions import Key

from keepsake.common.aws import AWS_ERRORS, translate_aws_error
from keepsake.common.models import CacheEntry, TokenUsage

ENTRY_PREFIX = "entry#"
REF_PREFIX = "ref#"


def partition_key(team: str, customer_id: str) -> str:
    return f"team#{team}#customer#{customer_id}"


def entry_sort_key(agent_id: str, cache_key: str) -> str:
    return f"{ENTRY_PREFIX}{agent_id}#{cache_key}"


def ref_sort_key(document_version: str, agent_id: str, cache_key: str) -> str:
    return f"{REF_PREFIX}{document_version}#{agent_id}#{cache_key}"


def _entry_item(entry: CacheEntry) -> dict[str, Any]:
    return {
        "pk": partition_key(entry.team, entry.customer_id),
        "sk": entry_sort_key(entry.agent_id, entry.cache_key),
        "item_type": "entry",
        "team": entry.team,
        "customer_id": entry.customer_id,
        "agent_id": entry.agent_id,
        "cache_key": entry.cache_key,
        "model_id": entry.model_id,
        "model_version": entry.model_version,
        "document_versions": list(entry.document_versions),
        "response_text": entry.response_text,
        "usage": json.dumps(entry.usage.__dict__),
        "created_at": entry.created_at,
        "expires_at": entry.expires_at,
    }


def _ref_items(entry: CacheEntry) -> list[dict[str, Any]]:
    return [
        {
            "pk": partition_key(entry.team, entry.customer_id),
            "sk": ref_sort_key(doc_version, entry.agent_id, entry.cache_key),
            "item_type": "ref",
            "doc_ref": f"doc#{doc_version}",
            "customer_id": entry.customer_id,
            "agent_id": entry.agent_id,
            "entry_sk": entry_sort_key(entry.agent_id, entry.cache_key),
            "expires_at": entry.expires_at,
        }
        for doc_version in entry.document_versions
    ]


def _entry_from_item(item: dict[str, Any]) -> CacheEntry:
    return CacheEntry(
        team=item["team"],
        customer_id=item["customer_id"],
        agent_id=item["agent_id"],
        cache_key=item["cache_key"],
        model_id=item["model_id"],
        model_version=item["model_version"],
        document_versions=tuple(item.get("document_versions", [])),
        response_text=item["response_text"],
        usage=TokenUsage(**json.loads(item["usage"])),
        created_at=int(item["created_at"]),
        expires_at=int(item["expires_at"]),
    )


class CacheRepository:
    def __init__(self, *, table: Any) -> None:
        self._table = table

    def put(self, entry: CacheEntry) -> None:
        try:
            with self._table.batch_writer() as batch:
                batch.put_item(Item=_entry_item(entry))
                for ref in _ref_items(entry):
                    batch.put_item(Item=ref)
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "write cache entry") from exc

    def get(
        self, *, team: str, customer_id: str, agent_id: str, cache_key: str, now_epoch: int
    ) -> CacheEntry | None:
        """The live entry for exactly this partition and key, or None (expired = miss)."""
        key = {"pk": partition_key(team, customer_id), "sk": entry_sort_key(agent_id, cache_key)}
        try:
            item = self._table.get_item(Key=key).get("Item")
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "read cache entry") from exc
        if not item or int(item["expires_at"]) <= now_epoch:
            return None
        return _entry_from_item(item)

    def list_for_customer(self, customer_id: str) -> tuple[CacheEntry, ...]:
        items = self._query_index("by-customer", "customer_id", customer_id)
        return tuple(_entry_from_item(i) for i in items if i["sk"].startswith(ENTRY_PREFIX))

    def delete_by_customer(self, customer_id: str) -> int:
        return self._delete_items(self._query_index("by-customer", "customer_id", customer_id))

    def delete_by_agent(self, agent_id: str) -> int:
        return self._delete_items(self._query_index("by-agent", "agent_id", agent_id))

    def delete_by_document_version(self, document_version: str) -> int:
        """Delete every entry that used `doc@version`, with all of its references (FR4.6)."""
        refs = self._query_index("by-doc", "doc_ref", f"doc#{document_version}")
        entry_keys = {(ref["pk"], ref["entry_sk"]) for ref in refs}
        return self._delete_entries(entry_keys)

    def delete_all(self) -> int:
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {"ProjectionExpression": "pk, sk"}
        try:
            while True:
                page = self._table.scan(**kwargs)
                items.extend(page.get("Items", []))
                if "LastEvaluatedKey" not in page:
                    break
                kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "scan cache") from exc
        return self._delete_items(items)

    def _query_index(self, index: str, attribute: str, value: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "IndexName": index,
            "KeyConditionExpression": Key(attribute).eq(value),
        }
        try:
            while True:
                page = self._table.query(**kwargs)
                items.extend(page.get("Items", []))
                if "LastEvaluatedKey" not in page:
                    break
                kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, f"query cache index {index}") from exc
        return items

    def _delete_items(self, items: list[dict[str, Any]]) -> int:
        """Delete the given items and, for every entry among them, its references."""
        entry_keys = {(i["pk"], i["sk"]) for i in items if i["sk"].startswith(ENTRY_PREFIX)}
        ref_keys = {(i["pk"], i["sk"]) for i in items if i["sk"].startswith(REF_PREFIX)}
        count = self._delete_entries(entry_keys)
        self._batch_delete(ref_keys)
        return count

    def _delete_entries(self, entry_keys: set[tuple[str, str]]) -> int:
        """Delete entries and their references; returns how many entries existed."""
        keys_to_delete: set[tuple[str, str]] = set()
        count = 0
        for pk, sk in entry_keys:
            try:
                item = self._table.get_item(Key={"pk": pk, "sk": sk}).get("Item")
            except AWS_ERRORS as exc:
                raise translate_aws_error(exc, "read cache entry") from exc
            if not item:
                continue
            count += 1
            keys_to_delete.add((pk, sk))
            for doc_version in item.get("document_versions", []):
                keys_to_delete.add(
                    (pk, ref_sort_key(doc_version, item["agent_id"], item["cache_key"]))
                )
        self._batch_delete(keys_to_delete)
        return count

    def _batch_delete(self, keys: set[tuple[str, str]]) -> None:
        if not keys:
            return
        try:
            with self._table.batch_writer() as batch:
                for pk, sk in keys:
                    batch.delete_item(Key={"pk": pk, "sk": sk})
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "delete cache items") from exc
