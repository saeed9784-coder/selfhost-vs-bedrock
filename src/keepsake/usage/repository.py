"""Usage-record data access (FR5.1).

Layout: `pk=team#<team>`, `sk=ts#<ISO-8601 UTC>#<request_id>`, so a team's records for a time
range are one Query. GSIs: `by-agent` (agent_id, sk) and `by-customer` (customer_id, sk, used by
customer erasure, FR6.2). Money values are stored as Decimal strings to avoid float drift.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from boto3.dynamodb.conditions import Attr, Key

from keepsake.common.aws import AWS_ERRORS, translate_aws_error
from keepsake.common.models import UsageRecord

_INT_FIELDS = (
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "full_price_tokens",
    "output_tokens",
    "latency_ms",
)
_MONEY_FIELDS = ("cost_usd", "saving_usd", "storage_cost_usd")
_END_SENTINEL = "~"  # sorts after every request id character


def _sort_key(record: UsageRecord) -> str:
    return f"ts#{record.timestamp}#{record.request_id}"


def _item(record: UsageRecord) -> dict[str, Any]:
    item: dict[str, Any] = {
        "pk": f"team#{record.team}",
        "sk": _sort_key(record),
        "request_id": record.request_id,
        "timestamp": record.timestamp,
        "agent_id": record.agent_id,
        "team": record.team,
        "customer_id": record.customer_id,
        "model_id": record.model_id,
        "cache_outcome": record.cache_outcome,
        "degraded": record.degraded,
    }
    item.update({name: getattr(record, name) for name in _INT_FIELDS})
    item.update({name: Decimal(str(round(getattr(record, name), 10))) for name in _MONEY_FIELDS})
    return item


def _record(item: dict[str, Any]) -> UsageRecord:
    return UsageRecord(
        request_id=item["request_id"],
        timestamp=item["timestamp"],
        agent_id=item["agent_id"],
        team=item["team"],
        customer_id=item["customer_id"],
        model_id=item["model_id"],
        cache_outcome=item["cache_outcome"],
        degraded=bool(item.get("degraded", False)),
        **{name: int(item[name]) for name in _INT_FIELDS},
        **{name: float(item[name]) for name in _MONEY_FIELDS},
    )


class UsageRepository:
    def __init__(self, *, table: Any) -> None:
        self._table = table

    def put(self, record: UsageRecord) -> None:
        try:
            self._table.put_item(Item=_item(record))
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "write usage record") from exc

    def query_team(self, team: str, start_iso: str, end_iso: str) -> tuple[UsageRecord, ...]:
        condition = Key("pk").eq(f"team#{team}") & Key("sk").between(
            f"ts#{start_iso}", f"ts#{end_iso}{_END_SENTINEL}"
        )
        return self._collect("query", {"KeyConditionExpression": condition}, "query team usage")

    def query_company(self, start_iso: str, end_iso: str) -> tuple[UsageRecord, ...]:
        """Every team's records in the range (finance view, FR5.4). POC scale: a filtered scan."""
        condition = Attr("sk").between(f"ts#{start_iso}", f"ts#{end_iso}{_END_SENTINEL}")
        return self._collect("scan", {"FilterExpression": condition}, "scan company usage")

    def delete_by_customer(self, customer_id: str) -> tuple[str, ...]:
        """Delete a customer's records; returns the removed request ids (FR6.2)."""
        records = self._collect(
            "query",
            {
                "IndexName": "by-customer",
                "KeyConditionExpression": Key("customer_id").eq(customer_id),
            },
            "query customer usage",
        )
        try:
            with self._table.batch_writer() as batch:
                for record in records:
                    batch.delete_item(Key={"pk": f"team#{record.team}", "sk": _sort_key(record)})
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "delete customer usage") from exc
        return tuple(record.request_id for record in records)

    def _collect(
        self, operation: str, kwargs: dict[str, Any], label: str
    ) -> tuple[UsageRecord, ...]:
        items: list[dict[str, Any]] = []
        call = getattr(self._table, operation)
        try:
            while True:
                page = call(**kwargs)
                items.extend(page.get("Items", []))
                if "LastEvaluatedKey" not in page:
                    break
                kwargs = {**kwargs, "ExclusiveStartKey": page["LastEvaluatedKey"]}
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, label) from exc
        return tuple(_record(item) for item in items)
