"""Append-only audit log (FR6.1).

Items: `pk=<subject>` (e.g. `req#<request_id>`, `doc#<document_id>`), `sk=<timestamp>#<event_id>`.
Every write is conditional on the key not existing, so an event can never be overwritten.
Customer ids are not written raw: request events carry `customer_ref`, a SHA-256 of the id, so
customer erasure (FR6.2) does not have to rewrite the append-only log.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from boto3.dynamodb.conditions import Key

from keepsake.common.aws import AWS_ERRORS, translate_aws_error
from keepsake.common.clock import Clock, iso, utc_now
from keepsake.common.models import AuditEvent


def customer_ref(customer_id: str) -> str:
    """A stable pseudonymous reference to a customer for the append-only audit log."""
    return hashlib.sha256(f"keepsake-customer:{customer_id}".encode()).hexdigest()


def _item(event: AuditEvent) -> dict[str, Any]:
    return {
        "pk": event.subject,
        "sk": f"{event.timestamp}#{event.event_id}",
        "event_id": event.event_id,
        "event_type": event.event_type,
        "actor": event.actor,
        "timestamp": event.timestamp,
        "details": json.dumps(event.details, sort_keys=True, default=str),
    }


def _event(item: dict[str, Any]) -> AuditEvent:
    return AuditEvent(
        subject=item["pk"],
        event_id=item["event_id"],
        event_type=item["event_type"],
        actor=item["actor"],
        timestamp=item["timestamp"],
        details=json.loads(item["details"]),
    )


class AuditLog:
    def __init__(self, *, table: Any, now: Clock = utc_now) -> None:
        self._table = table
        self._now = now

    def record(
        self, subject: str, event_type: str, actor: str, details: dict[str, Any]
    ) -> AuditEvent:
        """Append one event; raises DependencyUnavailable if the write fails."""
        event = AuditEvent(
            subject=subject,
            event_id=uuid.uuid4().hex,
            event_type=event_type,
            actor=actor,
            timestamp=iso(self._now()),
            details=details,
        )
        try:
            self._table.put_item(
                Item=_item(event),
                ConditionExpression="attribute_not_exists(pk) AND attribute_not_exists(sk)",
            )
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "write audit event") from exc
        return event

    def lookup(self, subject: str) -> tuple[AuditEvent, ...]:
        try:
            items = self._table.query(KeyConditionExpression=Key("pk").eq(subject)).get("Items", [])
        except AWS_ERRORS as exc:
            raise translate_aws_error(exc, "read audit events") from exc
        return tuple(_event(item) for item in items)

    def lookup_request(self, request_id: str) -> tuple[AuditEvent, ...]:
        return self.lookup(f"req#{request_id}")

    def lookup_document(self, document_id: str) -> tuple[AuditEvent, ...]:
        return self.lookup(f"doc#{document_id}")
