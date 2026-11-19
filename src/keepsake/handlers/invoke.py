"""POST /invoke: the agent-facing gateway endpoint (FR3.1).

Authentication: API Gateway IAM (SigV4) authorization. The agent identity is derived from the
verified caller role ARN through the SSM agent registry; an `agent_id` in the body, if present,
must match it, so one agent can never act as another.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from keepsake.common.errors import AccessDenied, ValidationError
from keepsake.common.http import Event, Response, api_handler, caller_arn, ok, parse_body
from keepsake.common.models import InvokeRequest, Message
from keepsake.handlers.dependencies import container
from keepsake.handlers.observability import LOGGER


def _messages(raw: Any, field_name: str) -> tuple[Message, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValidationError(f"{field_name} must be a list")
    messages = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValidationError(f"each {field_name} item must be an object")
        messages.append(Message(role=item.get("role"), text=item.get("text")))
    return tuple(messages)


def parse_invoke_request(body: dict[str, Any], agent_id: str) -> InvokeRequest:
    claimed = body.get("agent_id")
    if claimed is not None and claimed != agent_id:
        raise AccessDenied("agent_id does not match the calling identity")
    document_ids = body.get("document_ids", [])
    if not isinstance(document_ids, list):
        raise ValidationError("document_ids must be a list")
    return InvokeRequest(
        agent_id=agent_id,
        customer_id=body.get("customer_id"),
        document_ids=tuple(document_ids),
        messages=_messages(body.get("messages"), "messages"),
        history=_messages(body.get("history"), "history"),
    )


@LOGGER.inject_lambda_context
@api_handler("gateway")
def handler(event: Event, context: Any) -> Response:
    deps = container()
    agent_id = deps.config.resolve_agent(caller_arn(event))
    request = parse_invoke_request(parse_body(event), agent_id)
    result = deps.gateway().invoke(request)
    data = asdict(result)
    data["usage"] = asdict(result.usage)
    return ok(data)
