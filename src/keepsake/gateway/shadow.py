"""Shadow mode: call a second model after the primary returns and log both (FR3.6).

The shadow call is best effort: any failure is logged and never affects the primary response.
It runs synchronously after the primary call inside the same Lambda invocation, so it adds the
shadow model's latency to the agent's request only while shadow mode is configured.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from keepsake.common.errors import KeepsakeError
from keepsake.gateway.assembler import AssembledPrompt
from keepsake.gateway.bedrock_client import BedrockGateway, ModelResponse

LOGGER = logging.getLogger(__name__)
MAX_LOGGED_ANSWER_CHARS = 4000


@dataclass(frozen=True)
class ShadowComparison:
    request_id: str
    primary_model: str
    shadow_model: str
    primary_input_tokens: int
    shadow_input_tokens: int
    primary_output_tokens: int
    shadow_output_tokens: int
    primary_latency_ms: int
    shadow_latency_ms: int
    primary_answer: str
    shadow_answer: str


def run_shadow(
    bedrock: BedrockGateway,
    *,
    request_id: str,
    shadow_model_id: str,
    prompt: AssembledPrompt,
    primary: ModelResponse,
) -> ShadowComparison | None:
    try:
        shadow = bedrock.converse(shadow_model_id, prompt)
    except KeepsakeError as exc:
        LOGGER.warning(
            "shadow call failed; primary response unaffected",
            extra={"request_id": request_id, "shadow_model": shadow_model_id, "error": exc.message},
        )
        return None
    comparison = ShadowComparison(
        request_id=request_id,
        primary_model=primary.model_id,
        shadow_model=shadow.model_id,
        primary_input_tokens=primary.usage.total_input_tokens,
        shadow_input_tokens=shadow.usage.total_input_tokens,
        primary_output_tokens=primary.usage.output_tokens,
        shadow_output_tokens=shadow.usage.output_tokens,
        primary_latency_ms=primary.latency_ms,
        shadow_latency_ms=shadow.latency_ms,
        primary_answer=primary.text[:MAX_LOGGED_ANSWER_CHARS],
        shadow_answer=shadow.text[:MAX_LOGGED_ANSWER_CHARS],
    )
    LOGGER.info(
        "shadow comparison %s", json.dumps(asdict(comparison)), extra={"event": "shadow_comparison"}
    )
    return comparison
