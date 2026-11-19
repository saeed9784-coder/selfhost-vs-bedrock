"""Gateway orchestration (FR3.1-FR3.9, FR4.1, FR4.2, FR4.5, FR5.1, FR5.2, FR6.1).

Flow: route (model + blocks) -> load documents with access checks -> response-reuse lookup ->
assemble -> Converse -> store reuse entry -> shadow -> usage record, metrics, audit.

Degraded mode (FR3.9): when the context store or cache is unavailable the model is still called
at full price, without the unavailable part, and a `degraded` metric and log entry are emitted.
Access denials, model blocks and invalid input are never downgraded: they fail the request and
are audited.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import timedelta

from keepsake.cache.keys import compute_reuse_key
from keepsake.cache.repository import CacheRepository
from keepsake.common.audit import AuditLog, customer_ref
from keepsake.common.clock import Clock, epoch_seconds, iso, utc_now
from keepsake.common.config import ConfigProvider
from keepsake.common.errors import (
    AccessDenied,
    ConfigurationError,
    DependencyUnavailable,
    KeepsakeError,
    ModelBlocked,
)
from keepsake.common.models import (
    CacheEntry,
    InvokeRequest,
    InvokeResult,
    ModelCapability,
    ModelPrice,
    TokenUsage,
    UsageRecord,
)
from keepsake.gateway.assembler import AssembledPrompt, SharedDocument, assemble
from keepsake.gateway.bedrock_client import BedrockGateway, ModelResponse
from keepsake.gateway.router import Route, Router
from keepsake.gateway.shadow import run_shadow
from keepsake.store.service import DocumentService
from keepsake.usage.costing import RequestCost, price_model_call, price_reuse
from keepsake.usage.metrics import MetricsEmitter
from keepsake.usage.repository import UsageRepository

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatewayDependencies:
    config: ConfigProvider
    router: Router
    documents: DocumentService
    cache: CacheRepository
    usage: UsageRepository
    audit: AuditLog
    bedrock: BedrockGateway
    metrics: MetricsEmitter
    now: Clock = utc_now
    new_request_id: Callable[[], str] = field(default=lambda: uuid.uuid4().hex)
    perf_counter: Callable[[], float] = time.perf_counter


@dataclass(frozen=True)
class _Context:
    """Per-request state passed between steps."""

    request_id: str
    request: InvokeRequest
    route: Route
    price: ModelPrice
    storage_price: float
    documents: tuple[SharedDocument, ...]
    degraded_reasons: tuple[str, ...]
    reuse_key: str | None


class GatewayService:
    def __init__(self, deps: GatewayDependencies) -> None:
        self._d = deps

    def invoke(self, request: InvokeRequest) -> InvokeResult:
        started = self._d.perf_counter()
        request_id = self._d.new_request_id()
        route = self._route(request_id, request)
        documents, reasons = self._load_documents(request_id, route, request)
        context = self._context(request_id, request, route, documents, reasons)
        entry, reasons = self._lookup(context)
        context = _with_reasons(context, reasons)
        if entry is not None:
            response, cost, outcome = _reuse_response(context, entry)
        else:
            response, cost, outcome, reasons = self._call_model(context)
            context = _with_reasons(context, reasons)
        latency_ms = int((self._d.perf_counter() - started) * 1000)
        result = InvokeResult(
            request_id=request_id,
            answer=response.text,
            model_id=route.agent.model_id,
            model_version=route.capability.model_version,
            cache_outcome=outcome,
            document_versions=tuple(d.ref for d in context.documents),
            usage=response.usage,
            latency_ms=latency_ms,
            degraded=bool(context.degraded_reasons),
        )
        self._record(context, result, cost)
        return result

    # ---- steps ---------------------------------------------------------------------------
    def _route(self, request_id: str, request: InvokeRequest) -> Route:
        try:
            return self._d.router.route(request.agent_id, request.customer_id)
        except ModelBlocked as exc:
            self._audit(
                request_id,
                "model.blocked",
                request.agent_id,
                {
                    "agent_id": request.agent_id,
                    "customer_ref": customer_ref(request.customer_id),
                    "model_id": exc.model_id,
                    "customer_group": exc.customer_group,
                },
            )
            self._d.metrics.component_request("gateway", is_error=True)
            raise

    def _load_documents(
        self,
        request_id: str,
        route: Route,
        request: InvokeRequest,
    ) -> tuple[tuple[SharedDocument, ...], tuple[str, ...]]:
        loaded: list[SharedDocument] = []
        for document_id in request.document_ids:
            try:
                item = self._d.documents.current_for_agent(route.agent, document_id)
            except AccessDenied as exc:
                self._audit(
                    request_id,
                    "document.access_denied",
                    route.agent.agent_id,
                    {
                        "agent_id": route.agent.agent_id,
                        "document_id": document_id,
                        "reason": exc.message,
                    },
                )
                raise
            except DependencyUnavailable:
                self._degraded("context-store", request_id)
                return (), ("context-store",)
            loaded.append(
                SharedDocument(item.version.document_id, item.version.version_id, item.text)
            )
        return tuple(loaded), ()

    def _context(
        self,
        request_id: str,
        request: InvokeRequest,
        route: Route,
        documents: tuple[SharedDocument, ...],
        reasons: tuple[str, ...],
    ) -> _Context:
        prices, storage_price = self._d.config.prices()
        price = prices.get(route.agent.model_id)
        if price is None:
            raise ConfigurationError(f"no price configured for model {route.agent.model_id}")
        reuse_key = (
            None
            if reasons
            else compute_reuse_key(
                agent_id=route.agent.agent_id,
                customer_id=request.customer_id,
                model_id=route.agent.model_id,
                model_version=route.capability.model_version,
                system_prompt=route.agent.system_prompt,
                document_versions=[d.ref for d in documents],
                messages=(*request.history, *request.messages),
            )
        )
        return _Context(
            request_id, request, route, price, storage_price, documents, reasons, reuse_key
        )

    def _lookup(self, context: _Context) -> tuple[CacheEntry | None, tuple[str, ...]]:
        if context.reuse_key is None:
            return None, ()
        try:
            entry = self._d.cache.get(
                team=context.route.agent.team,
                customer_id=context.request.customer_id,
                agent_id=context.route.agent.agent_id,
                cache_key=context.reuse_key,
                now_epoch=epoch_seconds(self._d.now()),
            )
        except DependencyUnavailable:
            self._degraded("cache", context.request_id)
            return None, ("cache",)
        valid = (
            entry is not None
            and entry.model_id == context.route.agent.model_id
            and entry.model_version == context.route.capability.model_version
        )
        return (entry if valid else None), ()

    def _call_model(
        self,
        context: _Context,
    ) -> tuple[ModelResponse, RequestCost, str, tuple[str, ...]]:
        agent = context.route.agent
        prompt = self._prompt(context, context.route.capability)
        response = self._d.bedrock.converse(agent.model_id, prompt)
        stored_bytes, reasons = self._store(context, response)
        cost = price_model_call(
            response.usage,
            context.price,
            stored_bytes,
            agent.retention_hours,
            context.storage_price,
        )
        outcome = "provider_hit" if response.usage.cache_read_tokens > 0 else "miss"
        if context.route.shadow is not None:
            run_shadow(
                self._d.bedrock,
                request_id=context.request_id,
                shadow_model_id=context.route.shadow.model_id,
                prompt=self._prompt(context, context.route.shadow),
                primary=response,
            )
        return response, cost, outcome, reasons

    def _prompt(self, context: _Context, capability: ModelCapability) -> AssembledPrompt:
        return assemble(
            system_prompt=context.route.agent.system_prompt,
            documents=context.documents,
            history=context.request.history,
            messages=context.request.messages,
            capability=capability,
        )

    def _store(self, context: _Context, response: ModelResponse) -> tuple[int, tuple[str, ...]]:
        """Store the reuse entry; returns (bytes stored, degraded reasons)."""
        if context.reuse_key is None:
            return 0, ()
        now = self._d.now()
        agent = context.route.agent
        entry = CacheEntry(
            team=agent.team,
            customer_id=context.request.customer_id,
            agent_id=agent.agent_id,
            cache_key=context.reuse_key,
            model_id=agent.model_id,
            model_version=context.route.capability.model_version,
            document_versions=tuple(d.ref for d in context.documents),
            response_text=response.text,
            usage=response.usage,
            created_at=epoch_seconds(now),
            expires_at=epoch_seconds(now + timedelta(hours=agent.retention_hours)),
        )
        try:
            self._d.cache.put(entry)
        except DependencyUnavailable:
            self._degraded("cache-write", context.request_id)
            return 0, ("cache-write",)
        return len(response.text.encode("utf-8")), ()

    def _record(self, context: _Context, result: InvokeResult, cost: RequestCost) -> None:
        agent = context.route.agent
        record = UsageRecord(
            request_id=result.request_id,
            timestamp=iso(self._d.now()),
            agent_id=agent.agent_id,
            team=agent.team,
            customer_id=context.request.customer_id,
            model_id=agent.model_id,
            **_usage_tokens(result.usage, is_reuse=result.cache_outcome == "response_reuse"),
            cost_usd=cost.cost_usd,
            saving_usd=cost.saving_usd,
            storage_cost_usd=cost.storage_cost_usd,
            latency_ms=result.latency_ms,
            cache_outcome=result.cache_outcome,
            degraded=result.degraded,
        )
        try:
            self._d.usage.put(record)
        except KeepsakeError as exc:
            LOGGER.error(
                "usage record not written",
                extra={"request_id": result.request_id, "error": exc.message},
            )
            self._d.metrics.degraded("gateway", "usage-write")
        self._d.metrics.gateway_request(
            agent_id=agent.agent_id,
            team=agent.team,
            cache_outcome=result.cache_outcome,
            input_tokens=record.input_tokens,
            cache_read_tokens=record.cache_read_tokens,
            output_tokens=record.output_tokens,
            latency_ms=result.latency_ms,
            cost_usd=cost.cost_usd,
            saving_usd=cost.saving_usd,
            degraded=result.degraded,
        )
        self._audit(
            result.request_id,
            "gateway.request",
            agent.agent_id,
            {
                "agent_id": agent.agent_id,
                "team": agent.team,
                "customer_ref": customer_ref(context.request.customer_id),
                "model_id": result.model_id,
                "model_version": result.model_version,
                "document_versions": list(result.document_versions),
                "cache_outcome": result.cache_outcome,
                "degraded": list(context.degraded_reasons),
                "shadow_model": context.route.shadow.model_id if context.route.shadow else None,
            },
        )

    # ---- helpers -------------------------------------------------------------------------
    def _audit(self, request_id: str, event: str, actor: str, details: dict[str, object]) -> None:
        """Audit failures are logged at ERROR and counted; they do not fail the agent request."""
        try:
            self._d.audit.record(f"req#{request_id}", event, actor, details)
        except KeepsakeError as exc:
            LOGGER.error(
                "audit write failed",
                extra={"request_id": request_id, "event": event, "error": exc.message},
            )
            self._d.metrics.degraded("gateway", "audit-write")

    def _degraded(self, reason: str, request_id: str) -> None:
        LOGGER.warning(
            "degraded mode: calling the model at full price",
            extra={"request_id": request_id, "reason": reason},
        )
        self._d.metrics.degraded("gateway", reason)


def _with_reasons(context: _Context, reasons: tuple[str, ...]) -> _Context:
    if not reasons:
        return context
    return replace(context, degraded_reasons=(*context.degraded_reasons, *reasons))


def _usage_tokens(usage: TokenUsage, *, is_reuse: bool) -> dict[str, int]:
    """Token fields of a usage record (FR5.1).

    A model call records what Bedrock billed (`input_tokens` = all input tokens, of which
    `full_price_tokens` were billed at the full input price). A response-reuse hit sent nothing
    to the model: all of the original input is counted as served from cache.
    """
    if is_reuse:
        return {
            "input_tokens": 0,
            "cache_read_tokens": usage.total_input_tokens,
            "cache_write_tokens": 0,
            "full_price_tokens": 0,
            "output_tokens": 0,
        }
    return {
        "input_tokens": usage.total_input_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "full_price_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def _reuse_response(context: _Context, entry: CacheEntry) -> tuple[ModelResponse, RequestCost, str]:
    response = ModelResponse(
        model_id=entry.model_id,
        text=entry.response_text,
        usage=entry.usage,
        latency_ms=0,
        stop_reason="response_reuse",
    )
    return response, price_reuse(entry.usage, context.price), "response_reuse"
