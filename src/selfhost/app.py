"""HTTP API and chat UI host for the self-hosted inference gateway.

Run locally:  SELFHOST_REPLICAS='[{"name":"local","url":"http://127.0.0.1:8000"}]' \
              uvicorn selfhost.app:app --port 8080
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from selfhost import projection as projection_module
from selfhost import settings as settings_module
from selfhost.backends import BedrockBackend, ChatRequest, VllmBackend
from selfhost.prefix_index import PrefixIndex
from selfhost.router import Router, parse_vllm_metrics
from selfhost.service import ChatService
from selfhost.settings import Settings
from selfhost.telemetry import CloudWatchMetrics, Stats

LOGGER = logging.getLogger("selfhost.app")
REPO_WEB_DIR = Path(__file__).resolve().parents[2] / "web" / "chat"
LOAD_POLL_SECONDS = 5.0
MAX_TOTAL_CHARS = 400_000
ORIGIN_HEADER = "x-origin-verify"
OPEN_PATHS = frozenset({"/healthz"})
ENGINE_LABELS = {"vllm": "vLLM", "llama.cpp": "llama.cpp"}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatBody(BaseModel):
    system: str = Field(default="", max_length=8000)
    targets: dict[Literal["selfhost", "bedrock"], list[Message]] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)  # Bedrock Converse caps at 1.0
    workload: Literal["chat", "agent"] = "chat"

    @field_validator("targets")
    @classmethod
    def conversations_alternate(cls, value: dict[str, list[Message]]) -> dict[str, list[Message]]:
        for target, messages in value.items():
            if not 1 <= len(messages) <= 200:
                raise ValueError(f"{target}: between 1 and 200 messages")
            for position, message in enumerate(messages):
                expected = "user" if position % 2 == 0 else "assistant"
                if message.role != expected:
                    raise ValueError(f"{target}: messages must alternate, starting with user")
            if messages[-1].role != "user":
                raise ValueError(f"{target}: the last message must be from the user")
        return value


class ProjectionBody(BaseModel):
    monthly_input_tokens: float = Field(gt=0, le=1e13)
    output_ratio: float = Field(default=0.15, ge=0, le=5)
    prefix_reuse: float = Field(default=0.6, ge=0, lt=1)
    target_utilization: float = Field(default=0.6, gt=0, le=1)
    instance_type: str | None = None
    prefill_tps: float | None = Field(default=None, gt=0)
    decode_tps: float | None = Field(default=None, gt=0)
    online_hours: float | None = Field(default=None, gt=0, le=744)
    purchase: Literal["on_demand", "spot", "savings_plan"] = "on_demand"


class RateLimiter:
    """Sliding one-minute window per client key."""

    def __init__(self, per_minute: int, clock: Any = time.monotonic) -> None:
        self._limit = per_minute
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - 60:
            hits.popleft()
        if len(hits) >= self._limit:
            return False
        hits.append(now)
        if len(self._hits) > 10_000:
            self._hits = {k: v for k, v in self._hits.items() if v}
        return True


def _client_key(request: Request) -> str:
    """Rate-limit key. X-Forwarded-For's first entry is viewer-supplied, so it is not used;
    CloudFront-Viewer-Address is set by CloudFront itself."""
    viewer = request.headers.get("cloudfront-viewer-address", "")
    if viewer:
        return viewer.rsplit(":", 1)[0]
    return request.client.host if request.client else "?"


def _bearer_ok(request: Request, token: str) -> bool:
    supplied = request.headers.get("authorization", "")
    return hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode())


def create_app(
    settings: Settings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
    bedrock_client: Any = None,
    cloudwatch_client: Any = None,
    poll_load: bool = True,
) -> FastAPI:
    settings = settings or settings_module.from_env()
    catalog = settings_module.load_catalog()
    index = PrefixIndex(settings.retention_hours)
    router = Router(settings.replicas, index)
    stats = Stats()
    limiter = RateLimiter(settings.rate_limit_per_minute)
    state: dict[str, Any] = {}
    polled: set[str] = set() if poll_load else {r.name for r in settings.replicas}

    if bedrock_client is None:
        import boto3  # imported lazily so tests and offline runs need no AWS credentials

        bedrock_client = boto3.client("bedrock-runtime", region_name=settings.region)
    metrics = None
    if settings.metrics_namespace:
        if cloudwatch_client is None:
            import boto3

            cloudwatch_client = boto3.client("cloudwatch", region_name=settings.region)
        metrics = CloudWatchMetrics(cloudwatch_client, settings.metrics_namespace)

    async def poll(client: httpx.AsyncClient) -> None:
        while True:
            for replica in router.replicas:
                try:
                    response = await client.get(f"{replica.url}/metrics", timeout=3.0)
                    response.raise_for_status()
                    router.set_load(replica.name, parse_vllm_metrics(response.text))
                    polled.add(replica.name)
                except httpx.HTTPError as exc:
                    LOGGER.warning("metrics poll failed for %s: %s", replica.name, exc)
                    router.mark_unhealthy(replica.name)
                    polled.discard(replica.name)
            await asyncio.sleep(LOAD_POLL_SECONDS)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=10.0)
        )
        state["service"] = ChatService(
            settings,
            router,
            index,
            VllmBackend(
                client,
                settings.model.key,
                extra_body={"cache_prompt": True} if settings.model.engine == "llama.cpp" else None,
                reasoning=settings.model.reasoning,
            ),
            BedrockBackend(
                bedrock_client,
                settings.model.bedrock_model_id,
                reasoning=settings.model.bedrock_reasoning,
            ),
            stats,
            metrics,
        )
        poller = asyncio.create_task(poll(client)) if poll_load else None
        if metrics:
            metrics.start()
        try:
            yield
        finally:
            if poller:
                poller.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poller
            if metrics:
                await asyncio.to_thread(metrics.stop)
            if http_client is None:
                await client.aclose()

    app = FastAPI(title="Self-hosted inference gateway", lifespan=lifespan, docs_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Any:
        path = request.url.path
        if path in OPEN_PATHS:
            return await call_next(request)
        if settings.origin_secret and not hmac.compare_digest(
            request.headers.get(ORIGIN_HEADER, "").encode(), settings.origin_secret.encode()
        ):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        if path.startswith("/v1/"):
            if settings.access_token and not _bearer_ok(request, settings.access_token):
                return JSONResponse({"error": "missing or wrong access token"}, status_code=401)
            if request.method == "POST" and not limiter.allow(_client_key(request)):
                return JSONResponse(
                    {"error": "rate limit reached, try again in a minute"},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not path.startswith("/v1/"):
            # Page files change with every deploy; make browsers revalidate instead of reusing
            # a stale copy (the files carry a fixed 1980 modification time from the zip asset).
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness for the ALB (the page must load while the model downloads); ``model_ready``
        tells the UI whether any vLLM replica is answering yet."""
        replicas = {r.name: router.is_healthy(r.name) and r.name in polled for r in router.replicas}
        return {"ok": True, "model_ready": any(replicas.values()), "replicas": replicas}

    @app.get("/v1/config")
    async def config() -> dict[str, Any]:
        model = settings.model
        return {
            "model": {
                "key": model.key,
                "label": model.label,
                "hf_model": model.hf_model,
                "engine": ENGINE_LABELS.get(model.engine, model.engine),
                "build": model.hf_model.rsplit(":", 1)[-1] if ":" in model.hf_model else "full",
            },
            "bedrock": {
                "model_id": model.bedrock_model_id,
                "label": model.bedrock_label,
                "same_model": model.same_model,
                "input_per_million": model.input_per_million,
                "output_per_million": model.output_per_million,
            },
            "selfhost": {
                "instance_type": settings.instance_type,
                "usd_per_hour": settings.instance_usd_per_hour,
                "kind": settings.instance_kind,
                "purchase": settings.purchase,
                "fixed_monthly_usd": settings.fixed_monthly_usd,
                "prefill_tps": model.prefill_tps,
                "decode_tps": model.decode_tps,
                "target_utilization": settings.target_utilization,
                "replicas": [r.name for r in router.replicas],
            },
            "instances": {  # only hardware this model has a throughput figure for
                name: info for name, info in catalog["instances"].items() if model.fits(info)
            },
            "hours_per_month": settings.hours_per_month,
            "max_output_tokens": settings.max_output_tokens,
            "auth_required": bool(settings.access_token),
            "price_note": catalog.get("_note", ""),
        }

    @app.get("/v1/stats")
    async def get_stats() -> dict[str, Any]:
        uptime = stats.uptime_seconds()
        return {
            "targets": stats.snapshot(),
            "uptime_seconds": round(uptime, 1),
            "gpu_cost_since_start_usd": uptime / 3600 * settings.instance_usd_per_hour,
            "prefix_index_entries": len(index),
            "replicas": [
                {
                    "name": r.name,
                    "healthy": router.is_healthy(r.name),
                    "load": router.load(r.name).__dict__,
                }
                for r in router.replicas
            ],
        }

    @app.post("/v1/chat")
    async def chat(body: ChatBody) -> StreamingResponse:
        if body.max_tokens > settings.max_output_tokens:
            raise HTTPException(422, f"max_tokens is limited to {settings.max_output_tokens}")
        total = len(body.system) + sum(
            len(m.content) for messages in body.targets.values() for m in messages
        )
        if total > MAX_TOTAL_CHARS:
            raise HTTPException(413, "conversation too long; start a new chat")
        requests = {
            target: ChatRequest(
                system=body.system,
                messages=tuple({"role": m.role, "content": m.content} for m in messages),
                max_tokens=body.max_tokens,
                temperature=body.temperature,
            )
            for target, messages in body.targets.items()
        }
        service: ChatService = state["service"]

        async def events() -> AsyncIterator[bytes]:
            async for event in service.compare(requests, body.workload):
                yield f"data: {json.dumps(event)}\n\n".encode()
            yield b'data: {"type": "end"}\n\n'

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/projection")
    async def projection(body: ProjectionBody) -> dict[str, Any]:
        instance_type = body.instance_type or settings.instance_type
        if instance_type not in catalog["instances"]:
            raise HTTPException(422, f"unknown instance type {instance_type!r}")
        online_hours = min(body.online_hours or settings.hours_per_month, settings.hours_per_month)
        scenario = projection_module.Scenario(
            model=settings.model,
            output_ratio=body.output_ratio,
            prefix_reuse=body.prefix_reuse,
            target_utilization=body.target_utilization,
            online_hours=online_hours,
            hours_per_month=settings.hours_per_month,
            fixed_monthly_usd=sum(settings.fixed_monthly_usd.values()),
        )
        setup = projection_module.setup_for(
            catalog["instances"],
            settings.model,
            instance_type,
            body.purchase,
            (body.prefill_tps, body.decode_tps),
        )
        if setup is None:
            raise HTTPException(
                422,
                f"{settings.model.key} has no {body.purchase} price or throughput for "
                f"{instance_type}",
            )
        return {
            "point": projection_module.point(scenario, setup, body.monthly_input_tokens),
            "curve": projection_module.curve(scenario, setup),
            "break_even_input_tokens": projection_module.break_even(scenario, setup),
            "instance_type": instance_type,
            "instance_kind": setup.kind,
            "purchase": body.purchase,
            "usd_per_hour": setup.usd_per_hour,
            "online_hours": online_hours,
            "cheapest": projection_module.cheapest_setups(
                scenario, catalog["instances"], body.monthly_input_tokens
            ),
        }

    web_dir = Path(settings.web_dir) if settings.web_dir else REPO_WEB_DIR
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


def __getattr__(name: str) -> Any:
    """``uvicorn selfhost.app:app`` builds the app from the environment on first access."""
    if name == "app":
        logging.basicConfig(level=logging.INFO)
        built = create_app()
        globals()["app"] = built
        return built
    raise AttributeError(name)
