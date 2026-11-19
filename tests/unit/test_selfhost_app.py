"""HTTP tests for the self-hosted gateway: fake vLLM over httpx.MockTransport, fake Bedrock."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from selfhost import settings as settings_module
from selfhost.app import RateLimiter, create_app

TOKEN = "test-token"
ORIGIN = "origin-secret"
AUTH = {"Authorization": f"Bearer {TOKEN}", "X-Origin-Verify": ORIGIN}


def vllm_stream(cached: int = 0, text: str = "Hello there") -> bytes:
    chunks = [{"choices": [{"index": 0, "delta": {"content": w + " "}}]} for w in text.split()]
    chunks.append(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": len(text.split()),
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
    )
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


class FakeVllm:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail_hosts: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host in self.fail_hosts:
            raise httpx.ConnectError("refused", request=request)
        payload = json.loads(request.content)
        self.requests.append({"host": request.url.host, **payload})
        cached = 64 if len(self.requests) > 1 else 0
        return httpx.Response(
            200, content=vllm_stream(cached), headers={"content-type": "text/event-stream"}
        )


class FakeBedrock:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def converse_stream(self, **kwargs: Any) -> dict[str, Iterator[dict[str, Any]]]:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Hi "}}},
            {"contentBlockDelta": {"delta": {"text": "from Bedrock"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 100, "outputTokens": 3}}},
        ]
        return {"stream": iter(events)}


def make_client(
    fake_vllm: FakeVllm, bedrock: FakeBedrock, replicas: str | None = None, rate: int = 30
) -> TestClient:
    env = {
        "SELFHOST_ACCESS_TOKEN": TOKEN,
        "SELFHOST_ORIGIN_SECRET": ORIGIN,
        "SELFHOST_RATE_LIMIT_PER_MINUTE": str(rate),
    }
    if replicas:
        env["SELFHOST_REPLICAS"] = replicas
    settings = settings_module.from_env(env)
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake_vllm))
    app = create_app(settings, http_client=http, bedrock_client=bedrock, poll_load=False)
    return TestClient(app)


def events(response: httpx.Response) -> list[dict[str, Any]]:
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


def chat_body(**overrides: Any) -> dict[str, Any]:
    conversation = [{"role": "user", "content": "Hello"}]
    return {"targets": {"selfhost": conversation, "bedrock": conversation}, **overrides}


@pytest.fixture
def fake_vllm() -> FakeVllm:
    return FakeVllm()


def test_health_is_open_but_api_needs_origin_header_and_token(fake_vllm: FakeVllm) -> None:
    with make_client(fake_vllm, FakeBedrock()) as client:
        assert client.get("/healthz").json()["ok"] is True
        assert client.get("/v1/config").status_code == 403
        assert client.get("/v1/config", headers={"X-Origin-Verify": ORIGIN}).status_code == 401
        wrong = {"X-Origin-Verify": ORIGIN, "Authorization": "Bearer nope"}
        assert client.get("/v1/config", headers=wrong).status_code == 401
        config = client.get("/v1/config", headers=AUTH).json()
        assert config["bedrock"]["model_id"] == "us.meta.llama3-1-8b-instruct-v1:0"
        assert config["auth_required"] is True


def test_chat_streams_both_targets_with_costs(fake_vllm: FakeVllm) -> None:
    bedrock = FakeBedrock()
    with make_client(fake_vllm, bedrock) as client:
        response = client.post("/v1/chat", json=chat_body(system="Be brief"), headers=AUTH)
    assert response.status_code == 200
    got = events(response)
    assert got[-1] == {"type": "end"}
    text = {
        t: "".join(e["text"] for e in got if e.get("target") == t and e["type"] == "delta")
        for t in ("selfhost", "bedrock")
    }
    assert text == {"selfhost": "Hello there ", "bedrock": "Hi from Bedrock"}
    done = {e["target"]: e["metrics"] for e in got if e.get("type") == "done"}
    assert done["bedrock"]["cost_usd"] == pytest.approx((100 + 3) * 0.22 / 1e6)
    assert done["selfhost"]["input_tokens"] == 100
    assert done["selfhost"]["capacity_cost_usd"] > 0
    assert done["selfhost"]["replica"] == "local"
    assert fake_vllm.requests[0]["messages"][0] == {"role": "system", "content": "Be brief"}
    assert bedrock.calls[0]["system"] == [{"text": "Be brief"}]
    assert bedrock.calls[0]["messages"] == [{"role": "user", "content": [{"text": "Hello"}]}]


def test_stats_accumulate_cached_tokens(fake_vllm: FakeVllm) -> None:
    with make_client(fake_vllm, FakeBedrock()) as client:
        for _ in range(2):
            client.post("/v1/chat", json=chat_body(), headers=AUTH)
        stats = client.get("/v1/stats", headers=AUTH).json()
    selfhost = stats["targets"]["selfhost"]
    assert selfhost["requests"] == 2
    assert selfhost["cached_tokens"] == 64
    assert stats["targets"]["bedrock"]["requests"] == 2


def test_bedrock_failure_is_reported_without_breaking_selfhost(fake_vllm: FakeVllm) -> None:
    with make_client(fake_vllm, FakeBedrock(error=RuntimeError("throttled"))) as client:
        got = events(client.post("/v1/chat", json=chat_body(), headers=AUTH))
    errors = [e for e in got if e.get("type") == "error"]
    assert errors == [
        {"target": "bedrock", "type": "error", "message": "Bedrock call failed: RuntimeError"}
    ]
    assert any(e.get("target") == "selfhost" and e["type"] == "done" for e in got)


def test_unreachable_replica_is_retried_on_the_next(fake_vllm: FakeVllm) -> None:
    fake_vllm.fail_hosts = {"down"}
    replicas = json.dumps(
        [{"name": "a-down", "url": "http://down"}, {"name": "b-up", "url": "http://up"}]
    )
    with make_client(fake_vllm, FakeBedrock(), replicas=replicas) as client:
        body = chat_body(targets={"selfhost": [{"role": "user", "content": "hi"}]})
        got = events(client.post("/v1/chat", json=body, headers=AUTH))
        health = client.get("/healthz").json()
    done = [e for e in got if e.get("type") == "done"]
    assert len(done) == 1
    assert done[0]["metrics"]["replica"] == "b-up"  # ties go to the first; it failed over
    assert all(r["host"] == "up" for r in fake_vllm.requests)
    assert health["replicas"] == {"a-down": False, "b-up": True}


@pytest.mark.parametrize(
    "body",
    [
        {"targets": {}},
        {"targets": {"selfhost": [{"role": "assistant", "content": "x"}]}},
        {
            "targets": {
                "selfhost": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
            }
        },
        {"targets": {"openai": [{"role": "user", "content": "a"}]}},
        {"targets": {"selfhost": [{"role": "user", "content": "a"}]}, "max_tokens": 4096},
    ],
)
def test_chat_rejects_invalid_requests(fake_vllm: FakeVllm, body: dict[str, Any]) -> None:
    with make_client(fake_vllm, FakeBedrock()) as client:
        assert client.post("/v1/chat", json=body, headers=AUTH).status_code == 422


def test_rate_limit_returns_429(fake_vllm: FakeVllm) -> None:
    with make_client(fake_vllm, FakeBedrock(), rate=1) as client:
        assert client.post("/v1/chat", json=chat_body(), headers=AUTH).status_code == 200
        limited = client.post("/v1/chat", json=chat_body(), headers=AUTH)
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "60"


def test_rate_limiter_window_slides() -> None:
    now = [0.0]
    limiter = RateLimiter(2, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")
    now[0] = 61
    assert limiter.allow("a")


def test_projection_returns_point_curve_and_break_even(fake_vllm: FakeVllm) -> None:
    with make_client(fake_vllm, FakeBedrock()) as client:
        result = client.post(
            "/v1/projection",
            json={"monthly_input_tokens": 1e10, "prefix_reuse": 0.8, "output_ratio": 0.1},
            headers=AUTH,
        ).json()
        unknown = client.post(
            "/v1/projection",
            json={"monthly_input_tokens": 1e9, "instance_type": "x1.weird"},
            headers=AUTH,
        )
    assert len(result["curve"]) == 25
    assert result["point"]["bedrock_usd"] == pytest.approx(1e10 * 1.1 * 0.22 / 1e6, rel=1e-3)
    assert result["break_even_input_tokens"] is None  # default setup is the CPU node
    assert len(result["cheapest"]) == 3
    assert result["cheapest"][0]["selfhost_usd"] <= result["cheapest"][1]["selfhost_usd"]
    assert unknown.status_code == 422


def test_chat_ui_is_served(fake_vllm: FakeVllm) -> None:
    settings = replace(settings_module.from_env({}), origin_secret="", access_token="")
    app = create_app(settings, bedrock_client=FakeBedrock(), poll_load=False)
    with TestClient(app) as client:
        page = client.get("/")
    assert page.status_code == 200
    assert "Inference Cost Lab" in page.text


def test_bedrock_error_hides_details_but_keeps_code() -> None:
    from botocore.exceptions import ClientError

    error = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "arn:aws:iam::123:role/x"}},
        "ConverseStream",
    )
    with make_client(FakeVllm(), FakeBedrock(error=error)) as client:
        got = events(client.post("/v1/chat", json=chat_body(), headers=AUTH))
    message = next(e["message"] for e in got if e.get("type") == "error")
    assert "AccessDeniedException" in message
    assert "arn:aws" not in message


def test_unexpected_failure_still_ends_the_target(fake_vllm: FakeVllm) -> None:
    class Broken(FakeBedrock):
        def converse_stream(self, **kwargs: Any) -> Any:
            return {"stream": iter([{"contentBlockDelta": {"delta": None}}])}  # malformed

    with make_client(fake_vllm, Broken()) as client:
        got = events(client.post("/v1/chat", json=chat_body(), headers=AUTH))
    assert {"target": "bedrock", "type": "error", "message": "internal error"} in got
    assert got[-1] == {"type": "end"}


def test_rate_limit_key_ignores_spoofed_forwarded_for(fake_vllm: FakeVllm) -> None:
    with make_client(fake_vllm, FakeBedrock(), rate=1) as client:
        first = {**AUTH, "X-Forwarded-For": "1.1.1.1", "CloudFront-Viewer-Address": "9.9.9.9:443"}
        spoof = {**AUTH, "X-Forwarded-For": "2.2.2.2", "CloudFront-Viewer-Address": "9.9.9.9:555"}
        other = {**AUTH, "CloudFront-Viewer-Address": "8.8.8.8:443"}
        assert client.post("/v1/chat", json=chat_body(), headers=first).status_code == 200
        assert client.post("/v1/chat", json=chat_body(), headers=spoof).status_code == 429
        assert client.post("/v1/chat", json=chat_body(), headers=other).status_code == 200


def test_health_reports_model_not_ready_until_metrics_poll_succeeds() -> None:
    settings = settings_module.from_env({})
    app = create_app(
        settings,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(FakeVllm())),
        bedrock_client=FakeBedrock(),
        poll_load=False,
    )
    with TestClient(app) as client:
        assert client.get("/healthz").json()["model_ready"] is True  # no poller: trusted
    polling = create_app(
        settings,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(503, request=r))
        ),
        bedrock_client=FakeBedrock(),
        poll_load=True,
    )
    with TestClient(polling) as client:
        health = client.get("/healthz").json()
    assert health["ok"] is True
    assert health["model_ready"] is False


def test_heartbeat_ping_is_sent_while_waiting(
    fake_vllm: FakeVllm, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time as time_module

    from selfhost import service

    monkeypatch.setattr(service, "HEARTBEAT_SECONDS", 0.05)

    class Slow(FakeBedrock):
        def converse_stream(self, **kwargs: Any) -> Any:
            time_module.sleep(0.3)
            return super().converse_stream(**kwargs)

    with make_client(fake_vllm, Slow()) as client:
        body = chat_body(targets={"bedrock": [{"role": "user", "content": "hi"}]})
        got = events(client.post("/v1/chat", json=body, headers=AUTH))
    assert {"type": "ping"} in got
    assert any(e.get("type") == "done" for e in got)


def test_llamacpp_requests_ask_for_prompt_cache_and_read_cache_n(fake_vllm: FakeVllm) -> None:
    with make_client(fake_vllm, FakeBedrock()) as client:
        client.post("/v1/chat", json=chat_body(), headers=AUTH)
    assert fake_vllm.requests[0]["cache_prompt"] is True


def test_projection_uses_throughput_of_the_chosen_hardware(fake_vllm: FakeVllm) -> None:
    with make_client(fake_vllm, FakeBedrock()) as client:
        config = client.get("/v1/config", headers=AUTH).json()
        cpu = client.post(
            "/v1/projection",
            json={"monthly_input_tokens": 1e10, "instance_type": "c7i.2xlarge"},
            headers=AUTH,
        ).json()
        gpu = client.post(
            "/v1/projection",
            json={"monthly_input_tokens": 1e10, "instance_type": "g6e.xlarge"},
            headers=AUTH,
        ).json()
    assert config["model"]["engine"] == "llama.cpp"
    assert config["selfhost"]["kind"] == "cpu"
    assert cpu["instance_kind"] == "cpu"
    assert gpu["instance_kind"] == "gpu"
    assert cpu["point"]["instances"] > gpu["point"]["instances"]


def test_bedrock_reasoning_is_streamed_separately_from_the_answer(fake_vllm: FakeVllm) -> None:
    class Thinking(FakeBedrock):
        def converse_stream(self, **kwargs: Any) -> Any:
            events = [
                {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "Let me think. "}}}},
                {"contentBlockDelta": {"delta": {"text": "OK"}}},
                {"metadata": {"usage": {"inputTokens": 40, "outputTokens": 30}}},
            ]
            return {"stream": iter(events)}

    with make_client(fake_vllm, Thinking()) as client:
        body = chat_body(targets={"bedrock": [{"role": "user", "content": "hi"}]})
        got = events(client.post("/v1/chat", json=body, headers=AUTH))
    assert {"target": "bedrock", "type": "reasoning", "text": "Let me think. "} in got
    answer = "".join(e["text"] for e in got if e.get("type") == "delta")
    assert answer == "OK"
    done = next(e for e in got if e.get("type") == "done")
    assert done["metrics"]["output_tokens"] == 30  # reasoning tokens are billed


def test_projection_accepts_purchase_option_and_hours(fake_vllm: FakeVllm) -> None:
    with make_client(fake_vllm, FakeBedrock()) as client:
        result = client.post(
            "/v1/projection",
            json={
                "monthly_input_tokens": 1e9,
                "instance_type": "g6e.xlarge",
                "purchase": "spot",
                "online_hours": 220,
            },
            headers=AUTH,
        ).json()
        bad = client.post(
            "/v1/projection",
            json={"monthly_input_tokens": 1e9, "purchase": "free"},
            headers=AUTH,
        )
    assert (result["purchase"], result["online_hours"]) == ("spot", 220)
    assert result["break_even_input_tokens"] is not None
    assert bad.status_code == 422


def test_cpu_capacity_cost_follows_measured_time_not_assumed_speed(fake_vllm: FakeVllm) -> None:
    with make_client(fake_vllm, FakeBedrock()) as client:
        body = chat_body(targets={"selfhost": [{"role": "user", "content": "hi"}]})
        got = events(client.post("/v1/chat", json=body, headers=AUTH))
    metrics = next(e["metrics"] for e in got if e.get("type") == "done")
    # default deploy is llama.cpp on CPU at 60% target utilization
    assert metrics["capacity_cost_usd"] == pytest.approx(metrics["cost_usd"] / 0.6)


def test_reasoning_comparator_gets_extra_budget_and_pages_are_not_cached(
    fake_vllm: FakeVllm,
) -> None:
    from selfhost.backends import token_budget

    assert token_budget(512, reasoning=True) == 512 + 1024
    assert token_budget(6000, reasoning=True) == 8192
    assert token_budget(512, reasoning=False) == 512
    with make_client(fake_vllm, FakeBedrock()) as client:
        page = client.get("/", headers={"X-Origin-Verify": ORIGIN})
        api = client.get("/v1/config", headers=AUTH)
    assert page.headers["Cache-Control"] == "no-cache"
    assert "no-cache" not in api.headers.get("Cache-Control", "")


def test_selfhost_reasoning_content_is_streamed_as_reasoning() -> None:
    def thinking_vllm(request: httpx.Request) -> httpx.Response:
        chunks = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "Thinking. "}}]},
            {"choices": [{"index": 0, "delta": {"content": "Answer"}}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body.encode())

    with make_client(thinking_vllm, FakeBedrock()) as client:  # type: ignore[arg-type]
        body = chat_body(targets={"selfhost": [{"role": "user", "content": "hi"}]})
        got = events(client.post("/v1/chat", json=body, headers=AUTH))
    assert {"target": "selfhost", "type": "reasoning", "text": "Thinking. "} in got
    assert "".join(e["text"] for e in got if e.get("type") == "delta") == "Answer"
