"""Unit tests for the self-hosted gateway's pure logic: pricing, prefix index, router, settings."""

from __future__ import annotations

import pytest

from selfhost import projection
from selfhost import settings as settings_module
from selfhost.prefix_index import BLOCK_CHARS, PrefixIndex, block_hashes
from selfhost.pricing import (
    Throughput,
    Usage,
    Workload,
    bedrock_cost,
    monthly_comparison,
    selfhost_actual_cost,
    selfhost_capacity_cost,
)
from selfhost.router import NoReplicaForContext, Overloaded, ReplicaLoad, Router, parse_vllm_metrics
from selfhost.settings import Replica, SettingsError


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


THROUGHPUT = Throughput(prefill_tps=10_000, decode_tps=1_000)
LONG_CHAT = [{"role": "user", "content": "x" * (BLOCK_CHARS * 3)}]


# ---------- pricing ----------
def test_bedrock_cost_charges_input_and_output_per_million() -> None:
    assert bedrock_cost(Usage(1_000_000, 500_000), 0.22, 0.44) == pytest.approx(0.44)


def test_capacity_cost_skips_prefill_for_cached_tokens() -> None:
    cold = selfhost_capacity_cost(Usage(10_000, 100), THROUGHPUT, 3600.0, 1.0)
    warm = selfhost_capacity_cost(Usage(10_000, 100, cached_tokens=10_000), THROUGHPUT, 3600.0, 1.0)
    assert cold == pytest.approx(1.0 + 0.1)  # 1 s prefill + 0.1 s decode at $1/s
    assert warm == pytest.approx(0.1)


def test_capacity_cost_rejects_zero_utilization() -> None:
    with pytest.raises(ValueError, match="utilization"):
        selfhost_capacity_cost(Usage(1, 1), THROUGHPUT, 1.0, 0)


def test_actual_cost_is_shared_by_concurrent_requests() -> None:
    alone = selfhost_actual_cost(10, 1, 3600)
    shared = selfhost_actual_cost(10, 4, 3600)
    assert alone == pytest.approx(10)
    assert shared == pytest.approx(2.5)
    assert selfhost_actual_cost(10, 0.2, 3600) == pytest.approx(10)  # never below one share


def test_monthly_comparison_low_volume_favours_bedrock() -> None:
    result = monthly_comparison(
        Workload(1e8, 1e7, prefix_reuse=0.5),
        input_per_million=0.22,
        output_per_million=0.22,
        throughput=THROUGHPUT,
        usd_per_hour=2.0,
        fixed_monthly_usd=50,
        hours_per_month=730,
        target_utilization=0.6,
    )
    assert result.instances == 1
    assert result.bedrock_usd == pytest.approx(24.2)
    assert result.selfhost_usd == pytest.approx(1510)
    assert result.savings_usd < 0


def test_monthly_comparison_adds_instances_when_capacity_runs_out() -> None:
    result = monthly_comparison(
        Workload(1e12, 1e11, prefix_reuse=0.9),
        input_per_million=0.22,
        output_per_million=0.22,
        throughput=THROUGHPUT,
        usd_per_hour=2.0,
        fixed_monthly_usd=0,
        hours_per_month=730,
        target_utilization=0.5,
    )
    gpu_hours = (1e11 / 10_000 + 1e11 / 1_000) / 3600
    assert result.instances == -(-gpu_hours // 365)
    assert result.utilization <= 1


@pytest.mark.parametrize("reuse", [-0.1, 1.0])
def test_monthly_comparison_rejects_bad_reuse(reuse: float) -> None:
    with pytest.raises(ValueError, match="prefix_reuse"):
        monthly_comparison(
            Workload(1, 1, reuse),
            input_per_million=1,
            output_per_million=1,
            throughput=THROUGHPUT,
            usd_per_hour=1,
            fixed_monthly_usd=0,
            hours_per_month=730,
            target_utilization=0.5,
        )


# ---------- prefix index ----------
def test_block_hashes_are_chained_and_model_scoped() -> None:
    a = block_hashes(LONG_CHAT, model="m", tokenizer_version="t1")
    b = block_hashes(LONG_CHAT, model="m", tokenizer_version="t2")
    longer = block_hashes(
        [*LONG_CHAT, {"role": "assistant", "content": "y" * BLOCK_CHARS}],
        model="m",
        tokenizer_version="t1",
    )
    assert len(a) == 3
    assert set(a).isdisjoint(b)  # tokenizer change never reuses entries
    assert longer[: len(a)] == a  # a longer conversation shares the earlier prefix


def test_short_prompt_has_no_full_block() -> None:
    assert block_hashes([{"role": "user", "content": "hi"}], model="m", tokenizer_version="t") == ()


def test_prefix_index_matches_leading_blocks_until_ttl() -> None:
    clock = Clock()
    index = PrefixIndex({"chat": 1, "agent": 2}, clock=clock)
    hashes = block_hashes(LONG_CHAT, model="m", tokenizer_version="t")
    index.record(hashes[:2], "r1", "chat")
    assert index.matched_blocks(hashes, "r1") == 2
    assert index.matched_blocks(hashes, "r2") == 0
    clock.now += 3601
    assert index.matched_blocks(hashes, "r1") == 0
    assert index.evict_expired() == 2


# ---------- router ----------
def _router(clock: Clock, **overrides: int) -> tuple[Router, PrefixIndex]:
    index = PrefixIndex({"chat": 1}, clock=clock)
    replicas = [
        Replica("short", "http://a", max_context=8000, max_inflight=overrides.get("short", 2)),
        Replica("long", "http://b", max_context=32000, max_inflight=overrides.get("long", 2)),
    ]
    return Router(replicas, index, clock=clock), index


def test_router_prefers_replica_holding_the_prefix() -> None:
    clock = Clock()
    router, index = _router(clock)
    hashes = block_hashes(LONG_CHAT, model="m", tokenizer_version="t")
    router.set_load("long", ReplicaLoad(running=50))
    index.record(hashes, "long", "chat")
    decision = router.route(hashes, 1000)
    assert decision.replica.name == "long"
    assert decision.matched_blocks == 3


def test_router_uses_load_without_prefix_and_respects_context_tier() -> None:
    clock = Clock()
    router, _ = _router(clock)
    router.set_load("short", ReplicaLoad(running=5))
    assert router.route((), 1000).replica.name == "long"
    assert router.route((), 20_000).replica.name == "long"
    with pytest.raises(NoReplicaForContext):
        router.route((), 50_000)


def test_router_backpressure_and_unhealthy_replicas() -> None:
    clock = Clock()
    router, _ = _router(clock, long=1)
    with router.lease("long"), pytest.raises(Overloaded):
        router.route((), 20_000)
    router.mark_unhealthy("long")
    with pytest.raises(Overloaded):
        router.route((), 20_000)
    clock.now += 60
    assert router.route((), 20_000).replica.name == "long"


def test_busy_integral_measures_concurrency() -> None:
    clock = Clock()
    router, _ = _router(clock)
    start = router.busy_integral()
    with router.lease("short"), router.lease("long"):
        clock.now += 10
    assert router.busy_integral() - start == pytest.approx(20)


def test_parse_vllm_metrics_handles_old_and_new_names() -> None:
    text = """
# HELP vllm:num_requests_running running
vllm:num_requests_running{model_name="m"} 3.0
vllm:num_requests_waiting{model_name="m"} 1.0
vllm:gpu_cache_usage_perc{model_name="m"} 0.5
vllm:prefix_cache_hits_total{model_name="m"} 40
vllm:prefix_cache_queries_total{model_name="m"} 100
"""
    load = parse_vllm_metrics(text)
    assert (load.running, load.waiting, load.kv_usage) == (3.0, 1.0, 0.5)
    assert load.score == pytest.approx(3 + 2 + 5)
    assert parse_vllm_metrics("vllm:kv_cache_usage_perc 0.25").kv_usage == 0.25


# ---------- settings ----------
def test_settings_from_env_reads_catalog() -> None:
    settings = settings_module.from_env(
        {
            "SELFHOST_MODEL": "qwen3-32b",
            "SELFHOST_INSTANCE_TYPE": "p5.48xlarge",
            "SELFHOST_REPLICAS": '[{"name": "a", "url": "http://x/"}]',
        }
    )
    assert settings.model.bedrock_model_id == "qwen.qwen3-32b-v1:0"
    assert settings.instance_usd_per_hour == pytest.approx(55.04)
    assert settings.replicas[0].url == "http://x"


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"SELFHOST_MODEL": "nope"}, "Unknown model"),
        ({"SELFHOST_INSTANCE_TYPE": "t3.micro"}, "No price"),
        ({"SELFHOST_REPLICAS": "not json"}, "not valid JSON"),
        ({"SELFHOST_REPLICAS": "[]"}, "non-empty"),
        ({"SELFHOST_REPLICAS": '[{"url": "http://x"}]'}, "name and url"),
    ],
)
def test_settings_rejects_bad_configuration(env: dict[str, str], message: str) -> None:
    with pytest.raises(SettingsError, match=message):
        settings_module.from_env(env)


def test_llamacpp_metrics_and_cache_n_are_understood() -> None:
    from selfhost.backends import _openai_usage

    load = parse_vllm_metrics(
        "llamacpp:requests_processing 2\nllamacpp:requests_deferred 1\n"
        "llamacpp:kv_cache_usage_ratio 0.5\n"
    )
    assert (load.running, load.waiting, load.kv_usage) == (2.0, 1.0, 0.5)
    usage = _openai_usage({"prompt_tokens": 50, "completion_tokens": 5}, {"cache_n": 32})
    assert usage.cached_tokens == 32


def test_model_without_throughput_for_hardware_is_rejected() -> None:
    with pytest.raises(SettingsError, match="no cpu throughput"):
        settings_module.from_env(
            {"SELFHOST_MODEL": "qwen3-32b", "SELFHOST_INSTANCE_TYPE": "c7i.2xlarge"}
        )


def test_bedrock_comparator_override_uses_its_own_price() -> None:
    settings = settings_module.from_env({"SELFHOST_BEDROCK_MODEL": "minimax-m2.5"})
    assert settings.model.bedrock_model_id == "minimax.minimax-m2.5"
    assert (settings.model.input_per_million, settings.model.output_per_million) == (0.30, 1.20)
    assert settings.model.same_model is False
    assert settings_module.from_env({}).model.same_model is True
    with pytest.raises(SettingsError, match="Unknown Bedrock model"):
        settings_module.from_env({"SELFHOST_BEDROCK_MODEL": "gpt-9"})


# ---------- projection ----------
def _scenario(online_hours: float = 730) -> projection.Scenario:
    model = settings_module.from_env({"SELFHOST_BEDROCK_MODEL": "minimax-m2.5"}).model
    return projection.Scenario(model, 0.15, 0.6, 0.6, online_hours, 730, 50.0)


def test_savings_plan_bills_every_hour_but_spot_only_hours_online() -> None:
    catalog = settings_module.load_catalog()
    scenario = _scenario(online_hours=100)
    spot = projection.setup_for(catalog["instances"], scenario.model, "g6e.xlarge", "spot")
    plan = projection.setup_for(catalog["instances"], scenario.model, "g6e.xlarge", "savings_plan")
    assert spot and plan
    spot_cost = projection.compare(scenario, spot, 1e8).selfhost_usd
    plan_cost = projection.compare(scenario, plan, 1e8).selfhost_usd
    assert spot_cost == pytest.approx(100 * spot.usd_per_hour + 50)
    assert plan_cost == pytest.approx(730 * plan.usd_per_hour + 50)


def test_fewer_hours_online_lowers_break_even_and_cpu_never_breaks_even() -> None:
    catalog = settings_module.load_catalog()
    gpu = projection.setup_for(catalog["instances"], _scenario().model, "g6e.xlarge", "on_demand")
    cpu = projection.setup_for(catalog["instances"], _scenario().model, "c7i.2xlarge", "on_demand")
    always_on = projection.break_even(_scenario(730), gpu)
    business_hours = projection.break_even(_scenario(220), gpu)
    assert always_on and business_hours and business_hours < always_on
    result = projection.compare(_scenario(730), gpu, always_on)
    assert result.selfhost_usd <= result.bedrock_usd * 1.0001
    assert projection.break_even(_scenario(730), cpu) is None


def test_cheapest_setups_are_sorted_and_skip_unrunnable_hardware() -> None:
    catalog = settings_module.load_catalog()
    model = settings_module.from_env(
        {"SELFHOST_MODEL": "qwen3-32b", "SELFHOST_INSTANCE_TYPE": "g6e.xlarge"}
    ).model
    scenario = projection.Scenario(model, 0.15, 0.6, 0.6, 220, 730, 50.0)
    options = projection.cheapest_setups(scenario, catalog["instances"], 1e9, limit=50)
    assert options
    assert all(o["kind"] == "gpu" for o in options)  # qwen3-32b has no CPU figure
    costs = [o["selfhost_usd"] for o in options]
    assert costs == sorted(costs)


def test_online_hours_must_fit_in_the_month() -> None:
    with pytest.raises(ValueError, match="online_hours"):
        monthly_comparison(
            Workload(1, 1, 0.5),
            input_per_million=1,
            output_per_million=1,
            throughput=THROUGHPUT,
            usd_per_hour=1,
            fixed_monthly_usd=0,
            hours_per_month=730,
            target_utilization=0.5,
            online_hours=800,
        )


def test_throughput_scales_with_machine_size() -> None:
    catalog = settings_module.load_catalog()
    model = _scenario().model
    small = projection.setup_for(catalog["instances"], model, "c7i.xlarge", "on_demand")
    ref = projection.setup_for(catalog["instances"], model, "c7i.2xlarge", "on_demand")
    four_gpu = projection.setup_for(catalog["instances"], model, "g6e.12xlarge", "on_demand")
    one_gpu = projection.setup_for(catalog["instances"], model, "g6e.xlarge", "on_demand")
    assert small.throughput.decode_tps == pytest.approx(ref.throughput.decode_tps / 2)
    assert four_gpu.throughput.decode_tps == pytest.approx(one_gpu.throughput.decode_tps * 4)
    deployed = settings_module.from_env({"SELFHOST_INSTANCE_TYPE": "c6i.2xlarge"})
    assert deployed.model.decode_tps == pytest.approx(6.5 * 0.85)


def test_deployed_node_is_priced_by_purchase_option() -> None:
    spot = settings_module.from_env(
        {"SELFHOST_INSTANCE_TYPE": "c7i.xlarge", "SELFHOST_PURCHASE": "spot"}
    )
    assert spot.instance_usd_per_hour == pytest.approx(0.0502)
    assert spot.purchase == "spot"
    with pytest.raises(SettingsError, match="SELFHOST_PURCHASE"):
        settings_module.from_env({"SELFHOST_PURCHASE": "free"})


def test_minimax_only_offered_on_machines_with_enough_memory() -> None:
    catalog = settings_module.load_catalog()
    model = settings_module.from_env(
        {"SELFHOST_MODEL": "minimax-m2.5", "SELFHOST_INSTANCE_TYPE": "r7a.4xlarge"}
    ).model
    assert model.same_model and model.reasoning and model.bedrock_reasoning
    assert model.bedrock_model_id == "minimax.minimax-m2.5"
    assert projection.setup_for(catalog["instances"], model, "g6e.xlarge", "spot") is None
    assert projection.setup_for(catalog["instances"], model, "c7a.2xlarge", "spot") is None
    assert projection.setup_for(catalog["instances"], model, "g6e.12xlarge", "spot") is not None
    assert projection.setup_for(catalog["instances"], model, "r7a.4xlarge", "spot") is not None
