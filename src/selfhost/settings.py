"""Gateway settings, read once from the environment and the bundled price catalog."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).with_name("catalog.json")
DEFAULT_MODEL = "llama-3.1-8b"
DEFAULT_INSTANCE = "c7i.2xlarge"
DEFAULT_REPLICAS = '[{"name": "local", "url": "http://127.0.0.1:8000", "max_context": 32768}]'


class SettingsError(ValueError):
    """Raised when the environment or catalog describes an unusable configuration."""


@dataclass(frozen=True)
class Replica:
    """One vLLM endpoint the router may send requests to."""

    name: str
    url: str
    max_context: int = 32768
    max_inflight: int = 64


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    hf_model: str
    tokenizer_version: str
    bedrock_model_id: str
    bedrock_label: str
    same_model: bool  # False when Bedrock runs a different model than the self-hosted one
    bedrock_reasoning: bool  # thinks before answering; needs extra output-token budget
    input_per_million: float
    output_per_million: float
    engine: str
    prefill_tps: float  # for the deployed instance's kind (cpu or gpu)
    decode_tps: float
    throughput_by_kind: tuple[tuple[str, float, float], ...] = ()
    reasoning: bool = False  # the self-hosted model thinks before answering
    memory_gb: tuple[tuple[str, float], ...] = ()  # memory the model needs, per instance kind

    def throughput_for(self, kind: str) -> tuple[float, float] | None:
        return next(((p, d) for k, p, d in self.throughput_by_kind if k == kind), None)

    def fits(self, instance: dict[str, Any]) -> bool:
        """True when the instance kind has a throughput figure and enough memory for the model."""
        kind = str(instance.get("kind", "gpu"))
        if self.throughput_for(kind) is None:
            return False
        needed = dict(self.memory_gb).get(kind, 0.0)
        field = "gpu_memory_gib" if kind == "gpu" else "memory_gib"
        return float(instance.get(field, needed)) >= needed


@dataclass(frozen=True)
class Settings:
    model: ModelSpec
    replicas: tuple[Replica, ...]
    instance_type: str
    instance_usd_per_hour: float
    purchase: str
    instance_kind: str
    fixed_monthly_usd: dict[str, float]
    hours_per_month: float
    retention_hours: dict[str, float]
    region: str = "us-east-1"
    access_token: str = ""
    origin_secret: str = ""
    metrics_namespace: str = ""
    web_dir: str = ""
    rate_limit_per_minute: int = 30
    max_output_tokens: int = 2048
    target_utilization: float = 0.6


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bedrock_target(catalog: dict[str, Any], model_key: str, override: str = "") -> dict[str, Any]:
    """The Bedrock model to compare against: the same model by default, or a catalog override."""
    entry = catalog["models"].get(model_key)
    if entry is None:
        raise SettingsError(
            f"Unknown model {model_key!r}; choose one of {sorted(catalog['models'])}"
        )
    if not override:
        return {
            "label": entry["label"],
            "reasoning": bool(entry.get("reasoning", False)),
            "model_id": entry["bedrock_model_id"],
            "base_model_id": entry["bedrock_base_model_id"],
            "price": entry["bedrock_price"],
            "same_model": True,
        }
    other = catalog.get("bedrock_models", {}).get(override)
    if other is None:
        raise SettingsError(
            f"Unknown Bedrock model {override!r}; choose one of "
            f"{sorted(catalog.get('bedrock_models', {}))}"
        )
    return {**other, "same_model": False}


def _model_spec(
    catalog: dict[str, Any],
    key: str,
    instance_kind: str,
    bedrock_override: str = "",
    throughput_scale: float = 1.0,
) -> ModelSpec:
    target = bedrock_target(catalog, key, bedrock_override)
    entry = catalog["models"][key]
    price = target["price"]
    by_kind = entry["throughput_assumption"]
    throughput = by_kind.get(instance_kind)
    if throughput is None:
        raise SettingsError(f"Model {key!r} has no {instance_kind} throughput; it cannot run there")
    return ModelSpec(
        key=key,
        label=entry["label"],
        hf_model=entry["gguf"] if instance_kind == "cpu" else entry["hf_model"],
        tokenizer_version=entry["tokenizer_version"],
        bedrock_model_id=target["model_id"],
        bedrock_label=target["label"],
        same_model=target["same_model"],
        bedrock_reasoning=bool(target.get("reasoning", False)),
        reasoning=bool(entry.get("reasoning", False)),
        memory_gb=tuple(sorted((k, float(v)) for k, v in entry.get("memory_gb", {}).items())),
        input_per_million=float(price["input_per_million"]),
        output_per_million=float(price["output_per_million"]),
        engine="llama.cpp" if instance_kind == "cpu" else "vllm",
        prefill_tps=float(throughput["prefill_tokens_per_second"]) * throughput_scale,
        decode_tps=float(throughput["decode_tokens_per_second"]) * throughput_scale,
        throughput_by_kind=tuple(
            (kind, float(v["prefill_tokens_per_second"]), float(v["decode_tokens_per_second"]))
            for kind, v in sorted(by_kind.items())
        ),
    )


def _replicas(raw: str) -> tuple[Replica, ...]:
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SettingsError(f"SELFHOST_REPLICAS is not valid JSON: {exc}") from exc
    if not isinstance(items, list) or not items:
        raise SettingsError("SELFHOST_REPLICAS must be a non-empty JSON list")
    replicas = []
    for item in items:
        if not isinstance(item, dict) or not item.get("url") or not item.get("name"):
            raise SettingsError(f"Each replica needs a name and url, got {item!r}")
        replicas.append(
            Replica(
                name=str(item["name"]),
                url=str(item["url"]).rstrip("/"),
                max_context=int(item.get("max_context", 32768)),
                max_inflight=int(item.get("max_inflight", 64)),
            )
        )
    return tuple(replicas)


PRICE_FIELDS = {
    "on_demand": "usd_per_hour",
    "spot": "spot_usd_per_hour",
    "savings_plan": "savings_plan_usd_per_hour",
}


def _hourly_price(instance: dict[str, Any], purchase: str, instance_type: str) -> float:
    field = PRICE_FIELDS.get(purchase)
    if field is None:
        raise SettingsError(f"SELFHOST_PURCHASE must be one of {sorted(PRICE_FIELDS)}")
    if field not in instance:
        raise SettingsError(f"No {purchase} price for {instance_type} in catalog.json")
    return float(instance[field])


def from_env(env: dict[str, str] | None = None, catalog: dict[str, Any] | None = None) -> Settings:
    env = dict(os.environ if env is None else env)
    catalog = catalog or load_catalog()
    instance_type = env.get("SELFHOST_INSTANCE_TYPE", DEFAULT_INSTANCE)
    instance = catalog["instances"].get(instance_type)
    if instance is None:
        raise SettingsError(f"No price for instance type {instance_type!r} in catalog.json")
    purchase = env.get("SELFHOST_PURCHASE", "on_demand")
    return Settings(
        model=_model_spec(
            catalog,
            env.get("SELFHOST_MODEL", DEFAULT_MODEL),
            str(instance.get("kind", "gpu")),
            env.get("SELFHOST_BEDROCK_MODEL", ""),
            float(instance.get("throughput_scale", 1.0)),
        ),
        instance_kind=str(instance.get("kind", "gpu")),
        replicas=_replicas(env.get("SELFHOST_REPLICAS", DEFAULT_REPLICAS)),
        instance_type=instance_type,
        instance_usd_per_hour=_hourly_price(instance, purchase, instance_type),
        purchase=purchase,
        fixed_monthly_usd={k: float(v) for k, v in catalog["fixed_monthly_usd"].items()},
        hours_per_month=float(catalog["hours_per_month"]),
        retention_hours={k: float(v) for k, v in catalog["retention_hours"].items()},
        region=env.get("AWS_REGION", "us-east-1"),
        access_token=env.get("SELFHOST_ACCESS_TOKEN", ""),
        origin_secret=env.get("SELFHOST_ORIGIN_SECRET", ""),
        metrics_namespace=env.get("SELFHOST_METRICS_NAMESPACE", ""),
        web_dir=env.get("SELFHOST_WEB_DIR", ""),
        rate_limit_per_minute=int(env.get("SELFHOST_RATE_LIMIT_PER_MINUTE", "30")),
        max_output_tokens=int(env.get("SELFHOST_MAX_OUTPUT_TOKENS", "2048")),
        target_utilization=float(env.get("SELFHOST_TARGET_UTILIZATION", "0.6")),
    )
