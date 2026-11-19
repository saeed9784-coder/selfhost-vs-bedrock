"""Monthly cost projection: one chosen setup, the cost curve, and the cheapest setup available.

A "setup" is an instance type plus a purchase option:
* ``on_demand``: list price, billed only for the hours online.
* ``spot``: Spot price, billed only for the hours online; can be interrupted.
* ``savings_plan``: 1-year EC2 Instance Savings Plan, billed for every hour of the month.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from selfhost.pricing import MonthlyComparison, Throughput, Workload, monthly_comparison
from selfhost.settings import PRICE_FIELDS, ModelSpec

Purchase = Literal["on_demand", "spot", "savings_plan"]
PURCHASE_OPTIONS: tuple[Purchase, ...] = ("on_demand", "spot", "savings_plan")
PRICE_FIELD = PRICE_FIELDS
CURVE_POINTS = 25  # 1M .. 1T input tokens a month, four points per decade
SEARCH_LOW, SEARCH_HIGH = 1e5, 1e13


@dataclass(frozen=True)
class Scenario:
    """Everything about the workload and pricing except the setup being priced."""

    model: ModelSpec
    output_ratio: float
    prefix_reuse: float
    target_utilization: float
    online_hours: float
    hours_per_month: float
    fixed_monthly_usd: float


@dataclass(frozen=True)
class Setup:
    instance_type: str
    purchase: Purchase
    usd_per_hour: float
    throughput: Throughput
    kind: str


def setup_for(
    instances: dict[str, Any],
    model: ModelSpec,
    instance_type: str,
    purchase: Purchase,
    throughput_override: tuple[float | None, float | None] = (None, None),
) -> Setup | None:
    """The priced setup, or None when the model cannot run there or no price is known."""
    info = instances.get(instance_type)
    if info is None:
        return None
    kind = str(info.get("kind", "gpu"))
    if not model.fits(info):
        return None
    assumed = model.throughput_for(kind)
    price = info.get(PRICE_FIELD[purchase])
    if assumed is None or price is None:
        return None
    scale = float(info.get("throughput_scale", 1.0))  # bigger machine, proportionally faster
    prefill = throughput_override[0] or assumed[0] * scale
    decode = throughput_override[1] or assumed[1] * scale
    return Setup(instance_type, purchase, float(price), Throughput(prefill, decode), kind)


def compare(scenario: Scenario, setup: Setup, input_tokens: float) -> MonthlyComparison:
    return monthly_comparison(
        Workload(input_tokens, input_tokens * scenario.output_ratio, scenario.prefix_reuse),
        input_per_million=scenario.model.input_per_million,
        output_per_million=scenario.model.output_per_million,
        throughput=setup.throughput,
        usd_per_hour=setup.usd_per_hour,
        fixed_monthly_usd=scenario.fixed_monthly_usd,
        hours_per_month=scenario.hours_per_month,
        target_utilization=scenario.target_utilization,
        online_hours=scenario.online_hours,
        committed=setup.purchase == "savings_plan",
    )


def point(scenario: Scenario, setup: Setup, input_tokens: float) -> dict[str, Any]:
    result = compare(scenario, setup, input_tokens)
    return {
        "monthly_input_tokens": input_tokens,
        "bedrock_usd": round(result.bedrock_usd, 2),
        "selfhost_usd": round(result.selfhost_usd, 2),
        "instances": result.instances,
        "utilization": round(result.utilization, 4),
    }


def curve(scenario: Scenario, setup: Setup) -> list[dict[str, Any]]:
    return [point(scenario, setup, 10 ** (6 + step * 0.25)) for step in range(CURVE_POINTS)]


def break_even(scenario: Scenario, setup: Setup) -> float | None:
    """Smallest monthly input volume at which self-hosting costs no more than Bedrock.

    Self-hosting is a step function (whole instances) and Bedrock is linear, so the first
    crossing is found by scanning a fine log grid and then bisecting inside that step.
    """

    def cheaper(tokens: float) -> bool:
        result = compare(scenario, setup, tokens)
        return result.selfhost_usd <= result.bedrock_usd

    steps = 240
    previous = SEARCH_LOW
    for step in range(steps + 1):
        tokens = SEARCH_LOW * (SEARCH_HIGH / SEARCH_LOW) ** (step / steps)
        if cheaper(tokens):
            low, high = previous, tokens
            for _ in range(40):
                middle = (low * high) ** 0.5
                low, high = (low, middle) if cheaper(middle) else (middle, high)
            return high
        previous = tokens
    return None


def cheapest_setups(
    scenario: Scenario, instances: dict[str, Any], input_tokens: float, limit: int = 3
) -> list[dict[str, Any]]:
    """Every instance type and purchase option the model can run on, cheapest first."""
    options = []
    for instance_type in instances:
        for purchase in PURCHASE_OPTIONS:
            setup = setup_for(instances, scenario.model, instance_type, purchase)
            if setup is None:
                continue
            options.append(
                {
                    "instance_type": instance_type,
                    "purchase": purchase,
                    "kind": setup.kind,
                    **point(scenario, setup, input_tokens),
                }
            )
    return sorted(options, key=lambda o: o["selfhost_usd"])[:limit]
