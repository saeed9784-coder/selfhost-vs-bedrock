"""Phase 0 analyzer CLI.

    python -m keepsake.analyzer --sample --output reports/            # generated SOW traffic
    python -m keepsake.analyzer --input logs/*.json.gz --output out/  # local invocation logs

Run from the repository root with `src` on PYTHONPATH (or after `pip install -e .`).
Prices come from `config/prices.json`; no AWS access is needed for local runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from keepsake.analyzer import sample_logs
from keepsake.analyzer.logs import load_from_paths, parse_lines
from keepsake.analyzer.pipeline import run_analysis
from keepsake.analyzer.report import render_markdown
from keepsake.common.clock import utc_now
from keepsake.common.config import default_config_dir, parse_prices
from keepsake.common.errors import KeepsakeError


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="keepsake.analyzer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample", action="store_true", help="analyze generated sample logs")
    source.add_argument("--input", nargs="+", type=Path, help="invocation-log files")
    parser.add_argument("--output", type=Path, required=True, help="directory for the report")
    parser.add_argument(
        "--build-run-cost",
        type=float,
        default=500.0,
        help="monthly build-and-run cost of the cache in USD (default 500)",
    )
    parser.add_argument("--seed", type=int, default=7, help="sample generator seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.build_run_cost < 0:
        print("--build-run-cost must not be negative", file=sys.stderr)
        return 2
    try:
        prices, storage = parse_prices(
            json.loads((default_config_dir() / "prices.json").read_text(encoding="utf-8"))
        )
        if args.sample:
            parsed = parse_lines(sample_logs.to_lines(sample_logs.generate(seed=args.seed)))
        else:
            parsed = load_from_paths(args.input)
        report = run_analysis(parsed, prices, storage, args.build_run_cost, utc_now())
    except (KeepsakeError, OSError, ValueError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report.to_dict(), indent=2), "utf-8")
    (args.output / "report.md").write_text(render_markdown(report), "utf-8")
    print(f"verdict: {report.verdict}; report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
