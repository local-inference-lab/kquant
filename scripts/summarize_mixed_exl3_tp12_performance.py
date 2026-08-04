#!/usr/bin/env python3
"""Build the fail-closed natural-route TP12 P24/P33 performance gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kquant.tp12_performance_gate import (
    TP12PerformanceThresholds,
    summarize_tp12_performance,
)


def _positive_csv(value: str, *, maximum: int | None = None) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or len(set(parsed)) != len(parsed) or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive and unique")
    if maximum is not None and any(item > maximum for item in parsed):
        raise argparse.ArgumentTypeError(f"values must not exceed {maximum}")
    return parsed


def _layers(value: str) -> tuple[int, ...]:
    return _positive_csv(value, maximum=92)


def _tokens(value: str) -> tuple[int, ...]:
    return _positive_csv(value)


def _rows(value: str) -> tuple[int, ...]:
    return _positive_csv(value)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        action="append",
        required=True,
        help="one B12X routed benchmark JSON; repeat for every layer/rank",
    )
    parser.add_argument(
        "--isolated-benchmark",
        type=Path,
        required=True,
        help="B12X isolated P24/P33 decoder benchmark JSON",
    )
    parser.add_argument("--layers", type=_layers, required=True)
    parser.add_argument(
        "--isolated-rows", type=_rows, default=_rows("1,2,4,8")
    )
    parser.add_argument("--tokens", type=_tokens, default=_tokens("1,4,16"))
    parser.add_argument(
        "--allow-unsealed",
        action="store_true",
        help="diagnostic only; production acceptance requires sealed fixtures",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = summarize_tp12_performance(
        args.isolated_benchmark,
        args.benchmark,
        expected_layers=args.layers,
        expected_isolated_rows=args.isolated_rows,
        expected_tokens=args.tokens,
        thresholds=TP12PerformanceThresholds(),
        require_sealed_pool=not args.allow_unsealed,
    )
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": report["status"],
                "scope": report["scope"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
