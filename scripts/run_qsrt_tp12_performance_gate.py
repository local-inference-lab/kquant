#!/usr/bin/env python3
"""Run the final sealed-pool TP12 P24/P33 performance evidence matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import subprocess

from kquant.tp12_performance_gate import (
    TP12PerformanceThresholds,
    select_tp12_performance_layers,
    summarize_tp12_performance,
)
from scripts.export_qsrt_tp12_benchmark_fixture import export_fixture
from scripts.summarize_qsrt_tp12_runtime import summarize_runtime_mix


KIND = "kquant_kimi_k3_qsrt_tp12_performance_run"
SCHEMA_VERSION = 1
TP_SIZE = 12


def _layers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if (
        not result
        or len(set(result)) != len(result)
        or any(not 1 <= layer <= 92 for layer in result)
    ):
        raise argparse.ArgumentTypeError("layers must be unique values in 1..92")
    return tuple(sorted(result))


def _devices(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(result) != TP_SIZE or len(set(result)) != TP_SIZE:
        raise argparse.ArgumentTypeError(
            "exactly 12 unique physical GPU indices or UUIDs are required"
        )
    return result


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def _atomic_json(path: Path, document: dict[str, object]) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(document, handle, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_idle_gpus() -> None:
    result = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "GPU compute processes are active; refusing a contaminated benchmark:\n"
            f"{result.stdout.strip()}"
        )


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> None:
    if log_path.exists() or log_path.is_symlink():
        raise FileExistsError(log_path)
    with log_path.open("x") as log:
        log.write(f"command: {shlex.join(command)}\n")
        log.flush()
        subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
        log.flush()
        os.fsync(log.fileno())


def _benchmark_environment(device: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": device,
            "SPARKINFER_COMPILE_DISK_CACHE": "0",
        }
    )
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--devices",
        type=_devices,
        required=True,
        help="12 physical GPU indices or UUIDs, one per logical TP12 rank",
    )
    parser.add_argument(
        "--layers",
        type=_layers,
        help="explicit layer matrix; default selects five depth-balanced hotspots",
    )
    parser.add_argument("--selected-layer-count", type=_positive, default=5)
    parser.add_argument(
        "--b12x-root", type=Path, default=Path("/home/luke/projects/b12x")
    )
    parser.add_argument("--fixture-rows", type=_positive, default=2048)
    parser.add_argument("--warmup", type=_positive, default=20)
    parser.add_argument("--replays", type=_positive, default=200)
    parser.add_argument("--cold-replays", type=_nonnegative, default=50)
    parser.add_argument("--bootstrap-replicates", type=_positive, default=10_000)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_pool = args.candidate_pool.resolve()
    capture = args.capture.resolve()
    output = args.output_dir.resolve()
    b12x = args.b12x_root.resolve()
    b12x_python = b12x / ".venv/bin/python"
    isolated_script = b12x / "benchmarks/benchmark_trellis_pair_tp12.py"
    routed_script = b12x / "benchmarks/benchmark_trellis_pair_moe_tp12.py"
    for path in (b12x_python, isolated_script, routed_script):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"{output} exists; every performance run requires a fresh directory"
        )
    _require_idle_gpus()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    runtime_mix = summarize_runtime_mix(candidate_pool, capture)
    _atomic_json(output / "runtime-mix.json", runtime_mix)
    candidate_manifest = json.loads(
        (candidate_pool / "qsrt-candidate-manifest.json").read_text()
    )
    codebook = candidate_manifest.get("codebook")
    if codebook is None:
        raise ValueError("candidate manifest is missing its QSRT codebook")
    if codebook not in {"mcg", "sqg-normal-e4m3"}:
        raise ValueError(
            f"B12X performance harness does not support codebook {codebook!r}"
        )
    layers = (
        args.layers
        if args.layers is not None
        else select_tp12_performance_layers(
            runtime_mix,
            count=args.selected_layer_count,
            require_all_layers=True,
        )
    )
    if runtime_mix.get("candidate_pool_sealed") is not True:
        raise ValueError("final TP12 performance run requires a sealed pool")
    if not set(layers).issubset(set(runtime_mix["layers"])):
        raise ValueError("requested performance layers are absent from runtime mix")

    request: dict[str, object] = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "candidate_pool": str(candidate_pool),
        "capture": str(capture),
        "layers": list(layers),
        "logical_rank_devices": {
            str(rank): device for rank, device in enumerate(args.devices)
        },
        "fixture_rows": args.fixture_rows,
        "tokens": [1, 4, 16],
        "isolated_rows": [1, 2, 4, 8],
        "warmup": args.warmup,
        "replays": args.replays,
        "cold_replays": args.cold_replays,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "codebook": codebook,
        "b12x_root": str(b12x),
    }
    _atomic_json(output / "request.json", request)

    fixtures: dict[tuple[int, int], Path] = {}
    for layer in layers:
        for rank in range(TP_SIZE):
            path = output / f"fixture-l{layer:02d}-r{rank:02d}.safetensors"
            export_fixture(
                candidate_pool,
                capture,
                path,
                layer=layer,
                rank=rank,
                row_limit=args.fixture_rows,
                seed=args.seed,
            )
            fixtures[(layer, rank)] = path

    isolated_output = output / "isolated.json"
    isolated_command = [
        str(b12x_python),
        str(isolated_script),
        "--codebook",
        codebook,
        "--rows",
        "1,2,4,8",
        "--warmup",
        str(args.warmup),
        "--replays",
        str(args.replays),
        "--cold-replays",
        str(args.cold_replays),
        "--bootstrap-replicates",
        str(args.bootstrap_replicates),
        "--seed",
        str(args.seed),
        "--output",
        str(isolated_output),
    ]
    _run_logged(
        isolated_command,
        cwd=b12x,
        environment=_benchmark_environment(args.devices[0]),
        log_path=output / "isolated.log",
    )

    def run_rank(rank: int, device: str) -> list[Path]:
        paths: list[Path] = []
        for layer in layers:
            benchmark_output = output / f"routed-l{layer:02d}-r{rank:02d}.json"
            command = [
                str(b12x_python),
                str(routed_script),
                "--codebook",
                codebook,
                "--route-fixture",
                str(fixtures[(layer, rank)]),
                "--scenarios",
                "P33,sparse",
                "--tokens",
                "1,4,16",
                "--warmup",
                str(args.warmup),
                "--replays",
                str(args.replays),
                "--cold-replays",
                str(args.cold_replays),
                "--bootstrap-replicates",
                str(args.bootstrap_replicates),
                "--seed",
                str(args.seed),
                "--require-known-resources",
                "--require-no-local-memory",
                "--output",
                str(benchmark_output),
            ]
            _run_logged(
                command,
                cwd=b12x,
                environment=_benchmark_environment(device),
                log_path=output / f"routed-l{layer:02d}-r{rank:02d}.log",
            )
            paths.append(benchmark_output)
        return paths

    routed_outputs: list[Path] = []
    with ThreadPoolExecutor(max_workers=TP_SIZE) as executor:
        futures = {
            executor.submit(run_rank, rank, device): rank
            for rank, device in enumerate(args.devices)
        }
        for future in as_completed(futures):
            routed_outputs.extend(future.result())
    routed_outputs.sort()

    report = summarize_tp12_performance(
        isolated_output,
        routed_outputs,
        expected_layers=layers,
        expected_isolated_rows=(1, 2, 4, 8),
        expected_tokens=(1, 4, 16),
        expected_codebook=codebook,
        thresholds=TP12PerformanceThresholds(),
        require_sealed_pool=True,
    )
    summary_path = output / "performance-summary.json"
    _atomic_json(summary_path, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "summary": str(summary_path),
                "layers": list(layers),
                "status": report["status"],
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
