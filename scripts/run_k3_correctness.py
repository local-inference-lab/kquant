#!/usr/bin/env python
"""Run and compare owned Kimi-K3 correctness serving probes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from kquant.correctness import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    assert_gpus_idle,
    assert_port_free,
    compare_probe_responses,
    git_state,
    gpu_inventory,
    model_identity,
    package_versions,
    parse_env_assignments,
    probe_payload,
    process_snapshot,
    select_gpu_uuids,
    write_json,
)
from kquant.kernel_audit import KERNEL_PATHS, audit_kernel_path

DEFAULT_ENV = {
    "CUTE_DSL_ARCH": "sm_120a",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_LEVEL": "SYS",
    "NCCL_PROTO": "LL,LL128,Simple",
    "NCCL_BUFFSIZE": "2097152",
    "NCCL_MAX_NCHANNELS": "8",
    "OMP_NUM_THREADS": "16",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_USE_B12X_MOE": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
    "KDA_DISABLE_AUTOTUNE": "1",
    "B12X_MOE_FORCE_A16": "1",
    "VLLM_USE_B12X_FP8_GEMM": "1",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
    "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE": "134217728",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)


def _read_json_arg(value: str) -> dict[str, Any]:
    path = Path(value)
    if path.is_dir():
        path = path / "result.json"
    return json.loads(path.read_text())


def _python_executable(path: Path) -> Path:
    """Normalize an interpreter path without dereferencing a venv symlink."""
    return path.absolute()


def _http_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 10,
) -> dict[str, Any] | None:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


def _wait_ready(
    process: subprocess.Popen,
    base_url: str,
    *,
    started_at: float,
    boot_timeout: float,
    deadline: float,
    log_path: Path,
) -> None:
    last_report = 0.0
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            tail = log_path.read_text(errors="replace").splitlines()[-80:]
            raise RuntimeError(
                f"server exited during boot with status {code}\n" + "\n".join(tail)
            )
        try:
            _http_json(f"{base_url}/health", timeout=3)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        now = time.monotonic()
        if now - last_report >= 30:
            elapsed = int(now - started_at)
            print(f"booting: {elapsed}s elapsed, pid={process.pid}", flush=True)
            last_report = now
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy in {boot_timeout}s")


def _stop_process_group(process: subprocess.Popen, timeout: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=30)


def _server_command(args: argparse.Namespace, python: Path) -> list[str]:
    command = [
        str(python),
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.model),
        "--served-model-name",
        args.served_model_name,
        "--trust-remote-code",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tp),
        "--load-format",
        args.load_format,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    if args.kv_cache_memory_bytes is not None:
        command.extend(
            ["--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes)]
        )
    if args.graph_mode == "eager":
        command.append("--enforce-eager")
    else:
        command.extend(
            [
                "--compilation-config",
                json.dumps(
                    {
                        "mode": 0,
                        "cudagraph_mode": "PIECEWISE",
                        "cudagraph_capture_sizes": [1, 2, 4],
                    },
                    separators=(",", ":"),
                ),
            ]
        )
    command.extend(args.vllm_arg)
    return command


def run_server(args: argparse.Namespace) -> int:
    vllm_dir = args.vllm_dir.resolve()
    # Do not resolve the venv's python symlink: invoking its base-interpreter
    # target bypasses virtual-environment discovery and loses installed
    # packages such as torch and vLLM.
    python = _python_executable(args.python or vllm_dir / ".venv/bin/python")
    if not python.is_file():
        raise FileNotFoundError(python)
    model = args.model.resolve()
    selected = select_gpu_uuids(args.tp, args.gpus)
    assert_port_free(args.host, args.port)
    assert_gpus_idle(selected)

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir
        else (args.out_root / f"{stamp}-{_slug(args.tag)}").resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "server.log"

    env = os.environ.copy()
    env.update(DEFAULT_ENV)
    env.update(parse_env_assignments(args.env))
    env["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    env["PYTHONPATH"] = str(vllm_dir) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    if args.graph_mode == "piecewise":
        env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"

    command = _server_command(args, python)
    repo_roots = {
        "kquant": Path(__file__).resolve().parents[1],
        "vllm": vllm_dir,
        "b12x": args.b12x_dir.resolve(),
        "exllamav3": args.exllamav3_dir.resolve(),
    }
    manifest = {
        "schema_version": 1,
        "tag": args.tag,
        "started_at": datetime.now().astimezone().isoformat(),
        "model": model_identity(model),
        "repositories": {
            name: git_state(path) for name, path in repo_roots.items()
        },
        "runtime": package_versions(python),
        "gpu_inventory": gpu_inventory(),
        "selected_gpu_uuids": selected,
        "command": command,
        "environment": {
            key: env[key]
            for key in sorted(set(DEFAULT_ENV) | set(parse_env_assignments(args.env)))
            if key in env
        }
        | {"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]},
        "probe": probe_payload(
            model_name=args.served_model_name,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            logprobs=args.logprobs,
        ),
        "required_kernel_path": args.require_kernel_path,
    }
    write_json(run_dir / "manifest.json", manifest)

    result: dict[str, Any] = {
        "schema_version": 1,
        "tag": args.tag,
        "run_dir": str(run_dir),
        "status": "starting",
        "probes": [],
    }
    write_json(run_dir / "result.json", result)
    process: subprocess.Popen | None = None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=vllm_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            result["server_pid"] = process.pid
            result["status"] = "booting"
            write_json(run_dir / "result.json", result)
            boot_started_at = time.monotonic()
            _wait_ready(
                process,
                f"http://{args.host}:{args.port}",
                started_at=boot_started_at,
                boot_timeout=args.boot_timeout,
                deadline=boot_started_at + args.boot_timeout,
                log_path=log_path,
            )
            result["status"] = "probing"
            result["ready_after_seconds"] = (
                datetime.now().astimezone()
                - datetime.fromisoformat(manifest["started_at"])
            ).total_seconds()
            payload = manifest["probe"]
            for index in range(args.repeat_requests):
                response = _http_json(
                    f"http://{args.host}:{args.port}/v1/completions",
                    payload=payload,
                    timeout=args.probe_timeout,
                )
                if response is None:
                    raise RuntimeError("completion endpoint returned no response")
                write_json(run_dir / f"probe-{index:03d}.json", response)
                result["probes"].append(response)
            comparisons = []
            for index in range(1, len(result["probes"])):
                comparisons.append(
                    compare_probe_responses(
                        result["probes"][0],
                        result["probes"][index],
                        max_tokens=args.max_tokens,
                        logprob_limit=args.logprob_limit,
                    )
                )
            result["repeat_comparisons"] = comparisons
            result["repeat_pass"] = all(item["pass"] for item in comparisons)
            result["status"] = "passed" if result["repeat_pass"] else "failed"
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        pgid = process.pid if process is not None else None
        write_json(run_dir / "diagnostics.json", process_snapshot(pgid))
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        if process is not None:
            _stop_process_group(process, args.shutdown_timeout)
            result["server_exit_code"] = process.returncode
        if args.require_kernel_path is not None and log_path.is_file():
            kernel_audit = audit_kernel_path(
                log_path.read_text(errors="replace"),
                args.require_kernel_path,
            )
            result["kernel_audit"] = kernel_audit
            write_json(run_dir / "kernel-audit.json", kernel_audit)
            if not kernel_audit["pass"] and result["status"] == "passed":
                result["status"] = "failed"
        result["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(run_dir / "result.json", result)
    print(run_dir, flush=True)
    return 0 if result["status"] == "passed" else 2


def compare(args: argparse.Namespace) -> int:
    left = _read_json_arg(args.left)
    right = _read_json_arg(args.right)
    left_probes = left.get("probes")
    right_probes = right.get("probes")
    if not left_probes or not right_probes:
        raise ValueError("both results must contain at least one probe")
    repeat_noise = max(
        [
            float(item["max_abs_logprob_delta"])
            for result in (left, right)
            for item in result.get("repeat_comparisons", [])
        ],
        default=0.0,
    )
    limit = max(args.absolute_limit, 2 * repeat_noise)
    report = compare_probe_responses(
        left_probes[0],
        right_probes[0],
        max_tokens=args.max_tokens,
        logprob_limit=limit,
    )
    report.update(
        {
            "left_tag": left.get("tag"),
            "right_tag": right.get("tag"),
            "repeat_noise_floor": repeat_noise,
            "absolute_limit": args.absolute_limit,
        }
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["pass"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--tag", required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--tp", type=int, required=True)
    run.add_argument("--vllm-dir", type=Path, default=Path("/home/luke/projects/vllm"))
    run.add_argument("--b12x-dir", type=Path, default=Path("/home/luke/projects/b12x"))
    run.add_argument(
        "--exllamav3-dir",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    run.add_argument("--python", type=Path)
    run.add_argument("--out-root", type=Path, default=Path("out/correctness"))
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--gpus", help="comma-separated GPU indices or UUIDs")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8011)
    run.add_argument("--served-model-name", default="trunc")
    run.add_argument("--prompt", default=DEFAULT_PROMPT)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--logprobs", type=int, default=20)
    run.add_argument("--repeat-requests", type=int, default=2)
    run.add_argument("--logprob-limit", type=float, default=0.1)
    run.add_argument("--boot-timeout", type=int, default=1800)
    run.add_argument("--probe-timeout", type=int, default=600)
    run.add_argument("--shutdown-timeout", type=int, default=120)
    run.add_argument("--max-model-len", type=int, default=2048)
    run.add_argument("--max-num-batched-tokens", type=int, default=512)
    run.add_argument("--max-num-seqs", type=int, default=1)
    run.add_argument("--load-format", default="safetensors")
    run.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    run.add_argument("--kv-cache-memory-bytes", type=int)
    run.add_argument("--graph-mode", choices=("eager", "piecewise"), default="eager")
    run.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--vllm-arg", action="append", default=[], metavar="ARG")
    run.add_argument("--require-kernel-path", choices=KERNEL_PATHS)
    run.set_defaults(func=run_server)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("left")
    comparison.add_argument("right")
    comparison.add_argument("--output", type=Path)
    comparison.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    comparison.add_argument("--absolute-limit", type=float, default=0.1)
    comparison.set_defaults(func=compare)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
