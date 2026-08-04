#!/usr/bin/env python3
"""Capture and compare one TP12 Kimi-K3 model on the pinned KLD suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from kquant.correctness import (
    assert_gpus_idle,
    assert_port_free,
    git_state,
    gpu_inventory,
    model_identity,
    package_versions,
    process_snapshot,
    select_gpu_uuids,
    sha256_file,
)
from kquant.kernel_audit import KERNEL_PATHS, audit_kernel_path
from kquant.kld_capture import (
    atomic_write_json,
    load_json_object,
    prepare_suite_workspace,
    validate_finalized_manifest,
    validate_reference_root,
)
from kquant.kld_gate import (
    CANONICAL_EVALUATION_TOOLS_COMMIT,
    CANONICAL_REFERENCE_DATASET,
    CANONICAL_REFERENCE_DATASET_REVISION,
    CANONICAL_SUITE_TOKEN_HASH,
    validate_comparison_report,
    validate_suite_manifest,
)


TP_SIZE = 12
EXPECTED_VOCAB = 163_840
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _http_health(url: str, *, timeout: float = 3.0) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"health endpoint returned HTTP {response.status}")


def _wait_ready(
    process: subprocess.Popen[bytes],
    *,
    health_url: str,
    timeout: float,
    log_path: Path,
) -> float:
    started = time.monotonic()
    deadline = started + timeout
    last_report = 0.0
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            tail = log_path.read_text(errors="replace").splitlines()[-100:]
            raise RuntimeError(
                f"server exited during boot with status {code}\n" + "\n".join(tail)
            )
        try:
            _http_health(health_url)
            return time.monotonic() - started
        except (OSError, urllib.error.URLError, RuntimeError):
            pass
        now = time.monotonic()
        if now - last_report >= 30:
            print(
                f"booting TP12 KLD server: {int(now - started)}s elapsed, "
                f"pid={process.pid}",
                flush=True,
            )
            last_report = now
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy in {timeout:.0f}s")


def _stop_process_group(
    process: subprocess.Popen[bytes], *, timeout: float
) -> int | None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)
    return process.returncode


def _run_tool(command: list[str], *, log: TextIO) -> None:
    print("+ " + " ".join(command), file=log, flush=True)
    subprocess.run(
        command,
        check=True,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )


def build_server_environment(
    *,
    model: Path,
    served_model_name: str,
    capture_dir: Path,
    host: str,
    port: int,
    selected_gpus: list[str],
) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "K3_KQUANT_CAPTURE_DIR",
        "VLLM_KQUANT_CAPTURE_DIR",
        "VLLM_KQUANT_FINALIZE_FILE",
    ):
        env.pop(name, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(selected_gpus),
            "K3_MODEL_DIR": str(model),
            "K3_SERVED_MODEL_NAME": served_model_name,
            "K3_HOST": host,
            "K3_PORT": str(port),
            "K3_DSPARK": "0",
            "K3_KLD_CAPTURE_DIR": str(capture_dir),
            "K3_MAX_NUM_BATCHED_TOKENS": "256",
            "K3_MAX_NUM_SEQS": "1",
            # Emit one post-start repeat-check record from the actual MoE
            # execution path.  The kernel audit must be backed by runtime
            # evidence rather than merely by checkpoint-loader messages.
            "B12X_MOE_REPEAT_CHECK": "1",
            "B12X_MOE_REPEAT_CHECK_AFTER_ENGINE_START": "1",
            "B12X_MOE_REPEAT_CHECK_MAX_REPORTS": "1",
        }
    )
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/data/kld/kimi-k3-reference"),
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=Path(
            "/home/luke/projects/vllm/serve-kimi-k3-exl3-3p09-tp12.sh"
        ),
    )
    parser.add_argument(
        "--vllm-python",
        type=Path,
        default=Path("/home/luke/projects/vllm/.venv/bin/python"),
    )
    parser.add_argument("--vllm-dir", type=Path, default=Path("/home/luke/projects/vllm"))
    parser.add_argument("--b12x-dir", type=Path, default=Path("/home/luke/projects/b12x"))
    parser.add_argument(
        "--exllamav3-dir", type=Path, default=Path("/home/luke/projects/exllamav3")
    )
    parser.add_argument("--gpus", help="comma-separated GPU indices or UUIDs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="Kimi-K3")
    parser.add_argument("--boot-timeout", type=float, default=3600.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--shutdown-timeout", type=float, default=180.0)
    parser.add_argument(
        "--require-kernel-path", choices=KERNEL_PATHS, required=True
    )
    parser.add_argument(
        "--rehash-reference",
        action="store_true",
        help="Rehash all 39.98 GiB of canonical reference payloads before serving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not RUN_NAME_RE.fullmatch(args.run_name):
        raise ValueError("run-name may contain only letters, digits, '.', '_', and '-'")
    model = args.model.resolve()
    reference_root = args.reference_root.resolve()
    output_dir = args.output_dir.resolve()
    launcher = args.launcher.resolve()
    python = args.vllm_python.absolute()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse KLD output directory: {output_dir}")
    if not launcher.is_file():
        raise FileNotFoundError(launcher)
    if not python.is_file():
        raise FileNotFoundError(python)

    reference = validate_reference_root(
        reference_root, rehash_payloads=args.rehash_reference
    )
    model_info = model_identity(model)
    selected = select_gpu_uuids(TP_SIZE, args.gpus)
    assert_gpus_idle(selected)
    assert_port_free(args.host, args.port)

    output_dir.mkdir(parents=True, exist_ok=False)
    suite_dir = prepare_suite_workspace(reference_root, output_dir / "suite")
    capture_dir = output_dir / "capture-chunks"
    capture_dir.mkdir()
    finalized_dir = output_dir / "ref"
    runtime_path = output_dir / "runtime.json"
    result_path = output_dir / "result.json"
    server_log_path = output_dir / "server.log"
    tools_log_path = output_dir / "tools.log"
    comparison_path = output_dir / "kld-vs-full-mxfp4.json"

    repo_roots = {
        "kquant": Path(__file__).resolve().parents[1],
        "vllm": args.vllm_dir.resolve(),
        "b12x": args.b12x_dir.resolve(),
        "exllamav3": args.exllamav3_dir.resolve(),
    }
    runtime: dict[str, Any] = {
        "schema_version": 1,
        "kind": "kquant_kimi_k3_tp12_kld_capture_runtime",
        "run_name": args.run_name,
        "started_at": datetime.now().astimezone().isoformat(),
        "tp_size": TP_SIZE,
        "model": model_info,
        "reference": reference,
        "reference_dataset": CANONICAL_REFERENCE_DATASET,
        "reference_dataset_revision": CANONICAL_REFERENCE_DATASET_REVISION,
        "evaluation_tools_commit": CANONICAL_EVALUATION_TOOLS_COMMIT,
        "suite_token_hash_sha256": CANONICAL_SUITE_TOKEN_HASH,
        "repositories": {name: git_state(path) for name, path in repo_roots.items()},
        "runtime": package_versions(python),
        "gpu_inventory": gpu_inventory(),
        "selected_gpu_uuids": selected,
        "launcher": {"path": str(launcher), "sha256": sha256_file(launcher)},
        "served_model_name": args.served_model_name,
        "endpoint": f"http://{args.host}:{args.port}",
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "kquant_kimi_k3_tp12_kld_capture",
        "run_name": args.run_name,
        "output_dir": str(output_dir),
        "status": "starting",
    }
    atomic_write_json(runtime_path, runtime)
    atomic_write_json(result_path, result)

    env = build_server_environment(
        model=model,
        served_model_name=args.served_model_name,
        capture_dir=capture_dir,
        host=args.host,
        port=args.port,
        selected_gpus=selected,
    )
    recorded_env_names = (
        "CUDA_VISIBLE_DEVICES",
        "K3_MODEL_DIR",
        "K3_SERVED_MODEL_NAME",
        "K3_HOST",
        "K3_PORT",
        "K3_DSPARK",
        "K3_KLD_CAPTURE_DIR",
        "K3_MAX_NUM_BATCHED_TOKENS",
        "K3_MAX_NUM_SEQS",
        "B12X_MOE_FORCE_A16",
        "B12X_MOE_REPEAT_CHECK",
        "B12X_MOE_REPEAT_CHECK_AFTER_ENGINE_START",
        "B12X_MOE_REPEAT_CHECK_MAX_REPORTS",
        "VLLM_KQUANT_TRELLIS_W4A8",
    )
    runtime["server_environment"] = {
        name: env[name] for name in recorded_env_names if name in env
    }
    runtime["server_command"] = [str(launcher)]
    atomic_write_json(runtime_path, runtime)

    process: subprocess.Popen[bytes] | None = None
    server_log: TextIO | None = None
    try:
        server_log = server_log_path.open("x", encoding="utf-8")
        process = subprocess.Popen(
            [str(launcher)],
            cwd=args.vllm_dir.resolve(),
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        result["server_pid"] = process.pid
        result["status"] = "booting"
        atomic_write_json(result_path, result)
        runtime["ready_after_seconds"] = _wait_ready(
            process,
            health_url=f"http://{args.host}:{args.port}/health",
            timeout=args.boot_timeout,
            log_path=server_log_path,
        )
        runtime["ready_at"] = datetime.now().astimezone().isoformat()
        atomic_write_json(runtime_path, runtime)

        result["status"] = "capturing"
        atomic_write_json(result_path, result)
        tools = reference_root / "tools"
        with tools_log_path.open("x", encoding="utf-8") as tools_log:
            _run_tool(
                [
                    str(python),
                    str(tools / "capture-kimi-k3-kld-suite.py"),
                    "--url",
                    f"http://{args.host}:{args.port}/v1/completions",
                    "--model",
                    args.served_model_name,
                    "--suite-dir",
                    str(suite_dir),
                    "--capture-dir",
                    str(capture_dir),
                    "--run-name",
                    args.run_name,
                    "--timeout",
                    str(args.request_timeout),
                ],
                log=tools_log,
            )
        runtime["capture_finished_at"] = datetime.now().astimezone().isoformat()
        capture_record = suite_dir / f"capture-{args.run_name}.json"
        runtime["capture_record"] = {
            "path": str(capture_record),
            "sha256": sha256_file(capture_record),
        }

        result["status"] = "stopping-server"
        atomic_write_json(result_path, result)
        result["server_exit_code"] = _stop_process_group(
            process, timeout=args.shutdown_timeout
        )
        process = None
        server_log.close()
        server_log = None
        kernel_audit = audit_kernel_path(
            server_log_path.read_text(errors="replace"), args.require_kernel_path
        )
        atomic_write_json(output_dir / "kernel-audit.json", kernel_audit)
        result["kernel_audit"] = kernel_audit
        if not kernel_audit["pass"]:
            raise RuntimeError(
                "serving log did not prove the required kernel path: "
                f"{kernel_audit['missing_evidence']}"
            )
        runtime["kernel_audit"] = kernel_audit
        runtime["server_log_sha256"] = sha256_file(server_log_path)
        atomic_write_json(runtime_path, runtime)

        result["status"] = "finalizing"
        atomic_write_json(result_path, result)
        with tools_log_path.open("a", encoding="utf-8") as tools_log:
            _run_tool(
                [
                    str(python),
                    str(tools / "finalize-kimi-k3-kld-suite.py"),
                    "--suite-dir",
                    str(suite_dir),
                    "--capture-dir",
                    str(capture_dir),
                    "--run-name",
                    args.run_name,
                    "--output-dir",
                    str(finalized_dir),
                    "--expected-vocab",
                    str(EXPECTED_VOCAB),
                    "--runtime-manifest",
                    str(runtime_path),
                ],
                log=tools_log,
            )
            _run_tool(
                [
                    str(python),
                    str(tools / "compare-kimi-k3-kld-suite.py"),
                    "--reference-dir",
                    str(reference_root / "ref"),
                    "--candidate-dir",
                    str(finalized_dir),
                    "--suite-manifest",
                    str(reference_root / "suite-manifest.json"),
                    "--output",
                    str(comparison_path),
                ],
                log=tools_log,
            )

        suite_hash, windows = validate_suite_manifest(
            load_json_object(reference_root / "suite-manifest.json"),
            expected_suite_hash=CANONICAL_SUITE_TOKEN_HASH,
        )
        records = validate_finalized_manifest(
            finalized_dir / "manifest.json",
            suite_hash=suite_hash,
            windows=windows,
            label=args.run_name,
        )
        comparison = load_json_object(comparison_path)
        validate_comparison_report(
            comparison, label=args.run_name, expected_windows=windows
        )
        result.update(
            {
                "status": "passed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "windows": len(records),
                "positions": int(comparison["positions"]),
                "mean_kl_reference_to_candidate": float(
                    comparison["kl_reference_to_candidate"]["mean"]
                ),
                "finalized_manifest": {
                    "path": str(finalized_dir / "manifest.json"),
                    "sha256": sha256_file(finalized_dir / "manifest.json"),
                },
                "comparison": {
                    "path": str(comparison_path),
                    "sha256": sha256_file(comparison_path),
                },
            }
        )
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["diagnostics"] = process_snapshot(
            process.pid if process is not None else None
        )
        print(result["error"], file=sys.stderr, flush=True)
    finally:
        if process is not None:
            result["server_exit_code"] = _stop_process_group(
                process, timeout=args.shutdown_timeout
            )
        if server_log is not None:
            server_log.close()
        result.setdefault("finished_at", datetime.now().astimezone().isoformat())
        atomic_write_json(result_path, result)
    print(output_dir, flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
