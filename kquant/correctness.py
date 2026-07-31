"""Reproducible Kimi-K3 serving-correctness helpers.

The functions in this module are intentionally independent of vLLM so the
same probe files can be compared across source revisions and environments.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = (
    "The capital of France is Paris. The capital of Germany is"
)
DEFAULT_MAX_TOKENS = 8


def write_json(path: Path, value: Any) -> None:
    """Write stable, human-readable JSON."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo: Path) -> dict[str, Any]:
    """Return a compact identity for a Git worktree."""

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    if not (repo / ".git").exists():
        try:
            run("rev-parse", "--git-dir")
        except (OSError, subprocess.CalledProcessError):
            return {"path": str(repo.resolve()), "git": False}
    status = run("status", "--porcelain=v1")
    diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
        stderr=subprocess.STDOUT,
    )
    return {
        "path": str(repo.resolve()),
        "git": True,
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(status),
        "status": status.splitlines(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def model_identity(model_dir: Path) -> dict[str, Any]:
    """Hash the files that define a checkpoint's tensor map and behavior."""
    out: dict[str, Any] = {"path": str(model_dir.resolve())}
    for name in ("config.json", "model.safetensors.index.json"):
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        out[name] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{model_dir}: invalid or empty weight_map")
    files = sorted(set(weight_map.values()))
    missing = [name for name in files if not (model_dir / name).is_file()]
    out.update(
        {
            "tensor_count": len(weight_map),
            "shard_count": len(files),
            "missing_shards": missing,
        }
    )
    if missing:
        raise FileNotFoundError(
            f"{model_dir}: {len(missing)} indexed shards are missing"
        )
    return out


def _csv_rows(command: list[str]) -> list[list[str]]:
    output = subprocess.check_output(command, text=True).strip()
    if not output:
        return []
    return [[item.strip() for item in line.split(",")] for line in output.splitlines()]


def gpu_inventory() -> list[dict[str, Any]]:
    rows = _csv_rows(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    return [
        {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
        }
        for index, uuid, name, total, used in rows
    ]


def gpu_compute_apps() -> list[dict[str, Any]]:
    rows = _csv_rows(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return [
        {
            "gpu_uuid": uuid,
            "pid": int(pid),
            "process_name": process,
            "used_memory_mib": int(used),
        }
        for uuid, pid, process, used in rows
    ]


def select_gpu_uuids(tp_size: int, requested: str | None = None) -> list[str]:
    inventory = gpu_inventory()
    by_uuid = {gpu["uuid"]: gpu for gpu in inventory}
    if requested:
        values = [value.strip() for value in requested.split(",") if value.strip()]
        selected = []
        for value in values:
            if value in by_uuid:
                selected.append(value)
                continue
            try:
                selected.append(inventory[int(value)]["uuid"])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"unknown GPU selector {value!r}") from exc
    else:
        selected = [gpu["uuid"] for gpu in inventory[:tp_size]]
    if len(selected) != tp_size:
        raise ValueError(
            f"TP{tp_size} requires exactly {tp_size} GPUs, got {len(selected)}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("GPU selection contains duplicates")
    return selected


def assert_gpus_idle(selected: list[str]) -> None:
    occupied = [
        app for app in gpu_compute_apps() if app["gpu_uuid"] in set(selected)
    ]
    if occupied:
        raise RuntimeError(f"selected GPUs have compute processes: {occupied}")


def assert_port_free(host: str, port: int) -> None:
    """Fail if the requested listen address is already owned."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"{host}:{port} is not available") from exc


def package_versions(python: Path) -> dict[str, Any]:
    code = """
import importlib.metadata as md
import json
import platform
names = ("vllm", "torch", "transformers", "flashinfer-python", "sparkinfer",
         "safetensors", "exllamav3")
versions = {}
for name in names:
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"python": platform.python_version(), "packages": versions}))
"""
    output = subprocess.check_output([str(python), "-c", code], text=True)
    return json.loads(output)


def probe_payload(
    *,
    model_name: str,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    logprobs: int = 20,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "logprobs": logprobs,
        "echo": True,
        "return_token_ids": True,
    }


@dataclass(frozen=True)
class ProbeView:
    generated_token_ids: tuple[int, ...] | None
    generated_tokens: tuple[str, ...]
    chosen_logprobs: tuple[float | None, ...]
    text: str


def probe_view(response: dict[str, Any], max_tokens: int) -> ProbeView:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("probe response must contain exactly one choice")
    choice = choices[0]
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise ValueError("probe response is missing logprobs")
    tokens = logprobs.get("tokens")
    chosen = logprobs.get("token_logprobs")
    if not isinstance(tokens, list) or not isinstance(chosen, list):
        raise ValueError("probe response has malformed logprobs")
    token_ids = choice.get("token_ids")
    generated_ids = (
        tuple(int(value) for value in token_ids)
        if isinstance(token_ids, list)
        else None
    )
    return ProbeView(
        generated_token_ids=generated_ids,
        generated_tokens=tuple(str(value) for value in tokens[-max_tokens:]),
        chosen_logprobs=tuple(
            None if value is None else float(value) for value in chosen
        ),
        text=str(choice.get("text", "")),
    )


def compare_probe_responses(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    logprob_limit: float = 0.1,
) -> dict[str, Any]:
    """Compare deterministic output and chosen-token log probabilities."""
    lhs = probe_view(left, max_tokens)
    rhs = probe_view(right, max_tokens)
    if lhs.generated_token_ids is not None and rhs.generated_token_ids is not None:
        same_tokens = lhs.generated_token_ids == rhs.generated_token_ids
        token_identity = "token_ids"
        left_tokens: Any = list(lhs.generated_token_ids)
        right_tokens: Any = list(rhs.generated_token_ids)
    else:
        same_tokens = lhs.generated_tokens == rhs.generated_tokens
        token_identity = "token_strings"
        left_tokens = list(lhs.generated_tokens)
        right_tokens = list(rhs.generated_tokens)

    if len(lhs.chosen_logprobs) != len(rhs.chosen_logprobs):
        deltas: list[float] = []
        same_length = False
    else:
        same_length = True
        deltas = [
            abs(a - b)
            for a, b in zip(lhs.chosen_logprobs, rhs.chosen_logprobs)
            if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
        ]
    max_delta = max(deltas, default=math.inf)
    mean_delta = sum(deltas) / len(deltas) if deltas else math.inf
    passed = same_tokens and same_length and max_delta <= logprob_limit
    return {
        "pass": passed,
        "token_identity": token_identity,
        "same_generated_tokens": same_tokens,
        "left_generated_tokens": left_tokens,
        "right_generated_tokens": right_tokens,
        "same_logprob_length": same_length,
        "compared_logprobs": len(deltas),
        "max_abs_logprob_delta": max_delta,
        "mean_abs_logprob_delta": mean_delta,
        "logprob_limit": logprob_limit,
    }


def parse_env_assignments(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment assignment {value!r}")
        env[key] = item
    return env


def process_snapshot(process_group: int | None = None) -> dict[str, Any]:
    command = ["ps", "-eo", "pid,ppid,pgid,state,etimes,pcpu,pmem,command"]
    rows = subprocess.check_output(command, text=True).splitlines()
    if process_group is not None:
        header, *body = rows
        rows = [
            header,
            *[
                row
                for row in body
                if len(row.split(maxsplit=3)) >= 3
                and row.split(maxsplit=3)[2] == str(process_group)
            ],
        ]
    return {
        "process_group": process_group,
        "ps": rows,
        "compute_apps": gpu_compute_apps(),
    }
