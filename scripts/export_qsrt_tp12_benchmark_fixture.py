#!/usr/bin/env python3
"""Export natural TP12 routes and selected pair modes for the B12X gate.

The fixture contains no model weights or activations. It binds one completed
candidate layer and logical TP12 rank to naturally captured top-k routes,
applied gate weights, and the exact per-expert P24/P33 mode tables. B12X can
then time identical route batches against a uniform-P33 control without
depending on kquant's Python package or capture reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from kquant import constants as C
from kquant.capture import load_layer_samples
from kquant.qsrt import TP_SIZE, tp12_logical_pair_index
from scripts.summarize_qsrt_candidate_pool import summarize_candidate_pool


KIND = "kquant_kimi_k3_qsrt_tp12_benchmark_fixture"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_rows(count: int, limit: int, seed: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("capture layer has no route rows")
    if limit <= 0:
        raise ValueError("row limit must be positive")
    take = min(count, limit)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(count, generator=generator)[:take].sort().values


def build_fixture_tensors(
    *,
    layer: int,
    rank: int,
    modes_r13: torch.Tensor,
    modes_r2: torch.Tensor,
    routes: torch.Tensor,
    gates: torch.Tensor,
    observations: torch.Tensor,
    row_limit: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if layer not in C.MOE_LAYERS:
        raise ValueError("layer must be in 1..92")
    if not 0 <= rank < TP_SIZE:
        raise ValueError(f"rank must be in 0..{TP_SIZE - 1}")
    for name, modes in (("r13", modes_r13), ("r2", modes_r2)):
        if modes.dtype != torch.uint8 or tuple(modes.shape) != (C.NUM_EXPERTS,):
            raise ValueError(f"selected {name} modes have the wrong contract")
    if routes.ndim != 2 or tuple(routes.shape[1:]) != (C.TOP_K,):
        raise ValueError("captured routes must have shape [rows, top_k]")
    if gates.shape != routes.shape:
        raise ValueError("captured gates must have the same shape as routes")
    if observations.ndim != 1 or observations.numel() != routes.shape[0]:
        raise ValueError("captured observations do not align with routes")
    if routes.numel() and (
        int(routes.min()) < 0 or int(routes.max()) >= C.NUM_EXPERTS
    ):
        raise ValueError("captured routes contain an invalid expert ID")
    if not bool(torch.isfinite(gates).all()):
        raise ValueError("captured gates contain non-finite values")

    logical_pairs = torch.tensor(
        [
            tp12_logical_pair_index(layer, expert, rank)
            for expert in range(C.NUM_EXPERTS)
        ],
        dtype=torch.uint8,
    )
    fc1_pair_modes = (modes_r13 > logical_pairs).to(torch.uint8)
    fc2_pair_modes = (modes_r2 > logical_pairs).to(torch.uint8)
    indices = _selected_rows(int(routes.shape[0]), row_limit, seed)
    selected_routes = routes.index_select(0, indices).to(torch.int32).contiguous()
    selected_gates = gates.index_select(0, indices).to(torch.float32).contiguous()
    selected_observations = (
        observations.index_select(0, indices).to(torch.int64).contiguous()
    )
    routed_fc1_pair_modes = fc1_pair_modes[selected_routes.to(torch.int64)]
    routed_fc2_pair_modes = fc2_pair_modes[selected_routes.to(torch.int64)]
    routed_any_pair_modes = routed_fc1_pair_modes.bool() | routed_fc2_pair_modes.bool()
    tensors = {
        "fc1_pair_modes": fc1_pair_modes.contiguous(),
        "fc2_pair_modes": fc2_pair_modes.contiguous(),
        "topk_ids": selected_routes,
        "topk_weights": selected_gates,
        "observation_ids": selected_observations,
        "source_row_indices": indices.to(torch.int64),
    }
    summary: dict[str, object] = {
        "layer": layer,
        "rank": rank,
        "captured_rows_available": int(routes.shape[0]),
        "fixture_rows": int(indices.numel()),
        "p24_experts": int((fc1_pair_modes.bool() | fc2_pair_modes.bool()).sum()),
        "p24_expert_fraction": float(
            (fc1_pair_modes.bool() | fc2_pair_modes.bool()).float().mean()
        ),
        "fc1_p24_experts": int(fc1_pair_modes.sum()),
        "fc2_p24_experts": int(fc2_pair_modes.sum()),
        "fc1_p24_route_executions": int(routed_fc1_pair_modes.sum()),
        "fc2_p24_route_executions": int(routed_fc2_pair_modes.sum()),
        "fc1_p24_route_fraction": float(routed_fc1_pair_modes.float().mean()),
        "fc2_p24_route_fraction": float(routed_fc2_pair_modes.float().mean()),
        "p24_route_executions": int(routed_any_pair_modes.sum()),
        "p24_route_fraction": float(routed_any_pair_modes.float().mean()),
        "token_rows_touching_p24": int(routed_any_pair_modes.any(dim=1).sum()),
        "token_hit_fraction": float(
            routed_any_pair_modes.any(dim=1).float().mean()
        ),
        "row_selection": "seeded_without_replacement_sorted_source_order",
        "row_selection_seed": seed,
    }
    return tensors, summary


def export_fixture(
    candidate_pool: str | Path,
    capture: str | Path,
    output: str | Path,
    *,
    layer: int,
    rank: int,
    row_limit: int,
    seed: int,
) -> dict[str, object]:
    candidate_pool = Path(candidate_pool).resolve()
    capture = Path(capture).resolve()
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    pool_summary = summarize_candidate_pool(candidate_pool)
    completed = {
        int(value) for value in pool_summary["progress"]["completed_layers"]
    }
    if layer not in completed:
        raise ValueError(f"candidate layer {layer} is not atomically complete")

    metrics_path = (
        candidate_pool
        / "candidates"
        / f"qsrt-layer-{layer:05d}.metrics.safetensors"
    )
    selection_path = (
        candidate_pool
        / "candidates"
        / f"qsrt-layer-{layer:05d}.selection.json"
    )
    metrics = load_file(str(metrics_path), device="cpu")
    samples = load_layer_samples(capture, layer - 1)
    tensors, route_summary = build_fixture_tensors(
        layer=layer,
        rank=rank,
        modes_r13=metrics["selected_r13"],
        modes_r2=metrics["selected_r2"],
        routes=samples.input_experts,
        gates=samples.input_gates,
        observations=samples.input_observations,
        row_limit=row_limit,
        seed=seed,
    )
    manifest: dict[str, object] = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "candidate_pool": str(candidate_pool),
        "candidate_pool_sealed": bool(pool_summary["sealed"]),
        "candidate_logical_trellis_schema": pool_summary[
            "logical_trellis_schema"
        ],
        "candidate_manifest_sha256": _sha256(
            candidate_pool / "qsrt-candidate-manifest.json"
        ),
        "candidate_layer_metrics_sha256": _sha256(metrics_path),
        "candidate_layer_selection_sha256": _sha256(selection_path),
        "capture": str(capture),
        "capture_manifest_sha256": _sha256(capture / "manifest.json"),
        "capture_used_for_performance_replay_only": True,
        "mode_selection_or_allocation_performed": False,
        "tp_size": TP_SIZE,
        "num_experts": C.NUM_EXPERTS,
        "top_k": C.TOP_K,
        "hidden_size": C.LATENT,
        "local_intermediate_size": C.EXPERT_INTER // TP_SIZE,
        **route_summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        save_file(tensors, str(temporary), metadata={"manifest": json.dumps(manifest)})
        temporary.replace(output)
        descriptor = os.open(
            output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    manifest["fixture"] = str(output)
    manifest["fixture_sha256"] = _sha256(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_fixture(
        args.candidate_pool,
        args.capture,
        args.output,
        layer=args.layer,
        rank=args.rank,
        row_limit=args.rows,
        seed=args.seed,
    )
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
