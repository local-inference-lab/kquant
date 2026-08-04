"""All-expert EXL3 candidate pools and allocation materialization."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from kquant import constants as C
from kquant.dynstats import load_bundle


def _conditional_damage(
    residual_w1: np.ndarray,
    residual_w3: np.ndarray,
    residual_w2: np.ndarray,
    act_in: np.ndarray,
    act_mid: np.ndarray,
) -> np.ndarray:
    """Return expert-conditional diagonal-H damage for one layer."""

    arrays = {
        "residual.w1": np.asarray(residual_w1, dtype=np.float64),
        "residual.w3": np.asarray(residual_w3, dtype=np.float64),
        "residual.w2": np.asarray(residual_w2, dtype=np.float64),
        "act_m2_in": np.asarray(act_in, dtype=np.float64),
        "act_m2_mid": np.asarray(act_mid, dtype=np.float64),
    }
    expected = {
        "residual.w1": (C.NUM_EXPERTS, C.LATENT),
        "residual.w3": (C.NUM_EXPERTS, C.LATENT),
        "residual.w2": (C.NUM_EXPERTS, C.EXPERT_INTER),
        "act_m2_in": (C.NUM_EXPERTS, C.LATENT),
        "act_m2_mid": (C.NUM_EXPERTS, C.EXPERT_INTER),
    }
    for name, value in arrays.items():
        if value.shape != expected[name]:
            raise ValueError(
                f"{name} has shape {value.shape}, expected {expected[name]}"
            )
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must contain finite non-negative values")
    return (
        (arrays["residual.w1"] * arrays["act_m2_in"]).sum(axis=1)
        + (arrays["residual.w3"] * arrays["act_m2_in"]).sum(axis=1)
        + (arrays["residual.w2"] * arrays["act_m2_mid"]).sum(axis=1)
    )


def write_candidate_metrics(
    path: str | Path,
    *,
    layer: int,
    errors: dict[str, float | dict[str, float]],
    residuals: dict[str, torch.Tensor],
    experts: list[int],
) -> Path:
    """Write compact scalar and per-input-channel candidate error metrics."""

    if experts != list(range(C.NUM_EXPERTS)):
        raise ValueError("candidate metrics require all experts in canonical order")
    scalar = {
        "error_proxy": torch.empty((C.NUM_EXPERTS, 3), dtype=torch.float64),
        "error_numerator": torch.empty((C.NUM_EXPERTS, 3), dtype=torch.float64),
        "source_energy": torch.empty((C.NUM_EXPERTS, 3), dtype=torch.float64),
    }
    residual_by_matrix = {
        "residual.w1": torch.empty((C.NUM_EXPERTS, C.LATENT), dtype=torch.float32),
        "residual.w3": torch.empty((C.NUM_EXPERTS, C.LATENT), dtype=torch.float32),
        "residual.w2": torch.empty(
            (C.NUM_EXPERTS, C.EXPERT_INTER), dtype=torch.float32
        ),
    }
    for expert in experts:
        for matrix_idx, matrix in enumerate(C.EXPERT_MATRICES):
            value = errors[f"{layer}.{expert}.{matrix}"]
            if not isinstance(value, dict):
                raise TypeError(
                    f"candidate {layer}.{expert}.{matrix} has no full error metrics"
                )
            scalar["error_proxy"][expert, matrix_idx] = float(value["proxy"])
            scalar["error_numerator"][expert, matrix_idx] = float(value["numerator"])
            scalar["source_energy"][expert, matrix_idx] = float(value["denominator"])
            residual_by_matrix[f"residual.{matrix}"][expert].copy_(
                residuals[f"{expert}.{matrix}"]
            )
    tensors = {**scalar, **residual_by_matrix}
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(tensors, str(tmp))
    tmp.replace(path)
    return path


def load_candidate_damage(candidate_dir: str | Path) -> np.ndarray:
    """Return summed H-weighted demotion error as ``[92, 896]``."""

    candidate_dir = Path(candidate_dir)
    damage = np.full((C.NUM_MOE_LAYERS, C.NUM_EXPERTS), np.nan, dtype=np.float64)
    for layer in C.MOE_LAYERS:
        path = candidate_dir / f"exl3-layer-{layer:05d}.metrics.safetensors"
        metrics = load_file(str(path), device="cpu")
        numerator = metrics["error_numerator"].double().numpy()
        if numerator.shape != (C.NUM_EXPERTS, len(C.EXPERT_MATRICES)):
            raise ValueError(f"candidate metric {path} has shape {numerator.shape}")
        damage[layer - 1] = numerator.sum(axis=1)
    if not np.isfinite(damage).all():
        where = np.argwhere(~np.isfinite(damage))
        raise ValueError(f"candidate metrics contain non-finite cells: {where[:8]}")
    return damage


def score_candidates(
    candidate_dir: str | Path,
    stats_path: str | Path,
) -> np.ndarray:
    """Score the quality loss from demoting each layer/expert to EXL3.

    Dense-Hessian error is conditional on the shared layer activation
    distribution.  Multiplying by measured per-expert gate-square mass restores
    expert traffic/saliency and makes the global keep decision comparable.
    """

    stats = load_bundle(stats_path)
    if stats.manifest.get("model", C.MODEL_ID) != C.MODEL_ID:
        raise ValueError("candidate stats model does not match Kimi-K3")
    if stats.manifest.get("revision", C.REVISION) != C.REVISION:
        raise ValueError(
            "candidate stats revision does not match the source checkpoint"
        )
    gate_sq = np.asarray(stats["gate_sq_sum"], dtype=np.float64)
    damage = np.zeros((C.NUM_MOE_LAYERS, C.NUM_EXPERTS), dtype=np.float64)
    moment_keys = {"act_m2_in", "act_m2_mid"}
    present_moments = moment_keys.intersection(stats.keys())
    if present_moments and present_moments != moment_keys:
        missing = sorted(moment_keys - present_moments)
        raise ValueError(f"activation stats are incomplete; missing {missing}")
    have_conditional = present_moments == moment_keys
    if have_conditional:
        act_in = np.asarray(stats["act_m2_in"], dtype=np.float64)
        act_mid = np.asarray(stats["act_m2_mid"], dtype=np.float64)
        if act_in.shape != (C.NUM_MOE_LAYERS, C.NUM_EXPERTS, C.LATENT):
            raise ValueError(f"act_m2_in has unexpected shape {act_in.shape}")
        if act_mid.shape != (
            C.NUM_MOE_LAYERS,
            C.NUM_EXPERTS,
            C.EXPERT_INTER,
        ):
            raise ValueError(f"act_m2_mid has unexpected shape {act_mid.shape}")
        for layer in C.MOE_LAYERS:
            metrics = load_file(
                str(
                    Path(candidate_dir) / f"exl3-layer-{layer:05d}.metrics.safetensors"
                ),
                device="cpu",
            )
            row = layer - 1
            damage[row] = _conditional_damage(
                metrics["residual.w1"].numpy(),
                metrics["residual.w3"].numpy(),
                metrics["residual.w2"].numpy(),
                act_in[row],
                act_mid[row],
            )
    else:
        damage = load_candidate_damage(candidate_dir)
    if gate_sq.shape != damage.shape:
        raise ValueError(
            f"gate_sq_sum shape {gate_sq.shape} does not match {damage.shape}"
        )
    if not np.isfinite(gate_sq).all() or np.any(gate_sq < 0):
        raise ValueError("gate_sq_sum must contain finite non-negative values")
    scores = damage * gate_sq
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("candidate scores must contain finite non-negative values")
    return scores


def choose_allocation(scores: np.ndarray, keep_frac: float) -> dict:
    if scores.shape != (C.NUM_MOE_LAYERS, C.NUM_EXPERTS):
        raise ValueError(f"unexpected EXL3 score shape {scores.shape}")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("EXL3 scores must contain finite non-negative values")
    if not 0.0 <= keep_frac <= 1.0:
        raise ValueError("keep_frac must be in [0, 1]")
    count = scores.size
    n_keep = round(float(keep_frac) * count)
    keep_mask = np.zeros(count, dtype=bool)
    if n_keep:
        keep_idx = np.argpartition(scores.reshape(-1), -n_keep)[-n_keep:]
        keep_mask[keep_idx] = True
    keep_mask = keep_mask.reshape(scores.shape)
    return {
        str(layer): {
            "keep": np.flatnonzero(keep_mask[layer - 1]).tolist(),
            "exl3": np.flatnonzero(~keep_mask[layer - 1]).tolist(),
        }
        for layer in C.MOE_LAYERS
    }


def write_candidate_allocation(
    dest: str | Path,
    allocation: dict,
    *,
    keep_frac: float,
    stats_path: str | Path,
    candidate_dir: str | Path,
) -> Path:
    dest = Path(dest)
    n_keep = sum(len(value["keep"]) for value in allocation.values())
    document = {
        "meta": {
            "keep_frac": float(keep_frac),
            "n_keep": n_keep,
            "n_exl3": C.NUM_MOE_LAYERS * C.NUM_EXPERTS - n_keep,
            "traffic": "measured gate_sq_sum",
            "damage": (
                "expert-conditional diagonal-H residual energy when activation "
                "moments are available; otherwise shared full-H numerator"
            ),
            "stats": str(Path(stats_path).resolve()),
            "candidate_pool": str(Path(candidate_dir).resolve()),
        },
        "layers": allocation,
    }
    path = dest / "allocation-exl3.json"
    tmp = dest / ".allocation-exl3.json.tmp"
    tmp.write_text(json.dumps(document))
    tmp.replace(path)
    return path


def _expert_from_tensor_name(name: str) -> int:
    marker = ".experts."
    try:
        return int(name.split(marker, 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"cannot parse expert id from candidate tensor {name!r}"
        ) from exc


def materialize_allocation(
    candidate_dir: str | Path,
    dest: str | Path,
    allocation: dict,
    *,
    layers: list[int] | None = None,
) -> None:
    """Filter an all-expert pool into serveable selected-EXL3 layer shards."""

    candidate_dir, dest = Path(candidate_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    selected_layers = list(C.MOE_LAYERS) if layers is None else layers
    for layer in selected_layers:
        selected = set(map(int, allocation[str(layer)]["exl3"]))
        source = candidate_dir / f"exl3-layer-{layer:05d}.safetensors"
        target = dest / source.name
        tensors = {}
        with safe_open(source, framework="pt", device="cpu") as handle:
            for name in handle:
                if _expert_from_tensor_name(name) in selected:
                    tensors[name] = handle.get_tensor(name)
        expected = len(selected) * len(C.EXPERT_MATRICES) * 3
        if len(tensors) != expected:
            raise ValueError(
                f"layer {layer}: selected {len(tensors)} candidate tensors, expected {expected}"
            )
        tmp = dest / f".{target.name}.{os.getpid()}.tmp"
        save_file(tensors, str(tmp))
        tmp.replace(target)
