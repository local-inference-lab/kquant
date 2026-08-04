"""Test codec-style channel gains plus a shared 4-D weight palette.

Every source tensor comes from retained MXFP4 experts inside the packaged
interim EXL3 checkpoint.  The official model is neither resolved nor loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from kquant import constants as C
from kquant.codec_research import (
    apply_residual_vq,
    index_entropy,
    normalize_matrix_gains,
    normalized_mse,
    train_residual_vq,
)
from kquant.interim_weights import InterimKeepStore, routed_kept_experts


def _vectorize(matrix: torch.Tensor, orientation: str) -> torch.Tensor:
    ordered = matrix.contiguous() if orientation == "row" else matrix.T.contiguous()
    return ordered.reshape(-1, 4)


def _sample_index(length: int, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(length, generator=generator)[: min(length, count)]


def _gain_samples(
    matrix: torch.Tensor,
    *,
    mode: str,
    orientation: str,
    count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    normalized, gains = normalize_matrix_gains(
        matrix, mode=mode, vector_orientation=orientation
    )
    factor = gains.row[:, None] * gains.column[None, :]
    normalized_vectors = _vectorize(normalized, orientation)
    index = _sample_index(normalized_vectors.shape[0], count, seed)
    metadata = {
        "stored_gain_bits": gains.stored_bits,
        "row_gain_min": float(gains.row.min()),
        "row_gain_max": float(gains.row.max()),
        "column_gain_min": float(gains.column.min()),
        "column_gain_max": float(gains.column.max()),
    }
    return (
        normalized_vectors[index].float().contiguous(),
        _vectorize(matrix, orientation)[index].float().contiguous(),
        _vectorize(factor, orientation)[index].float().contiguous(),
        metadata,
    )


def _training_samples(
    store: InterimKeepStore,
    *,
    layer: int,
    experts: list[int],
    matrix_name: str,
    mode: str,
    orientation: str,
    count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_expert = math.ceil(count / len(experts))
    normalized_chunks = []
    source_chunks = []
    factor_chunks = []
    for offset, expert in enumerate(experts):
        matrix = store.load_matrix(layer, expert, matrix_name)
        normalized, source, factor, _ = _gain_samples(
            matrix,
            mode=mode,
            orientation=orientation,
            count=per_expert,
            seed=seed + offset * 10_000,
        )
        normalized_chunks.append(normalized)
        source_chunks.append(source)
        factor_chunks.append(factor)
        del matrix
    return tuple(
        torch.cat(chunks)[:count].contiguous()
        for chunks in (normalized_chunks, source_chunks, factor_chunks)
    )


def _score(
    normalized: torch.Tensor,
    source: torch.Tensor,
    factor: torch.Tensor,
    model,
) -> dict:
    reconstruction, labels = apply_residual_vq(normalized, model)
    original_reconstruction = reconstruction.float() * factor
    entropy = index_entropy(labels[0], 4096)
    return {
        "original_coordinate_nmse": normalized_mse(source, original_reconstruction),
        "normalized_coordinate_nmse": normalized_mse(normalized, reconstruction),
        "index_entropy_bits": entropy,
        "entropy_rate_bpw": entropy / 4,
        "codebook_occupancy": int(torch.unique(labels[0]).numel()) / 4096,
    }


def _run_mode(
    store: InterimKeepStore,
    *,
    args: argparse.Namespace,
    training_experts: list[int],
    held_out_expert: int,
    mode: str,
    orientation: str,
    seed: int,
    device: torch.device,
) -> dict:
    training_normalized, training_source, training_factor = _training_samples(
        store,
        layer=args.layer,
        experts=training_experts,
        matrix_name=args.matrix,
        mode=mode,
        orientation=orientation,
        count=args.train_vectors,
        seed=seed,
    )
    held_out_matrix = store.load_matrix(args.layer, held_out_expert, args.matrix)
    held_normalized, held_source, held_factor, gain_metadata = _gain_samples(
        held_out_matrix,
        mode=mode,
        orientation=orientation,
        count=args.validation_vectors,
        seed=seed + 500_000,
    )
    tensors = [
        training_normalized,
        training_source,
        training_factor,
        held_normalized,
        held_source,
        held_factor,
    ]
    tensors = [tensor.to(device) for tensor in tensors]
    (
        training_normalized,
        training_source,
        training_factor,
        held_normalized,
        held_source,
        held_factor,
    ) = tensors
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    model = train_residual_vq(
        training_normalized,
        stages=1,
        codebook_size=4096,
        iterations=args.iterations,
        seed=seed,
    )
    training_metrics = _score(
        training_normalized, training_source, training_factor, model
    )
    held_out_metrics = _score(held_normalized, held_source, held_factor, model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    full_matrix_weights = held_out_matrix.numel()
    codebook_bits = sum(codebook.numel() * 16 for codebook in model.codebooks)
    gain_bpw = gain_metadata["stored_gain_bits"] / full_matrix_weights
    del held_out_matrix, tensors, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "gain_mode": mode,
        "vector_orientation": orientation,
        "vector_dimension": 4,
        "codebook_size": 4096,
        "index_rate_bpw": 3.0,
        "gain_rate_bpw": gain_bpw,
        "layer_shared_codebook_rate_bpw": codebook_bits
        / (C.NUM_EXPERTS * full_matrix_weights),
        "total_rate_bpw": 3.0
        + gain_bpw
        + codebook_bits / (C.NUM_EXPERTS * full_matrix_weights),
        "gain_metadata": gain_metadata,
        "training": training_metrics,
        "held_out": held_out_metrics,
        "training_and_encoding_seconds": elapsed,
    }


def _load_exl3_anchor(path: Path, expert: int, layer: int, matrix: str) -> dict:
    payload = json.loads(path.read_text())
    baseline = payload.get("exl3_baseline")
    source = payload.get("source", {})
    if (
        baseline is None
        or baseline.get("expert") != expert
        or source.get("layer") != layer
        or source.get("matrix") != matrix
    ):
        raise ValueError(f"EXL3 reference {path} does not match this experiment")
    return {"path": str(path.resolve()), **baseline}


def run(args: argparse.Namespace) -> dict:
    store = InterimKeepStore(args.checkpoint)
    ordered, traffic = routed_kept_experts(store, args.capture, args.layer)
    held_out_expert = ordered[0]
    training_experts = ordered[1 : args.train_experts + 1]
    if len(training_experts) != args.train_experts:
        raise ValueError("not enough retained experts for held-out validation")
    device = torch.device(args.device)
    results = []
    for orientation_index, orientation in enumerate(args.orientations):
        for mode_index, mode in enumerate(args.gain_modes):
            results.append(
                _run_mode(
                    store,
                    args=args,
                    training_experts=training_experts,
                    held_out_expert=held_out_expert,
                    mode=mode,
                    orientation=orientation,
                    seed=args.seed
                    + 1_000_000 * orientation_index
                    + 10_000 * mode_index,
                    device=device,
                )
            )
    rows, columns = C.EXPERT_SHAPES[args.matrix]
    exl3_side_rate = 16 * (rows + columns) / (rows * columns)
    return {
        "kind": "kquant_gain_vq_experiment",
        "schema_version": 1,
        "source": {
            "checkpoint": str(store.root),
            "constraint": "interim EXL3 hybrid only; official model not loaded",
            "capture": str(args.capture.resolve()),
            "layer": args.layer,
            "matrix": args.matrix,
            "held_out_expert": held_out_expert,
            "training_experts": training_experts,
            "captured_gate_square_mass": {
                str(expert): traffic.get(expert, 0.0)
                for expert in [held_out_expert, *training_experts]
            },
        },
        "experiment": {
            "training_vectors": args.train_vectors,
            "validation_vectors": args.validation_vectors,
            "iterations": args.iterations,
            "seed": args.seed,
            "validation": "held-out expert",
            "results": results,
        },
        "exl3_anchor": _load_exl3_anchor(
            args.exl3_reference, held_out_expert, args.layer, args.matrix
        ),
        "exl3_nominal_storage": {
            "trellis_rate_bpw": 3.0,
            "fp16_row_and_column_scale_rate_bpw": exl3_side_rate,
            "total_rate_bpw_excluding_container": 3.0 + exl3_side_rate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/models/Kimi-K3-EXL3-3p09-serve"),
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=Path("/data/kquant/k3-route-aware-pilot-v2-exl3-r4.kqcapture"),
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--matrix", choices=C.EXPERT_MATRICES, default="w1")
    parser.add_argument("--train-experts", type=int, default=4)
    parser.add_argument("--train-vectors", type=int, default=1_048_576)
    parser.add_argument("--validation-vectors", type=int, default=1_048_576)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--orientations", nargs="+", choices=("row", "column"), default=("column",)
    )
    parser.add_argument(
        "--gain-modes",
        nargs="+",
        choices=("none", "aligned", "cross", "separable"),
        default=("none", "aligned", "cross", "separable"),
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exl3-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= C.NUM_MOE_LAYERS:
        parser.error("--layer must be 1..92")
    if (
        min(
            args.train_experts,
            args.train_vectors,
            args.validation_vectors,
            args.iterations,
        )
        <= 0
    ):
        parser.error("experiment sizes must be positive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
