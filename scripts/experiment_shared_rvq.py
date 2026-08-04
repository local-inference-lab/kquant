"""Test shared RVQ and PVQ transfers on held-out interim K3 experts.

The experiment trains codebooks on retained experts from an already packaged
interim EXL3 checkpoint and validates on a different retained expert.  It
never resolves or instantiates the official checkpoint.
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
    normalized_mse,
    pvq_codebook_size,
    pvq_gain_shape,
    train_residual_vq,
)
from kquant.interim_weights import InterimKeepStore, routed_kept_experts

PVQ_POWERS = (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5)


def _rvq_configurations(vector_dimension: int) -> tuple[tuple[int, int], ...]:
    """Fixed-rate additive dictionaries at exactly three bits per weight."""

    total_index_bits = 3 * vector_dimension
    return tuple(
        (total_index_bits // index_bits, 1 << index_bits)
        for index_bits in (12, 8, 6, 4)
        if total_index_bits % index_bits == 0
    )


def _pvq_configurations(vector_dimension: int) -> tuple[tuple[int, int, int], ...]:
    """Pulse/gain splits that total exactly three bits per weight."""

    total_bits = 3 * vector_dimension
    configurations = []
    for pulses in range(1, 65):
        shape_bits = math.ceil(math.log2(pvq_codebook_size(vector_dimension, pulses)))
        gain_bits = total_bits - shape_bits
        if 2 <= gain_bits <= 16:
            configurations.append((pulses, shape_bits, gain_bits))
        if gain_bits < 2:
            break
    return tuple(configurations)


def _vectorize(
    matrix: torch.Tensor, vector_dimension: int, orientation: str
) -> torch.Tensor:
    if orientation == "row":
        ordered = matrix.contiguous()
    elif orientation == "column":
        ordered = matrix.T.contiguous()
    else:
        raise ValueError(f"unsupported vector orientation: {orientation}")
    if ordered.numel() % vector_dimension:
        raise ValueError(
            f"matrix size {ordered.numel()} is not divisible by {vector_dimension}"
        )
    return ordered.reshape(-1, vector_dimension)


def _sample_vectors(vectors: torch.Tensor, count: int, *, seed: int) -> torch.Tensor:
    count = min(count, vectors.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    index = torch.randperm(vectors.shape[0], generator=generator)[:count]
    return vectors[index].float().contiguous()


def _load_samples(
    store: InterimKeepStore,
    *,
    layer: int,
    experts: list[int],
    matrix: str,
    vector_dimension: int,
    orientation: str,
    total_count: int,
    seed: int,
) -> torch.Tensor:
    per_expert = math.ceil(total_count / len(experts))
    chunks = []
    for offset, expert in enumerate(experts):
        weight = store.load_matrix(layer, expert, matrix)
        vectors = _vectorize(weight, vector_dimension, orientation)
        chunks.append(_sample_vectors(vectors, per_expert, seed=seed + 10_000 * offset))
        del weight, vectors
    return torch.cat(chunks)[:total_count].contiguous()


def _rvq_stage_metrics(
    vectors: torch.Tensor,
    reconstruction: torch.Tensor,
    labels: tuple[torch.Tensor, ...],
    codebook_size: int,
) -> dict:
    entropies = [index_entropy(stage, codebook_size) for stage in labels]
    occupancies = [int(torch.unique(stage).numel()) / codebook_size for stage in labels]
    return {
        "nmse": normalized_mse(vectors, reconstruction),
        "stage_index_entropy_bits": entropies,
        "entropy_rate_bpw": sum(entropies) / vectors.shape[1],
        "stage_codebook_occupancy": occupancies,
    }


def _run_rvq(
    training: torch.Tensor,
    validation: torch.Tensor,
    *,
    device: torch.device,
    iterations: int,
    seed: int,
    full_matrix_weights: int,
) -> list[dict]:
    training_device = training.to(device)
    validation_device = validation.to(device)
    results = []
    configurations = _rvq_configurations(training.shape[1])
    for configuration_index, (stages, codebook_size) in enumerate(configurations):
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        started = time.perf_counter()
        model = train_residual_vq(
            training_device,
            stages=stages,
            codebook_size=codebook_size,
            iterations=iterations,
            seed=seed + configuration_index,
        )
        training_reconstruction, training_labels = apply_residual_vq(
            training_device, model
        )
        validation_reconstruction, validation_labels = apply_residual_vq(
            validation_device, model
        )
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        elapsed = time.perf_counter() - started
        codebook_bits = sum(codebook.numel() * 16 for codebook in model.codebooks)
        results.append(
            {
                "stages": stages,
                "codebook_size": codebook_size,
                "vector_dimension": model.vector_dimension,
                "fixed_index_rate_bpw": model.bits_per_weight,
                "codebook_storage_bytes": codebook_bits // 8,
                "rate_including_one_expert_codebook_bpw": (
                    model.bits_per_weight + codebook_bits / full_matrix_weights
                ),
                "rate_including_layer_shared_codebook_bpw": (
                    model.bits_per_weight
                    + codebook_bits / (C.NUM_EXPERTS * full_matrix_weights)
                ),
                "training": _rvq_stage_metrics(
                    training_device,
                    training_reconstruction,
                    training_labels,
                    codebook_size,
                ),
                "held_out": _rvq_stage_metrics(
                    validation_device,
                    validation_reconstruction,
                    validation_labels,
                    codebook_size,
                ),
                "training_and_encoding_seconds": elapsed,
            }
        )
        del model, training_reconstruction, validation_reconstruction
        del training_labels, validation_labels
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def _run_pvq(
    training: torch.Tensor,
    validation: torch.Tensor,
    *,
    device: torch.device,
) -> dict:
    training_device = training.to(device)
    validation_device = validation.to(device)
    candidates = []
    for pulses, shape_bits, gain_bits in _pvq_configurations(training.shape[1]):
        for power in PVQ_POWERS:
            encoded = pvq_gain_shape(
                training_device,
                pulses=pulses,
                power=power,
                gain_bits=gain_bits,
            )
            candidates.append(
                {
                    "pulses": pulses,
                    "shape_bits": shape_bits,
                    "gain_bits": gain_bits,
                    "power": power,
                    "bits_per_weight": encoded.bits_per_weight,
                    "training_nmse": normalized_mse(
                        training_device, encoded.reconstruction
                    ),
                }
            )
    best = min(candidates, key=lambda candidate: candidate["training_nmse"])
    held_out = pvq_gain_shape(
        validation_device,
        pulses=best["pulses"],
        power=best["power"],
        gain_bits=best["gain_bits"],
    )
    return {
        "selection": "minimum training NMSE; held-out expert not consulted",
        "candidates": candidates,
        "best": {
            **best,
            "held_out_nmse": normalized_mse(validation_device, held_out.reconstruction),
            "gain_range_overhead": "two scalars per encoded matrix",
        },
    }


def _load_exl3_anchor(
    path: Path | None, *, layer: int, matrix: str, expert: int
) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    source = payload.get("source", {})
    baseline = payload.get("exl3_baseline")
    if (
        baseline is None
        or source.get("layer") != layer
        or source.get("matrix") != matrix
        or baseline.get("expert") != expert
    ):
        raise ValueError(
            f"EXL3 anchor {path} does not describe layer {layer} "
            f"expert {expert} {matrix}"
        )
    return {"path": str(path.resolve()), **baseline}


def run(args: argparse.Namespace) -> dict:
    store = InterimKeepStore(args.checkpoint)
    ordered, traffic = routed_kept_experts(store, args.capture, args.layer)
    needed = args.train_experts + 1
    if len(ordered) < needed:
        raise ValueError(f"need {needed} retained experts, found {len(ordered)}")
    if args.held_out_expert is None:
        held_out_expert = ordered[0]
    elif args.held_out_expert not in ordered:
        raise ValueError(f"expert {args.layer}.{args.held_out_expert} is not retained")
    else:
        held_out_expert = args.held_out_expert
    training_experts = [expert for expert in ordered if expert != held_out_expert][
        : args.train_experts
    ]
    full_matrix_weights = math.prod(C.EXPERT_SHAPES[args.matrix])
    device = torch.device(args.device)

    orientations = {}
    for orientation_index, orientation in enumerate(args.orientations):
        orientation_seed = args.seed + 1_000_000 * orientation_index
        training = _load_samples(
            store,
            layer=args.layer,
            experts=training_experts,
            matrix=args.matrix,
            vector_dimension=args.vector_dimension,
            orientation=orientation,
            total_count=args.train_vectors,
            seed=orientation_seed,
        )
        held_out_matrix = store.load_matrix(args.layer, held_out_expert, args.matrix)
        held_out_vectors = _sample_vectors(
            _vectorize(held_out_matrix, args.vector_dimension, orientation),
            args.validation_vectors,
            seed=orientation_seed + 500_000,
        )
        orientations[orientation] = {
            "training_vectors": int(training.shape[0]),
            "held_out_vectors": int(held_out_vectors.shape[0]),
            "rvq": _run_rvq(
                training,
                held_out_vectors,
                device=device,
                iterations=args.iterations,
                seed=orientation_seed,
                full_matrix_weights=full_matrix_weights,
            ),
            "pvq": _run_pvq(training, held_out_vectors, device=device),
        }
        del training, held_out_matrix, held_out_vectors
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "kind": "kquant_shared_rvq_experiment",
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
            "vector_dimension": args.vector_dimension,
            "fixed_target_bpw": 3.0,
            "iterations": args.iterations,
            "seed": args.seed,
            "validation": "held-out expert",
            "orientations": orientations,
        },
        "exl3_anchor": _load_exl3_anchor(
            args.exl3_reference,
            layer=args.layer,
            matrix=args.matrix,
            expert=held_out_expert,
        ),
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
    parser.add_argument("--vector-dimension", type=int, default=8)
    parser.add_argument("--train-experts", type=int, default=4)
    parser.add_argument("--held-out-expert", type=int)
    parser.add_argument("--train-vectors", type=int, default=262_144)
    parser.add_argument("--validation-vectors", type=int, default=262_144)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--orientations", nargs="+", choices=("row", "column"), default=("row",)
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exl3-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= C.NUM_MOE_LAYERS:
        parser.error("--layer must be 1..92")
    if (
        min(
            args.vector_dimension,
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
