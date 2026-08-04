"""Test a video-codec-style sparse enhancement layer over shared weight VQ.

A 12-bit, 4-weight palette is the base layer.  The encoder flags high-error
16x16 tiles and adds a tiny residual palette only in those tiles, analogous to
block mode maps and enhancement coefficients in image/video codecs.  All
weights come from retained experts in the interim EXL3 checkpoint; the
official checkpoint is never resolved or loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from kquant import constants as C
from kquant.capture import load_layer_samples
from kquant.codec_research import (
    ResidualVQModel,
    apply_residual_vq,
    index_entropy,
    train_residual_vq,
)
from kquant.interim_weights import InterimKeepStore, routed_kept_experts

TILE_SIZE = 16
VECTOR_DIMENSION = 4


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(
        1.0 - probability
    )


def _vectorize(matrix: torch.Tensor, orientation: str) -> torch.Tensor:
    if orientation == "row":
        ordered = matrix.contiguous()
    elif orientation == "column":
        ordered = matrix.T.contiguous()
    else:
        raise ValueError(f"unsupported vector orientation: {orientation}")
    return ordered.reshape(-1, VECTOR_DIMENSION)


def _unvectorize(
    vectors: torch.Tensor, shape: tuple[int, int], orientation: str
) -> torch.Tensor:
    if orientation == "row":
        return vectors.reshape(shape)
    if orientation == "column":
        return vectors.reshape(shape[1], shape[0]).T.contiguous()
    raise ValueError(f"unsupported vector orientation: {orientation}")


def _tile_vectors(tiles: torch.Tensor, orientation: str) -> torch.Tensor:
    if orientation == "column":
        tiles = tiles.transpose(-1, -2).contiguous()
    elif orientation != "row":
        raise ValueError(f"unsupported vector orientation: {orientation}")
    return tiles.reshape(-1, VECTOR_DIMENSION)


def _matrix_tiles(matrix: torch.Tensor) -> torch.Tensor:
    rows, columns = matrix.shape
    if rows % TILE_SIZE or columns % TILE_SIZE:
        raise ValueError(f"matrix shape {tuple(matrix.shape)} is not tile aligned")
    return (
        matrix.reshape(rows // TILE_SIZE, TILE_SIZE, columns // TILE_SIZE, TILE_SIZE)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _selected_tile_residuals(
    residual: torch.Tensor, fraction: float, orientation: str
) -> tuple[torch.Tensor, float, int, int]:
    tiles = _matrix_tiles(residual)
    energy = tiles.double().square().sum(dim=(-2, -1))
    total_tiles = energy.numel()
    selected_tiles = max(1, round(fraction * total_tiles))
    flat_index = energy.flatten().topk(selected_tiles).indices
    selected = tiles.reshape(total_tiles, TILE_SIZE, TILE_SIZE)[flat_index]
    return (
        _tile_vectors(selected, orientation),
        float(energy.flatten()[flat_index].sum()),
        selected_tiles,
        total_tiles,
    )


def _sample_training_vectors(
    store: InterimKeepStore,
    *,
    layer: int,
    experts: list[int],
    matrix: str,
    orientation: str,
    count: int,
    seed: int,
) -> torch.Tensor:
    per_expert = math.ceil(count / len(experts))
    chunks = []
    for offset, expert in enumerate(experts):
        weight = store.load_matrix(layer, expert, matrix)
        vectors = _vectorize(weight, orientation)
        generator = torch.Generator(device="cpu").manual_seed(seed + offset * 10_000)
        index = torch.randperm(vectors.shape[0], generator=generator)[:per_expert]
        chunks.append(vectors[index].float())
        del weight, vectors
    return torch.cat(chunks)[:count].contiguous()


def _encode_base(
    matrix: torch.Tensor,
    model: ResidualVQModel,
    *,
    orientation: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = matrix.float().to(device)
    vectors = _vectorize(source, orientation)
    reconstruction, labels = apply_residual_vq(vectors, model)
    return source - _unvectorize(
        reconstruction, tuple(source.shape), orientation
    ), labels[0]


def _train_base(
    store: InterimKeepStore,
    args: argparse.Namespace,
    experts: list[int],
    device: torch.device,
) -> ResidualVQModel:
    training = _sample_training_vectors(
        store,
        layer=args.layer,
        experts=experts,
        matrix=args.matrix,
        orientation=args.orientation,
        count=args.train_vectors,
        seed=args.seed,
    ).to(device)
    model = train_residual_vq(
        training,
        stages=1,
        codebook_size=4096,
        iterations=args.iterations,
        seed=args.seed,
    )
    del training
    return model


def run(args: argparse.Namespace) -> dict:
    store = InterimKeepStore(args.checkpoint)
    ordered, traffic = routed_kept_experts(store, args.capture, args.layer)
    if args.held_out_expert is None:
        held_out_expert = ordered[0]
    elif args.held_out_expert not in ordered:
        raise ValueError(f"expert {args.layer}.{args.held_out_expert} is not retained")
    else:
        held_out_expert = args.held_out_expert
    training_experts = [expert for expert in ordered if expert != held_out_expert][
        : args.train_experts
    ]
    if len(training_experts) != args.train_experts:
        raise ValueError("not enough retained experts for held-out validation")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    base_model = _train_base(store, args, training_experts, device)

    residual_training: dict[float, list[torch.Tensor]] = {
        fraction: [] for fraction in args.tile_fractions
    }
    training_tile_shares: dict[float, list[float]] = {
        fraction: [] for fraction in args.tile_fractions
    }
    for expert in training_experts:
        weight = store.load_matrix(args.layer, expert, args.matrix)
        residual, _ = _encode_base(
            weight, base_model, orientation=args.orientation, device=device
        )
        total_error = float(residual.double().square().sum())
        for fraction in args.tile_fractions:
            selected, energy, _, _ = _selected_tile_residuals(
                residual, fraction, args.orientation
            )
            residual_training[fraction].append(selected.cpu())
            training_tile_shares[fraction].append(energy / total_error)
        del weight, residual
        if device.type == "cuda":
            torch.cuda.empty_cache()

    residual_models: dict[tuple[float, int], ResidualVQModel] = {}
    for fraction in args.tile_fractions:
        training = torch.cat(residual_training[fraction]).to(device)
        for codebook_size in args.residual_codebook_sizes:
            residual_models[(fraction, codebook_size)] = train_residual_vq(
                training,
                stages=1,
                codebook_size=codebook_size,
                iterations=args.iterations,
                seed=args.seed + round(fraction * 1_000_000) + codebook_size,
            )
        del training

    held_out_weight = store.load_matrix(args.layer, held_out_expert, args.matrix)
    held_out_residual, base_labels = _encode_base(
        held_out_weight, base_model, orientation=args.orientation, device=device
    )
    source_energy = float(held_out_weight.double().square().sum())
    base_error = float(held_out_residual.double().square().sum())
    base_entropy = index_entropy(base_labels, 4096)
    layer_samples = load_layer_samples(args.capture, args.layer - 1)
    activation_selected = layer_samples.mid_experts == held_out_expert
    captured_activation = None
    if torch.count_nonzero(activation_selected):
        activation = layer_samples.mid_values[activation_selected].float().to(device)
        activation_weight = (
            layer_samples.mid_weights[activation_selected].double().to(device)
        )
        source_device = held_out_weight.float().to(device)
        reference_output = activation @ source_device.T
        reconstruction_output = activation @ (source_device - held_out_residual).T
        denominator = (
            reference_output.double().square().sum(dim=1) * activation_weight
        ).sum()
        numerator = (
            (reference_output - reconstruction_output).double().square().sum(dim=1)
            * activation_weight
        ).sum()
        captured_activation = {
            "rows": int(activation.shape[0]),
            "weighted_output_nmse": float(numerator / denominator),
        }
    full_matrix_weights = held_out_weight.numel()
    base_codebook_bits = sum(codebook.numel() * 16 for codebook in base_model.codebooks)
    enhancements = []
    for fraction in args.tile_fractions:
        selected, selected_energy, selected_tiles, total_tiles = (
            _selected_tile_residuals(held_out_residual, fraction, args.orientation)
        )
        actual_fraction = selected_tiles / total_tiles
        for codebook_size in args.residual_codebook_sizes:
            model = residual_models[(fraction, codebook_size)]
            correction, labels = apply_residual_vq(selected, model)
            corrected_selected_error = float(
                (selected - correction).double().square().sum()
            )
            corrected_error = base_error - selected_energy + corrected_selected_error
            residual_index_bits = int(math.log2(codebook_size))
            residual_entropy = index_entropy(labels[0], codebook_size)
            residual_codebook_bits = sum(
                codebook.numel() * 16 for codebook in model.codebooks
            )
            codebook_rate = (base_codebook_bits + residual_codebook_bits) / (
                C.NUM_EXPERTS * full_matrix_weights
            )
            fixed_flag_rate = 1 / (TILE_SIZE * TILE_SIZE)
            entropy_flag_rate = _binary_entropy(actual_fraction) / (
                TILE_SIZE * TILE_SIZE
            )
            enhancements.append(
                {
                    "tile_fraction": actual_fraction,
                    "selected_tiles": selected_tiles,
                    "total_tiles": total_tiles,
                    "residual_codebook_size": codebook_size,
                    "residual_index_bits_per_vector": residual_index_bits,
                    "base_error_share_in_selected_tiles": selected_energy / base_error,
                    "training_mean_base_error_share_in_selected_tiles": sum(
                        training_tile_shares[fraction]
                    )
                    / len(training_tile_shares[fraction]),
                    "selected_tile_error_reduction": 1
                    - corrected_selected_error / selected_energy,
                    "nmse": corrected_error / source_energy,
                    "relative_nmse_vs_base": corrected_error / base_error,
                    "fixed_rate_bpw": 3.0
                    + fixed_flag_rate
                    + actual_fraction * residual_index_bits / VECTOR_DIMENSION
                    + codebook_rate,
                    "zero_order_entropy_lower_bound_bpw": base_entropy
                    / VECTOR_DIMENSION
                    + entropy_flag_rate
                    + actual_fraction * residual_entropy / VECTOR_DIMENSION
                    + codebook_rate,
                    "rate_components": {
                        "base_fixed_index_bpw": 3.0,
                        "base_entropy_index_bpw": base_entropy / VECTOR_DIMENSION,
                        "fixed_tile_flag_bpw": fixed_flag_rate,
                        "entropy_tile_flag_bpw": entropy_flag_rate,
                        "residual_fixed_index_bpw": actual_fraction
                        * residual_index_bits
                        / VECTOR_DIMENSION,
                        "residual_entropy_index_bpw": actual_fraction
                        * residual_entropy
                        / VECTOR_DIMENSION,
                        "layer_shared_codebooks_bpw": codebook_rate,
                    },
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "kind": "kquant_sparse_enhancement_vq_experiment",
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
        "base": {
            "vector_dimension": VECTOR_DIMENSION,
            "orientation": args.orientation,
            "codebook_size": 4096,
            "fixed_rate_bpw": 3.0
            + base_codebook_bits / (C.NUM_EXPERTS * full_matrix_weights),
            "zero_order_entropy_lower_bound_bpw": base_entropy / VECTOR_DIMENSION
            + base_codebook_bits / (C.NUM_EXPERTS * full_matrix_weights),
            "index_entropy_bits": base_entropy,
            "full_weight_nmse": base_error / source_energy,
            "captured_activation": captured_activation,
        },
        "enhancements": enhancements,
        "experiment": {
            "tile_size": TILE_SIZE,
            "training_vectors": args.train_vectors,
            "iterations": args.iterations,
            "seed": args.seed,
            "selection": "highest base-layer squared-error tiles",
            "validation": "held-out expert",
            "elapsed_seconds": elapsed,
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
    parser.add_argument("--orientation", choices=("row", "column"), default="column")
    parser.add_argument("--train-experts", type=int, default=4)
    parser.add_argument("--held-out-expert", type=int)
    parser.add_argument("--train-vectors", type=int, default=1_048_576)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--tile-fractions", nargs="+", type=float, default=(0.01, 0.02, 0.04, 0.08)
    )
    parser.add_argument(
        "--residual-codebook-sizes", nargs="+", type=int, default=(2, 4, 16)
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= C.NUM_MOE_LAYERS:
        parser.error("--layer must be 1..92")
    if any(not 0 < fraction < 1 for fraction in args.tile_fractions):
        parser.error("tile fractions must be between zero and one")
    if any(size <= 1 or size & (size - 1) for size in args.residual_codebook_sizes):
        parser.error("residual codebook sizes must be powers of two")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
