"""Test activation-whitened shared VQ on interim K3 ``w2`` weights.

This transfers transform coding from image/audio codecs: a layer-shared
diagonal or 4x4 block transform whitens the measured activation metric before
a 12-bit, 4-weight palette is trained.  Retained source weights, activations,
and Hessians all come from the interim EXL3 artifact/capture; the official
checkpoint is never resolved or loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from kquant import constants as C
from kquant.capture import load_layer_hessians, load_layer_samples
from kquant.codec_research import (
    apply_residual_vq,
    index_entropy,
    normalized_mse,
    train_residual_vq,
)
from kquant.interim_weights import InterimKeepStore, routed_kept_experts

VECTOR_DIMENSION = 4


def _block_diagonal(hessian: torch.Tensor) -> torch.Tensor:
    blocks = hessian.shape[0] // VECTOR_DIMENSION
    reshaped = hessian.reshape(blocks, VECTOR_DIMENSION, blocks, VECTOR_DIMENSION)
    index = torch.arange(blocks)
    return reshaped[index, :, index, :]


def _transform_factors(hessian: torch.Tensor, mode: str) -> tuple[torch.Tensor, int]:
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Hessian must be square")
    if hessian.shape[0] % VECTOR_DIMENSION:
        raise ValueError("Hessian dimension is not 4-vector aligned")
    dimension = hessian.shape[0]
    blocks = dimension // VECTOR_DIMENSION
    mean_diagonal = hessian.diag().double().mean().clamp_min(1e-30)
    normalized = hessian.float() / mean_diagonal.float()
    if mode == "none":
        factor = torch.eye(VECTOR_DIMENSION).expand(blocks, -1, -1).clone()
        stored_bits = 0
    elif mode == "diagonal":
        floor = normalized.diag().mean() * 1e-6
        scale = normalized.diag().clamp_min(floor).sqrt()
        factor = torch.diag_embed(scale.reshape(blocks, VECTOR_DIMENSION))
        stored_bits = dimension * 16
    elif mode == "block4_cholesky":
        block_hessian = _block_diagonal(normalized)
        regularization = block_hessian.diagonal(dim1=1, dim2=2).mean() * 1e-5
        identity = torch.eye(VECTOR_DIMENSION)[None]
        factor = torch.linalg.cholesky(block_hessian + regularization * identity)
        stored_bits = factor.numel() * 16
    else:
        raise ValueError(f"unsupported Hessian transform: {mode}")
    # Decoder-side factors are FP16. Include their rounding in all scores.
    return factor.half().float().contiguous(), stored_bits


def _transform_weight(matrix: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    blocks = matrix.reshape(matrix.shape[0], -1, VECTOR_DIMENSION).float()
    return torch.einsum("obi,bij->obj", blocks, factor).reshape_as(matrix)


def _inverse_transform_weight(
    transformed: torch.Tensor, factor: torch.Tensor
) -> torch.Tensor:
    inverse = torch.linalg.inv(factor)
    blocks = transformed.reshape(transformed.shape[0], -1, VECTOR_DIMENSION).float()
    return torch.einsum("obi,bij->obj", blocks, inverse).reshape_as(transformed)


def _training_vectors(
    store: InterimKeepStore,
    *,
    layer: int,
    experts: list[int],
    factor: torch.Tensor,
    count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    per_expert = math.ceil(count / len(experts))
    chunks = []
    for offset, expert in enumerate(experts):
        weight = store.load_matrix(layer, expert, "w2").to(device)
        transformed = _transform_weight(weight, factor)
        vectors = transformed.reshape(-1, VECTOR_DIMENSION)
        generator = torch.Generator(device=device).manual_seed(seed + 10_000 * offset)
        index = torch.randperm(vectors.shape[0], device=device, generator=generator)[
            :per_expert
        ]
        chunks.append(vectors[index])
        del weight, transformed, vectors
    return torch.cat(chunks)[:count].contiguous()


def _activation_metrics(
    source: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    capture: Path,
    layer: int,
    expert: int,
    device: torch.device,
) -> dict[str, dict | None]:
    samples = load_layer_samples(capture, layer - 1)
    metrics = {}
    for name, split in (("train", 0), ("validation", 1), ("all", None)):
        selected = samples.mid_experts == expert
        if split is not None:
            selected &= samples.mid_split == split
        if not torch.count_nonzero(selected):
            metrics[name] = None
            continue
        inputs = samples.mid_values[selected].float().to(device)
        weights = samples.mid_weights[selected].double().to(device)
        reference_output = inputs @ source.T
        quantized_output = inputs @ reconstruction.T
        denominator = (reference_output.double().square().sum(dim=1) * weights).sum()
        numerator = (
            (reference_output - quantized_output).double().square().sum(dim=1) * weights
        ).sum()
        metrics[name] = {
            "rows": int(inputs.shape[0]),
            "weighted_output_nmse": float(numerator / denominator),
        }
    return metrics


def _load_anchor(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    return {"path": str(path.resolve()), **payload["exl3_baseline"]}


def run(args: argparse.Namespace) -> dict:
    store = InterimKeepStore(args.checkpoint)
    ordered, traffic = routed_kept_experts(store, args.capture, args.layer)
    held_out_expert = (
        ordered[0] if args.held_out_expert is None else args.held_out_expert
    )
    if held_out_expert not in ordered:
        raise ValueError(f"expert {args.layer}.{held_out_expert} is not retained")
    training_experts = [expert for expert in ordered if expert != held_out_expert][
        : args.train_experts
    ]
    if len(training_experts) != args.train_experts:
        raise ValueError("not enough retained experts for held-out validation")
    _, h2 = load_layer_hessians(args.hessians, args.layer)
    device = torch.device(args.device)
    source = store.load_matrix(args.layer, held_out_expert, "w2").float().to(device)
    full_matrix_weights = source.numel()
    results = []
    for mode_index, mode in enumerate(args.transforms):
        factor, transform_bits = _transform_factors(h2, mode)
        factor = factor.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        training = _training_vectors(
            store,
            layer=args.layer,
            experts=training_experts,
            factor=factor,
            count=args.train_vectors,
            seed=args.seed + 1_000_000 * mode_index,
            device=device,
        )
        model = train_residual_vq(
            training,
            stages=1,
            codebook_size=4096,
            iterations=args.iterations,
            seed=args.seed + mode_index,
        )
        transformed_source = _transform_weight(source, factor)
        transformed_reconstruction, labels = apply_residual_vq(
            transformed_source.reshape(-1, VECTOR_DIMENSION), model
        )
        transformed_reconstruction = transformed_reconstruction.reshape_as(source)
        reconstruction = _inverse_transform_weight(transformed_reconstruction, factor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        codebook_bits = sum(codebook.numel() * 16 for codebook in model.codebooks)
        results.append(
            {
                "transform": mode,
                "vector_dimension": VECTOR_DIMENSION,
                "codebook_size": 4096,
                "fixed_index_rate_bpw": 3.0,
                "layer_shared_transform_bytes": transform_bits // 8,
                "layer_shared_codebook_bytes": codebook_bits // 8,
                "total_amortized_rate_bpw": 3.0
                + (transform_bits + codebook_bits)
                / (C.NUM_EXPERTS * full_matrix_weights),
                "original_weight_nmse": normalized_mse(source, reconstruction),
                "transformed_weight_nmse": normalized_mse(
                    transformed_source, transformed_reconstruction
                ),
                "index_entropy_bits": index_entropy(labels[0], 4096),
                "captured_activation": _activation_metrics(
                    source,
                    reconstruction,
                    capture=args.capture,
                    layer=args.layer,
                    expert=held_out_expert,
                    device=device,
                ),
                "training_and_encoding_seconds": elapsed,
            }
        )
        del training, model, transformed_source, transformed_reconstruction
        del reconstruction, labels, factor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "kind": "kquant_hessian_vq_experiment",
        "schema_version": 1,
        "source": {
            "checkpoint": str(store.root),
            "constraint": "interim EXL3 hybrid only; official model not loaded",
            "capture": str(args.capture.resolve()),
            "hessians": str(args.hessians.resolve()),
            "layer": args.layer,
            "matrix": "w2",
            "held_out_expert": held_out_expert,
            "training_experts": training_experts,
            "captured_gate_square_mass": {
                str(expert): traffic.get(expert, 0.0)
                for expert in [held_out_expert, *training_experts]
            },
        },
        "experiment": {
            "training_vectors": args.train_vectors,
            "iterations": args.iterations,
            "seed": args.seed,
            "results": results,
        },
        "anchors": {
            "identity_h_exl3": _load_anchor(args.identity_exl_reference),
            "captured_h_exl3": _load_anchor(args.hessian_exl_reference),
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
    parser.add_argument(
        "--hessians",
        type=Path,
        default=Path("out/k3-route-aware-pilot-v2-exl3-r4-subset.kqhess"),
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--held-out-expert", type=int)
    parser.add_argument("--train-experts", type=int, default=4)
    parser.add_argument("--train-vectors", type=int, default=1_048_576)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=("none", "diagonal", "block4_cholesky"),
        default=("none", "diagonal", "block4_cholesky"),
    )
    parser.add_argument("--identity-exl-reference", type=Path)
    parser.add_argument("--hessian-exl-reference", type=Path)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
