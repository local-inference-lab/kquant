"""Test activation-context palette coding on interim K3 ``w2`` weights.

Modern codecs do not force one transform or quantizer on every block. This
experiment assigns each four-channel group to an activation-energy context,
trains a separate layer-shared palette for each context, and varies the fixed
index width while holding the average at three bits per weight. Context maps
are expert-specific and tiny compared with a weight matrix.

Only retained MXFP4 weights from the packaged interim EXL3 checkpoint and raw
samples from its capture are read. The official checkpoint is never resolved
or loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from kquant import constants as C
from kquant.capture import LayerSamples, load_layer_samples
from kquant.codec_research import (
    apply_residual_vq,
    contextual_index_rate,
    index_entropy,
    normalized_mse,
    rank_contexts,
    train_residual_vq,
)
from kquant.interim_weights import InterimKeepStore, routed_kept_experts

VECTOR_DIMENSION = 4


def _parse_allocations(value: str) -> tuple[tuple[int, ...], ...]:
    result = []
    for configuration in value.split(";"):
        bits = tuple(int(item) for item in configuration.split(","))
        if not bits or min(bits) <= 0:
            raise argparse.ArgumentTypeError("index widths must be positive")
        result.append(bits)
    contexts = len(result[0])
    if any(len(bits) != contexts for bits in result):
        raise argparse.ArgumentTypeError(
            "all rate allocations must have the same context count"
        )
    expected_sum = 3 * VECTOR_DIMENSION * contexts
    if any(sum(bits) != expected_sum for bits in result):
        raise argparse.ArgumentTypeError(
            "each allocation must average exactly three bits per weight"
        )
    return tuple(result)


def _expert_scores(
    samples: LayerSamples,
    expert: int,
    *,
    split: int | None,
    request_documents: Mapping[int, str] | None = None,
) -> tuple[torch.Tensor, int, int]:
    selected = samples.mid_experts == expert
    if split is not None:
        selected &= samples.mid_split == split
    request_steps = torch.bitwise_right_shift(samples.mid_observations, 32)
    if request_documents is not None:
        allowed = torch.tensor(
            sorted(request_documents), dtype=request_steps.dtype
        )
        selected &= torch.isin(request_steps, allowed)
    rows = int(torch.count_nonzero(selected))
    if not rows:
        raise ValueError(f"expert {expert} has no selected middle samples")
    values = samples.mid_values[selected].float()
    weights = samples.mid_weights[selected].double()
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError(f"expert {expert} has non-positive sample weight")
    square = (values.double().square() * weights[:, None]).sum(dim=0) / denominator
    documents = int(torch.unique(request_steps[selected]).numel())
    return square.reshape(-1, VECTOR_DIMENSION).mean(dim=1).float(), rows, documents


def _contexts_for_expert(
    samples: LayerSamples,
    expert: int,
    contexts: int,
    *,
    split: int | None = 0,
    request_documents: Mapping[int, str] | None = None,
) -> tuple[torch.Tensor, dict]:
    scores, rows, documents = _expert_scores(
        samples,
        expert,
        split=split,
        request_documents=request_documents,
    )
    assignments = rank_contexts(scores, contexts)
    return assignments, {
        "training_rows": rows,
        "training_documents": documents,
        "sample_split": "all" if split is None else int(split),
        "request_filter": (
            None
            if request_documents is None
            else {
                "minimum": min(request_documents),
                "maximum": max(request_documents),
                "documents": len(request_documents),
                "encoding": "observation_id >> 32",
            }
        ),
        "score_min": float(scores.min()),
        "score_median": float(scores.median()),
        "score_max": float(scores.max()),
        "context_score_means": [
            float(scores[assignments == context].mean()) for context in range(contexts)
        ],
    }


def _sample_vectors(
    matrix: torch.Tensor,
    assignments: torch.Tensor,
    context: int,
    count: int,
    *,
    seed: int,
) -> torch.Tensor:
    blocks = matrix.reshape(matrix.shape[0], -1, VECTOR_DIMENSION)
    vectors = blocks[:, assignments == context].reshape(-1, VECTOR_DIMENSION)
    count = min(count, vectors.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    index = torch.randperm(vectors.shape[0], generator=generator)[:count]
    return vectors[index].float().contiguous()


def _training_vectors(
    store: InterimKeepStore,
    samples: LayerSamples,
    *,
    layer: int,
    experts: list[int],
    contexts: int,
    vectors_per_context: int,
    seed: int,
) -> tuple[list[torch.Tensor], dict[int, dict]]:
    per_expert = math.ceil(vectors_per_context / len(experts))
    chunks: list[list[torch.Tensor]] = [[] for _ in range(contexts)]
    metadata = {}
    for offset, expert in enumerate(experts):
        assignments, context_meta = _contexts_for_expert(samples, expert, contexts)
        metadata[expert] = context_meta
        matrix = store.load_matrix(layer, expert, "w2")
        for context in range(contexts):
            chunks[context].append(
                _sample_vectors(
                    matrix,
                    assignments,
                    context,
                    per_expert,
                    seed=seed + 10_000 * offset + context,
                )
            )
        del matrix
    return (
        [torch.cat(parts)[:vectors_per_context] for parts in chunks],
        metadata,
    )


def _activation_metrics(
    source: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    samples: LayerSamples,
    validation_samples: LayerSamples | None = None,
    expert: int,
    device: torch.device,
    training_request_documents: Mapping[int, str] | None = None,
    validation_request_documents: Mapping[int, str] | None = None,
) -> dict[str, dict | None]:
    metrics = {}
    if validation_samples is None:
        selections = (
            ("train", samples, 0, training_request_documents),
            ("validation", samples, 1, validation_request_documents),
            ("all", samples, None, training_request_documents),
        )
    else:
        selections = (
            ("train", samples, None, training_request_documents),
            (
                "validation",
                validation_samples,
                None,
                validation_request_documents,
            ),
        )
    for name, selected_samples, split, request_documents in selections:
        selected = selected_samples.mid_experts == expert
        if split is not None:
            selected &= selected_samples.mid_split == split
        request_steps = torch.bitwise_right_shift(
            selected_samples.mid_observations, 32
        )
        selected_before_request_filter = int(torch.count_nonzero(selected))
        if request_documents is not None:
            allowed = torch.tensor(
                sorted(request_documents), dtype=request_steps.dtype
            )
            selected &= torch.isin(request_steps, allowed)
        if not torch.count_nonzero(selected):
            metrics[name] = None
            continue
        inputs = selected_samples.mid_values[selected].float().to(device)
        weights = selected_samples.mid_weights[selected].double().to(device)
        reference = inputs @ source.T
        candidate = inputs @ reconstruction.T
        row_reference_energy = reference.double().square().sum(dim=1) * weights
        row_squared_error = (
            (reference - candidate).double().square().sum(dim=1) * weights
        )
        denominator = row_reference_energy.sum()
        numerator = row_squared_error.sum()
        if denominator <= 0:
            raise ValueError(
                f"expert {expert} has non-positive {name} reference energy"
            )
        selected_steps = request_steps[selected].tolist()
        row_reference_energy = row_reference_energy.cpu()
        row_squared_error = row_squared_error.cpu()
        row_weights = weights.cpu()
        request_contributions = []
        for request_step in sorted(set(selected_steps)):
            indices = [
                index
                for index, value in enumerate(selected_steps)
                if value == request_step
            ]
            index = torch.tensor(indices, dtype=torch.long)
            request_contributions.append(
                {
                    "request_step": request_step,
                    "document_hash": (
                        None
                        if request_documents is None
                        else request_documents[request_step]
                    ),
                    "rows": len(indices),
                    "weight_sum": float(row_weights[index].sum()),
                    "weighted_squared_error": float(
                        row_squared_error[index].sum()
                    ),
                    "weighted_reference_energy": float(
                        row_reference_energy[index].sum()
                    ),
                }
            )
        metrics[name] = {
            "rows": int(inputs.shape[0]),
            "documents": len(request_contributions),
            "excluded_unmapped_rows": (
                selected_before_request_filter - int(inputs.shape[0])
            ),
            "weight_sum": float(weights.sum()),
            "weighted_squared_error": float(numerator),
            "weighted_reference_energy": float(denominator),
            "weighted_output_nmse": float(numerator / denominator),
            "request_contributions": request_contributions,
        }
    return metrics


def _load_anchor(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    return {"path": str(path.resolve()), **payload["exl3_baseline"]}


def _encode_configuration(
    source: torch.Tensor,
    assignments: torch.Tensor,
    training: list[torch.Tensor],
    index_bits: tuple[int, ...],
    *,
    iterations: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    blocks = source.reshape(source.shape[0], -1, VECTOR_DIMENSION)
    reconstruction = torch.empty_like(blocks)
    context_metrics = []
    codebook_bits = 0
    entropy_bits = 0.0
    total_vectors = 0
    for context, bits in enumerate(index_bits):
        codebook_size = 1 << bits
        batch_size = 2048 if bits >= 15 else 8192
        model = train_residual_vq(
            training[context].to(device),
            stages=1,
            codebook_size=codebook_size,
            iterations=iterations,
            batch_size=batch_size,
            seed=seed + context,
        )
        selected = blocks[:, assignments == context]
        vectors = selected.reshape(-1, VECTOR_DIMENSION)
        decoded, labels = apply_residual_vq(
            vectors,
            model,
            batch_size=batch_size,
        )
        reconstruction[:, assignments == context] = decoded.reshape_as(selected)
        count = vectors.shape[0]
        entropy = index_entropy(labels[0], codebook_size)
        entropy_bits += entropy * count
        total_vectors += count
        stored_bits = sum(codebook.numel() * 16 for codebook in model.codebooks)
        codebook_bits += stored_bits
        context_metrics.append(
            {
                "context": context,
                "index_bits": bits,
                "codebook_size": codebook_size,
                "codebook_bytes": stored_bits // 8,
                "vectors": count,
                "index_entropy_bits": entropy,
                "codebook_occupancy": int(torch.unique(labels[0]).numel())
                / codebook_size,
                "weight_nmse": normalized_mse(vectors, decoded),
            }
        )
        del model, selected, vectors, decoded, labels
    return reconstruction.reshape_as(source), {
        "contexts": context_metrics,
        "codebook_bits": codebook_bits,
        "entropy_rate_bpw": entropy_bits / (total_vectors * VECTOR_DIMENSION),
    }


def run(args: argparse.Namespace) -> dict:
    store = InterimKeepStore(args.checkpoint)
    samples = load_layer_samples(args.capture, args.layer - 1)
    ordered, traffic = routed_kept_experts(store, args.capture, args.layer)
    available = [
        expert
        for expert in ordered
        if torch.any((samples.mid_experts == expert) & (samples.mid_split == 0))
    ]
    if args.held_out_expert is None:
        if not available:
            raise ValueError("no retained expert has training capture rows")
        held_out = available[0]
    elif args.held_out_expert not in available:
        raise ValueError("held-out expert is not retained with training samples")
    else:
        held_out = args.held_out_expert
    training_experts = [expert for expert in available if expert != held_out][
        : args.train_experts
    ]
    if len(training_experts) != args.train_experts:
        raise ValueError(
            f"need {args.train_experts} captured training experts, "
            f"found {len(training_experts)}"
        )

    context_count = len(args.allocations[0])
    training, training_meta = _training_vectors(
        store,
        samples,
        layer=args.layer,
        experts=training_experts,
        contexts=context_count,
        vectors_per_context=args.train_vectors_per_context,
        seed=args.seed,
    )
    held_out_contexts, held_out_meta = _contexts_for_expert(
        samples, held_out, context_count
    )
    device = torch.device(args.device)
    source = store.load_matrix(args.layer, held_out, "w2").float().to(device)
    held_out_contexts = held_out_contexts.to(device)
    full_matrix_weights = source.numel()
    map_bits = held_out_contexts.numel() * math.ceil(math.log2(context_count))
    results = []
    for configuration_index, index_bits in enumerate(args.allocations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        reconstruction, coding = _encode_configuration(
            source,
            held_out_contexts,
            training,
            index_bits,
            iterations=args.iterations,
            seed=args.seed + configuration_index * 1_000_000,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        fixed_rate = contextual_index_rate(
            held_out_contexts,
            index_bits,
            vector_dimension=VECTOR_DIMENSION,
        )
        results.append(
            {
                "index_bits_low_to_high_context": list(index_bits),
                "fixed_index_rate_bpw": fixed_rate,
                "expert_context_map_bytes": map_bits // 8,
                "layer_shared_codebook_bytes": coding["codebook_bits"] // 8,
                "total_amortized_rate_bpw": fixed_rate
                + map_bits / full_matrix_weights
                + coding["codebook_bits"] / (C.NUM_EXPERTS * full_matrix_weights),
                "weight_nmse": normalized_mse(source, reconstruction),
                "entropy_rate_lower_bound_bpw": coding["entropy_rate_bpw"],
                "captured_activation": _activation_metrics(
                    source,
                    reconstruction,
                    samples=samples,
                    expert=held_out,
                    device=device,
                ),
                "context_metrics": coding["contexts"],
                "training_and_encoding_seconds": elapsed,
            }
        )
        del reconstruction
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "kind": "kquant_context_palette_experiment",
        "schema_version": 1,
        "source": {
            "checkpoint": str(store.root),
            "constraint": "interim EXL3 hybrid only; official model not loaded",
            "capture": str(args.capture.resolve()),
            "layer": args.layer,
            "matrix": "w2",
            "held_out_expert": held_out,
            "training_experts": training_experts,
            "captured_gate_square_mass": {
                str(expert): traffic.get(expert, 0.0)
                for expert in [held_out, *training_experts]
            },
        },
        "context": {
            "definition": (
                "equal-population rank bins of four-channel mean activation "
                "energy, computed independently per expert from training split"
            ),
            "held_out": held_out_meta,
            "training": {str(key): value for key, value in training_meta.items()},
            "map_can_be_folded_into_intermediate_neuron_permutation": True,
        },
        "experiment": {
            "training_vectors_per_context": args.train_vectors_per_context,
            "iterations": args.iterations,
            "seed": args.seed,
            "results": results,
        },
        "anchors": {
            "identity_h_exl3": _load_anchor(args.identity_exl3_anchor),
            "captured_h_exl3": _load_anchor(args.hessian_exl3_anchor),
        },
        "limitations": [
            "pilot capture has only four tokens and cannot support a quality claim",
            "palettes are simulated; no production decoder or MMA kernel exists",
            "context-map and codebook rates are included but alignment is not",
        ],
    }


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--held-out-expert", type=int)
    parser.add_argument("--train-experts", type=int, default=4)
    parser.add_argument(
        "--allocations",
        type=_parse_allocations,
        default=_parse_allocations("12,12,12,12;10,11,13,14"),
        help="semicolon-separated low-to-high context index widths",
    )
    parser.add_argument("--train-vectors-per-context", type=int, default=524288)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--identity-exl3-anchor", type=Path)
    parser.add_argument("--hessian-exl3-anchor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
