"""Evaluate coupled TP12 ``(r13, r2)`` rate-transfer schedules.

The resident teacher, route samples, Hessians, and source weights in this
experiment all come from the interim EXL3 hybrid.  It never resolves or loads
the official checkpoint.  The same learned intermediate-neuron permutation
is applied to ``w1`` rows, ``w3`` rows, and ``w2`` columns.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F

from kquant.capture import LayerSamples, load_layer_hessians, load_layer_samples
from kquant.codec_research import normalized_mse
from kquant.interim_weights import InterimKeepStore
from kquant.mixed_exl3 import (
    RATE_TRANSFER_MODES,
    coupled_intermediate_permutation,
    expand_group_order,
    tp12_storage_group_order,
)
from kquant.tp_simulator import comparison_metrics, situ
from scripts.experiment_codec_transfers import _load_exl3_quantizer
from scripts.experiment_context_exl3 import (
    _contexts_for_expert,
    _parse_r_values,
    _quantize_full_hessian_mixed,
)
from scripts.experiment_retained_w2_rate_transfer import (
    _read_json,
    _request_documents,
    _validate_corpus_reports,
)


def _expert_output(
    inputs: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    middle = situ(F.linear(inputs, w1), F.linear(inputs, w3))
    return F.linear(middle, w2)


def _selected_expert_rows(
    samples: LayerSamples,
    expert: int,
    split: int | None,
    request_documents: Mapping[int, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    matches = samples.input_experts == expert
    selected = matches.any(dim=1)
    if split is not None:
        selected &= samples.input_split == split
    request_steps = torch.bitwise_right_shift(samples.input_observations, 32)
    if request_documents is not None:
        allowed = torch.tensor(sorted(request_documents), dtype=request_steps.dtype)
        selected &= torch.isin(request_steps, allowed)
    if not torch.count_nonzero(selected):
        return (
            samples.input_values[:0],
            samples.input_gates[:0, 0],
            samples.routed_latent[:0],
            request_steps[:0],
        )
    gates = (samples.input_gates[selected] * matches[selected]).sum(dim=1)
    return (
        samples.input_values[selected],
        gates,
        samples.routed_latent[selected],
        request_steps[selected],
    )


def _coupled_activation_metrics(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reconstruction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    training_samples: LayerSamples,
    validation_samples: LayerSamples | None,
    expert: int,
    device: torch.device,
    training_request_documents: Mapping[int, str] | None = None,
    training_confirmation_documents: Mapping[int, str] | None = None,
    validation_request_documents: Mapping[int, str] | None = None,
) -> dict[str, dict | None]:
    metrics: dict[str, dict | None] = {}
    if validation_samples is None:
        selections = (
            ("train", training_samples, 0, training_request_documents),
            ("validation", training_samples, 1, validation_request_documents),
            ("all", training_samples, None, training_request_documents),
        )
    else:
        selections = [
            ("train", training_samples, None, training_request_documents),
        ]
        if training_confirmation_documents is not None:
            selections.append(
                (
                    "confirmation",
                    training_samples,
                    None,
                    training_confirmation_documents,
                )
            )
        selections.append(
            (
                "validation",
                validation_samples,
                None,
                validation_request_documents,
            )
        )
    for name, samples, split, request_documents in selections:
        inputs, gates, teacher_mixture, request_steps = _selected_expert_rows(
            samples,
            expert,
            split,
            request_documents,
        )
        if not inputs.shape[0]:
            metrics[name] = None
            continue
        inputs = inputs.float().to(device)
        gates = gates.float().to(device)
        teacher_mixture = teacher_mixture.float().to(device)
        reference = _expert_output(inputs, *source)
        candidate = _expert_output(inputs, *reconstruction)
        delta = candidate - reference
        route_weights = gates.double().square()
        row_contribution_denominator = (
            reference.double().square().sum(dim=1) * route_weights
        )
        row_contribution_numerator = (
            delta.double().square().sum(dim=1) * route_weights
        )
        row_mixture_denominator = teacher_mixture.double().square().sum(dim=1)
        contribution_denominator = row_contribution_denominator.sum()
        contribution_numerator = row_contribution_numerator.sum()
        mixture_denominator = row_mixture_denominator.sum()
        # Applying this expert's gate to its output error produces the same
        # numerator as the gate-square-weighted isolated-expert self term.
        mixture_numerator = contribution_numerator
        if contribution_denominator <= 0 or mixture_denominator <= 0:
            raise ValueError(f"expert {expert} has non-positive {name} energy")
        row_contribution_numerator = row_contribution_numerator.cpu()
        row_contribution_denominator = row_contribution_denominator.cpu()
        row_mixture_denominator = row_mixture_denominator.cpu()
        row_weights = route_weights.cpu()
        selected_steps = request_steps.tolist()
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
                    "gate_square_sum": float(row_weights[index].sum()),
                    "route_weighted_squared_error": float(
                        row_contribution_numerator[index].sum()
                    ),
                    "route_weighted_reference_energy": float(
                        row_contribution_denominator[index].sum()
                    ),
                    "teacher_mixture_energy": float(
                        row_mixture_denominator[index].sum()
                    ),
                }
            )
        metrics[name] = {
            "rows": int(inputs.shape[0]),
            "documents": len(request_contributions),
            "gate_square_sum": float(route_weights.sum()),
            "route_weighted_squared_error": float(contribution_numerator),
            "route_weighted_reference_energy": float(contribution_denominator),
            "route_weighted_expert_nmse": float(
                contribution_numerator / contribution_denominator
            ),
            "teacher_mixture_energy": float(mixture_denominator),
            "teacher_mixture_replacement_squared_error": float(mixture_numerator),
            "teacher_mixture_replacement_nmse": float(
                mixture_numerator / mixture_denominator
            ),
            "request_contributions": request_contributions,
        }
    return metrics


def run(args: argparse.Namespace) -> dict:
    store = InterimKeepStore(args.checkpoint)
    if args.expert not in store.kept_experts(args.layer):
        raise ValueError(f"expert {args.layer}.{args.expert} is not retained")
    provenance = None
    training_request_documents = None
    validation_request_documents = None
    if args.validation_capture is not None:
        if args.training_report is None or args.validation_report is None:
            raise ValueError(
                "document-disjoint validation requires both corpus reports"
            )
        provenance = _validate_corpus_reports(
            args.training_report,
            args.validation_report,
            training_capture=args.capture,
            validation_capture=args.validation_capture,
        )
        training_request_documents = _request_documents(
            _read_json(args.training_report)
        )
        validation_request_documents = _request_documents(
            _read_json(args.validation_report)
        )
    samples = load_layer_samples(args.capture, args.layer - 1)
    validation_samples = (
        None
        if args.validation_capture is None
        else load_layer_samples(args.validation_capture, args.layer - 1)
    )
    context_count = RATE_TRANSFER_MODES[0].context_count
    block_contexts, context_meta = _contexts_for_expert(
        samples,
        args.expert,
        context_count,
        # A separate document-disjoint capture makes every row in this
        # training capture eligible for proposal statistics. Preserve the
        # internal split for standalone pilot captures.
        split=None if validation_samples is not None else 0,
        request_documents=training_request_documents,
    )
    h13, h2 = load_layer_hessians(args.hessians, args.layer)
    device = torch.device(args.device)
    source = tuple(
        store.load_matrix(args.layer, args.expert, matrix).float().to(device)
        for matrix in ("w1", "w3", "w2")
    )
    block_contexts = block_contexts.to(device)
    quantize_exl3 = _load_exl3_quantizer(args.exllamav3_root)
    quantizer_module = __import__(quantize_exl3.__module__, fromlist=["*"])
    candidates: dict[tuple[str, int], tuple[torch.Tensor, dict]] = {}
    candidate_timings: dict[tuple[str, int], float] = {}
    for matrix, source_matrix in zip(("w1", "w3", "w2"), source, strict=True):
        r_values = args.r2_values if matrix == "w2" else args.r13_values
        for r in r_values:
            mode = RATE_TRANSFER_MODES[r]
            bit_widths = mode.context_bits
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            reconstruction, coding = _quantize_full_hessian_mixed(
                source_matrix,
                block_contexts,
                bit_widths,
                matrix=matrix,
                hessian=h2 if matrix == "w2" else h13,
                layer=args.layer,
                layout="importance_ordered",
                device=device,
                quantizer_module=quantizer_module,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            candidates[(matrix, r)] = (
                reconstruction,
                {
                    **coding,
                    "weight_nmse": normalized_mse(source_matrix, reconstruction),
                },
            )
            candidate_timings[(matrix, r)] = time.perf_counter() - started

    physical_group_order = tp12_storage_group_order(
        block_contexts, RATE_TRANSFER_MODES[0]
    )
    physical_permutation = expand_group_order(physical_group_order)
    closure_inputs, _, _, _ = _selected_expert_rows(
        samples,
        args.expert,
        None,
        training_request_documents,
    )
    closure_inputs = closure_inputs.float().to(device)
    results = []
    for r13 in args.r13_values:
        for r2 in args.r2_values:
            if args.shared_only and r13 != r2:
                continue
            reconstructed = (
                candidates[("w1", r13)][0],
                candidates[("w3", r13)][0],
                candidates[("w2", r2)][0],
            )
            matrix_results = [
                candidates[("w1", r13)][1],
                candidates[("w3", r13)][1],
                candidates[("w2", r2)][1],
            ]
            physical = coupled_intermediate_permutation(
                reconstructed[0],
                reconstructed[1],
                reconstructed[2],
                physical_permutation,
            )
            canonical_output = _expert_output(closure_inputs, *reconstructed)
            physical_output = _expert_output(closure_inputs, *physical)
            permutation_closure = comparison_metrics(
                canonical_output, physical_output
            )
            if not torch.allclose(
                canonical_output, physical_output, atol=2e-4, rtol=2e-5
            ):
                raise ValueError("coupled physical permutation failed SiTU closure")
            results.append(
                {
                    "mode": f"R{r13}" if r13 == r2 else f"R{r13}/R{r2}",
                    "r13": r13,
                    "r2": r2,
                    "shared_r": r13 == r2,
                    "bit_widths_w13_low_to_high_record": list(
                        RATE_TRANSFER_MODES[r13].context_bits
                    ),
                    "bit_widths_w2_low_to_high_record": list(
                        RATE_TRANSFER_MODES[r2].context_bits
                    ),
                    "quantization_seconds_sum": sum(
                        candidate_timings[key]
                        for key in (("w1", r13), ("w3", r13), ("w2", r2))
                    ),
                    "matrix_results": matrix_results,
                    "total_trellis_bytes": sum(
                        result["trellis_descriptor"]["payload_bytes"]
                        for result in matrix_results
                    ),
                    "total_auxiliary_bytes": sum(
                        result["scale_bits"] // 8 for result in matrix_results
                    ),
                    "provisional_expert_mode_bits": 8,
                    "coupled_activation": _coupled_activation_metrics(
                        source,
                        reconstructed,
                        training_samples=samples,
                        validation_samples=validation_samples,
                        expert=args.expert,
                        device=device,
                        training_request_documents=training_request_documents,
                        validation_request_documents=validation_request_documents,
                    ),
                    "physical_permutation_closure": permutation_closure,
                }
            )
            del physical, canonical_output, physical_output

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "kind": "kquant_coupled_rate_transfer_exl3_experiment",
        "schema_version": 2,
        "source": {
            "checkpoint": str(store.root),
            "constraint": "interim EXL3 hybrid only; official model not loaded",
            "capture": str(args.capture.resolve()),
            "validation_capture": (
                None
                if args.validation_capture is None
                else str(args.validation_capture.resolve())
            ),
            "hessians": str(args.hessians.resolve()),
            "layer": args.layer,
            "expert": args.expert,
            "matrices": ["w1", "w3", "w2"],
            "tp_size": 12,
            "provenance": provenance,
        },
        "context": {
            "definition": (
                "24 equal-population 128-channel records ranked by four-channel "
                "post-SiTU energy from the experiment's training capture rows"
            ),
            **context_meta,
        },
        "candidate_space": {
            "r13_values": list(args.r13_values),
            "r2_values": list(args.r2_values),
            "shared_only": args.shared_only,
            "common_physical_permutation": True,
            "matrix_candidates_cached": True,
        },
        "results": results,
        "limitations": [
            "support counts and document-level uncertainty must authorize mode selection",
            "global scale search remains K3-like for mixed-rate modes",
            "w2 still uses teacher H2 rather than candidate-hhat covariance",
            "the provisional eight-bit (r13,r2) field is not a frozen alphabet",
            "no production TP12 decoder consumes this schema yet",
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
    parser.add_argument(
        "--hessians",
        type=Path,
        default=Path("out/k3-route-aware-pilot-v2-exl3-r4-subset.kqhess"),
    )
    parser.add_argument(
        "--validation-capture",
        type=Path,
        help=(
            "optional document-disjoint capture used only for held-out metrics; "
            "when supplied, all rows in --capture may inform ranking"
        ),
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        help="corpus report for --capture; required with --validation-capture",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        help=(
            "document-fold report for --validation-capture; required with "
            "--validation-capture"
        ),
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--r13-values", type=_parse_r_values, default=(0, 3, 6))
    parser.add_argument("--r2-values", type=_parse_r_values, default=(0, 3, 6))
    parser.add_argument("--shared-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(args.output)
    print(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
