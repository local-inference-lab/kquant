"""Test candidate-specific post-SiTU Hessians for official K3 experts.

For each requested shared ``R_r`` mode, official ``w1`` and ``w3`` are first
encoded with the captured global input Hessian.  Their reconstructed outputs
produce the actual candidate post-SiTU activations ``hhat`` on routed training
rows.  ``w2`` is then encoded against one of three covariance families:

* the captured all-route interim-teacher H2;
* an expert-local covariance from the official source activations; or
* an expert-local covariance from the quantized candidate activations.

The two local covariances are tested at fixed shrinkage strengths toward the
global H2.  All choices retain the same TP12 payload and are evaluated with
complete coupled expert output on document-disjoint validation data.  Source
weights are streamed as individual official MXFP4 tensors; the official model
is never instantiated.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from kquant.candidate_hessian import (
    covariance_comparison,
    partition_documents,
    shrink_covariance,
    subset_request_documents,
    weighted_covariance,
)
from kquant.capture import load_layer_hessians, load_layer_samples
from kquant.codec_research import normalized_mse
from kquant.mixed_exl3 import (
    RATE_TRANSFER_MODES,
    coupled_intermediate_permutation,
    expand_group_order,
    tp12_storage_group_order,
)
from kquant.source_weights import OfficialMXFP4Store
from kquant.tp_simulator import comparison_metrics, situ
from scripts.experiment_codec_transfers import _load_exl3_quantizer
from scripts.experiment_context_exl3 import (
    _parse_r_values,
    _quantize_full_hessian_mixed,
)
from scripts.experiment_context_palettes import _contexts_for_expert
from scripts.experiment_coupled_exl3 import (
    _coupled_activation_metrics,
    _selected_expert_rows,
)
from scripts.experiment_retained_w2_rate_transfer import (
    _read_json,
    _request_documents,
    _validate_corpus_reports,
)


def _parse_alphas(value: str) -> tuple[float, ...]:
    try:
        alphas = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "alphas must be comma-separated numbers"
        ) from exc
    if (
        not alphas
        or any(not math.isfinite(alpha) or not 0.0 < alpha <= 1.0 for alpha in alphas)
        or len(set(alphas)) != len(alphas)
    ):
        raise argparse.ArgumentTypeError(
            "alphas must be unique finite values in (0, 1]"
        )
    return alphas


def _middle(
    inputs: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor
) -> torch.Tensor:
    return situ(F.linear(inputs, w1), F.linear(inputs, w3))


def _weighted_middle_nmse(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    gates: torch.Tensor,
) -> float:
    weights = gates.double().square()
    numerator = (
        (candidate - reference).double().square().sum(dim=1) * weights
    ).sum()
    denominator = reference.double().square().sum(dim=1).mul(weights).sum()
    if denominator <= 0:
        raise ValueError("source middle activations have non-positive energy")
    return float(numerator / denominator)


def _variant_name(kind: str, alpha: float) -> str:
    value = f"{alpha:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{kind}_a{value}"


def _coding_evidence(coding: dict, *, source: torch.Tensor) -> dict:
    return {
        "proxy": coding["proxy"],
        "weight_nmse": None,
        "index_bits": coding["index_bits"],
        "scale_bits": coding["scale_bits"],
        "trellis_and_scale_bpw": (
            coding["index_bits"] + coding["scale_bits"]
        )
        / source.numel(),
        "trellis_descriptor": coding["trellis_descriptor"],
        "tp12_rank_trellis_bpw": coding["postpack_tp12_rank_trellis_bpw"],
        "reference_edge_roundtrip": coding["reference_edge_roundtrip"],
        "reference_state_roundtrip": coding["reference_state_roundtrip"],
        "stored_decode_vs_encoder": coding["stored_decode_vs_encoder"],
        "permutation_is_128_context_aligned": coding[
            "permutation_is_128_context_aligned"
        ],
        "scale_tensor_shapes": coding["scale_tensor_shapes"],
    }


def _quantize_matrix(
    source: torch.Tensor,
    contexts: torch.Tensor,
    *,
    matrix: str,
    r: int,
    hessian: torch.Tensor,
    layer: int,
    device: torch.device,
    quantizer_module,
) -> tuple[torch.Tensor, dict]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    reconstruction, coding = _quantize_full_hessian_mixed(
        source,
        contexts,
        RATE_TRANSFER_MODES[r].context_bits,
        matrix=matrix,
        hessian=hessian,
        layer=layer,
        layout="importance_ordered",
        device=device,
        quantizer_module=quantizer_module,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evidence = _coding_evidence(coding, source=source)
    evidence["weight_nmse"] = normalized_mse(source, reconstruction)
    evidence["quantization_seconds"] = time.perf_counter() - started
    return reconstruction, evidence


def _covariance_variants(
    global_h2: torch.Tensor,
    source_expert_h2: torch.Tensor,
    candidate_hhat: torch.Tensor,
    alphas: tuple[float, ...],
):
    yield "global_teacher", "global_teacher", 0.0, global_h2
    for alpha in alphas:
        yield (
            _variant_name("source_expert", alpha),
            "source_expert",
            alpha,
            shrink_covariance(global_h2, source_expert_h2, alpha),
        )
        yield (
            _variant_name("candidate_hhat", alpha),
            "candidate_hhat",
            alpha,
            shrink_covariance(global_h2, candidate_hhat, alpha),
        )


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    provenance = _validate_corpus_reports(
        args.training_report,
        args.validation_report,
        training_capture=args.capture,
        validation_capture=args.validation_capture,
    )
    training_report = _read_json(args.training_report)
    validation_report = _read_json(args.validation_report)
    resident_checkpoint = args.checkpoint.resolve()
    for name, report in (("training", training_report), ("validation", validation_report)):
        teacher = Path(str(report.get("model_dir", ""))).resolve()
        if teacher != resident_checkpoint:
            raise ValueError(f"{name} corpus teacher {teacher} != {resident_checkpoint}")
    training_request_documents = _request_documents(training_report)
    validation_request_documents = _request_documents(validation_report)
    training_documents = [
        str(document["document_hash"])
        for document in training_report["documents"]
    ]
    fit_documents, confirmation_documents = partition_documents(
        training_documents,
        modulus=args.fit_modulus,
        confirmation_index=args.confirmation_index,
    )
    fit_request_documents = subset_request_documents(
        training_request_documents, fit_documents
    )
    confirmation_request_documents = subset_request_documents(
        training_request_documents, confirmation_documents
    )

    samples = load_layer_samples(args.capture, args.layer - 1)
    validation_samples = load_layer_samples(args.validation_capture, args.layer - 1)
    contexts, context_meta = _contexts_for_expert(
        samples,
        args.expert,
        RATE_TRANSFER_MODES[0].context_count,
        split=None,
        request_documents=fit_request_documents,
    )
    fit_inputs, fit_gates, _fit_mixture, fit_steps = _selected_expert_rows(
        samples,
        args.expert,
        None,
        fit_request_documents,
    )
    if not fit_inputs.shape[0]:
        raise ValueError(f"expert {args.expert} has no routed covariance-fit inputs")
    (
        confirmation_inputs,
        _confirmation_gates,
        _confirmation_mixture,
        confirmation_steps,
    ) = _selected_expert_rows(
        samples,
        args.expert,
        None,
        confirmation_request_documents,
    )
    if not confirmation_inputs.shape[0]:
        raise ValueError(f"expert {args.expert} has no routed confirmation inputs")
    validation_inputs, _validation_gates, _validation_mixture, validation_steps = (
        _selected_expert_rows(
            validation_samples,
            args.expert,
            None,
            validation_request_documents,
        )
    )
    if not validation_inputs.shape[0]:
        raise ValueError(f"expert {args.expert} has no routed validation inputs")

    hessian_manifest = _read_json(args.hessians / "manifest.json")
    if Path(hessian_manifest.get("source_capture", "")).resolve() != args.capture.resolve():
        raise ValueError("Hessian bundle does not come from the training capture")
    if hessian_manifest.get("sample_split") != "all":
        raise ValueError("fit-document Hessians must retain every row in each fit document")
    expected_request_steps = sorted(fit_request_documents)
    if hessian_manifest.get("request_step_filter") != expected_request_steps:
        raise ValueError(
            "Hessian bundle request-step filter does not match the covariance-fit documents"
        )
    h13, global_h2 = load_layer_hessians(args.hessians, args.layer)
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    device = torch.device(args.device)
    source = tuple(
        store.load_matrix(args.layer, args.expert, matrix).float().to(device)
        for matrix in ("w1", "w3", "w2")
    )
    contexts = contexts.to(device)
    fit_inputs_device = fit_inputs.float().to(device)
    fit_gates_device = fit_gates.float().to(device)
    source_middle = _middle(fit_inputs_device, source[0], source[1])
    source_expert_h2, gate_square_sum = weighted_covariance(
        source_middle,
        fit_gates_device.square(),
        device=device,
        chunk_rows=args.covariance_chunk_rows,
    )
    quantize_exl3 = _load_exl3_quantizer(args.exllamav3_root)
    quantizer_module = __import__(quantize_exl3.__module__, fromlist=["*"])

    physical_group_order = tp12_storage_group_order(
        contexts, RATE_TRANSFER_MODES[0]
    )
    physical_permutation = expand_group_order(physical_group_order)
    closure_inputs = fit_inputs_device
    modes = []
    global_byte_contract = None
    for r in args.r_values:
        w1_reconstruction, w1_evidence = _quantize_matrix(
            source[0],
            contexts,
            matrix="w1",
            r=r,
            hessian=h13,
            layer=args.layer,
            device=device,
            quantizer_module=quantizer_module,
        )
        w3_reconstruction, w3_evidence = _quantize_matrix(
            source[1],
            contexts,
            matrix="w3",
            r=r,
            hessian=h13,
            layer=args.layer,
            device=device,
            quantizer_module=quantizer_module,
        )
        candidate_middle = _middle(
            fit_inputs_device, w1_reconstruction, w3_reconstruction
        )
        candidate_h2, candidate_gate_square_sum = weighted_covariance(
            candidate_middle,
            fit_gates_device.square(),
            device=device,
            chunk_rows=args.covariance_chunk_rows,
        )
        if not math.isclose(
            candidate_gate_square_sum, gate_square_sum, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("candidate/source covariance weight sums differ")

        variants = []
        mode_byte_contract = None
        for name, hessian_kind, alpha, hessian in _covariance_variants(
            global_h2,
            source_expert_h2,
            candidate_h2,
            args.alphas,
        ):
            w2_reconstruction, w2_evidence = _quantize_matrix(
                source[2],
                contexts,
                matrix="w2",
                r=r,
                hessian=hessian,
                layer=args.layer,
                device=device,
                quantizer_module=quantizer_module,
            )
            reconstructed = (
                w1_reconstruction,
                w3_reconstruction,
                w2_reconstruction,
            )
            physical = coupled_intermediate_permutation(
                *reconstructed, physical_permutation
            )
            canonical_output = F.linear(
                _middle(closure_inputs, reconstructed[0], reconstructed[1]),
                reconstructed[2],
            )
            physical_output = F.linear(
                _middle(closure_inputs, physical[0], physical[1]), physical[2]
            )
            permutation_closure = comparison_metrics(
                canonical_output, physical_output
            )
            if not torch.allclose(
                canonical_output, physical_output, atol=2e-4, rtol=2e-5
            ):
                raise ValueError("common physical permutation failed SiTU closure")
            byte_contract = (
                sum(
                    evidence["trellis_descriptor"]["payload_bytes"]
                    for evidence in (w1_evidence, w3_evidence, w2_evidence)
                ),
                sum(
                    evidence["scale_bits"] // 8
                    for evidence in (w1_evidence, w3_evidence, w2_evidence)
                ),
            )
            if mode_byte_contract not in (None, byte_contract):
                raise ValueError("Hessian variants changed the exact codec bytes")
            if global_byte_contract not in (None, byte_contract):
                raise ValueError("rate modes changed the exact codec bytes")
            mode_byte_contract = byte_contract
            global_byte_contract = byte_contract
            variants.append(
                {
                    "name": name,
                    "hessian_kind": hessian_kind,
                    "local_covariance_alpha": alpha,
                    "w2": w2_evidence,
                    "total_trellis_bytes": byte_contract[0],
                    "total_auxiliary_bytes": byte_contract[1],
                    "coupled_activation": _coupled_activation_metrics(
                        source,
                        reconstructed,
                        training_samples=samples,
                        validation_samples=validation_samples,
                        expert=args.expert,
                        device=device,
                        training_request_documents=fit_request_documents,
                        training_confirmation_documents=(
                            confirmation_request_documents
                        ),
                        validation_request_documents=validation_request_documents,
                    ),
                    "physical_permutation_closure": permutation_closure,
                }
            )
            del (
                w2_reconstruction,
                physical,
                canonical_output,
                physical_output,
                hessian,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        modes.append(
            {
                "mode": f"R{r}",
                "r": r,
                "w1": w1_evidence,
                "w3": w3_evidence,
                "training_middle": {
                    "rows": int(fit_inputs.shape[0]),
                    "gate_square_sum": gate_square_sum,
                    "candidate_to_source_weighted_nmse": _weighted_middle_nmse(
                        source_middle, candidate_middle, fit_gates_device
                    ),
                    "source_expert_covariance_vs_global": covariance_comparison(
                        global_h2, source_expert_h2
                    ),
                    "candidate_hhat_covariance_vs_global": covariance_comparison(
                        global_h2, candidate_h2
                    ),
                    "candidate_hhat_covariance_vs_source_expert": covariance_comparison(
                        source_expert_h2, candidate_h2
                    ),
                    "local_covariance_rank_upper_bound": min(
                        int(fit_inputs.shape[0]), int(candidate_middle.shape[1])
                    ),
                },
                "variants": variants,
                "exact_byte_contract": {
                    "total_trellis_bytes": mode_byte_contract[0],
                    "total_auxiliary_bytes": mode_byte_contract[1],
                    "all_hessian_variants_equal": True,
                },
            }
        )
        del w1_reconstruction, w3_reconstruction, candidate_middle, candidate_h2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "kind": "kquant_candidate_hhat_experiment",
        "schema_version": 2,
        "constraint": (
            "resident teacher and captures are interim EXL3; three official "
            "expert matrices are streamed individually and the official model "
            "is never instantiated"
        ),
        "signature": {
            "checkpoint": str(resident_checkpoint),
            "resident_teacher_checkpoint": str(resident_checkpoint),
            "source_checkpoint": str(store.root),
            "source_revision": store.revision,
            "capture": str(args.capture.resolve()),
            "validation_capture": str(args.validation_capture.resolve()),
            "hessians": str(args.hessians.resolve()),
            "layer": args.layer,
            "expert": args.expert,
            "r_values": list(args.r_values),
            "alphas": list(args.alphas),
            "tp_size": 12,
            "document_split": {
                "algorithm": "blake2b-64(document_hash) modulo",
                "modulus": args.fit_modulus,
                "confirmation_index": args.confirmation_index,
                "fit_documents": len(fit_documents),
                "confirmation_documents": len(confirmation_documents),
                "fit_request_steps": expected_request_steps,
            },
            "provenance": provenance,
        },
        "selection_contract": {
            "covariance_fit": (
                "fit-document routed rows only, weighted by applied gate squared"
            ),
            "confirmation_usage": (
                "held out from context ranking and every encoding Hessian"
            ),
            "validation_usage": "reporting only",
            "global_h2": (
                "captured interim-teacher all-route layer covariance on fit documents"
            ),
            "source_expert_h2": "official source w1/w3 post-SiTU expert-local covariance",
            "candidate_hhat": "encoded w1/w3 post-SiTU expert-local covariance",
        },
        "context": context_meta,
        "covariance_support": {
            "fit_rows": int(fit_inputs.shape[0]),
            "fit_documents_routed": int(torch.unique(fit_steps).numel()),
            "fit_documents_corpus": len(fit_documents),
            "confirmation_rows": int(confirmation_inputs.shape[0]),
            "confirmation_documents_routed": int(
                torch.unique(confirmation_steps).numel()
            ),
            "confirmation_documents_corpus": len(confirmation_documents),
            "validation_rows": int(validation_inputs.shape[0]),
            "validation_documents": int(torch.unique(validation_steps).numel()),
            "gate_square_sum": gate_square_sum,
        },
        "modes": modes,
        "exact_byte_contract": {
            "total_trellis_bytes": global_byte_contract[0],
            "total_auxiliary_bytes": global_byte_contract[1],
            "all_modes_and_hessian_variants_equal": True,
        },
        "complete": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(args.output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/models/Kimi-K3-EXL3-3p09-serve"),
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--validation-capture", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--hessians", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--r-values", type=_parse_r_values, required=True)
    parser.add_argument(
        "--alphas",
        type=_parse_alphas,
        default=(0.125, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--fit-modulus", type=int, default=4)
    parser.add_argument("--confirmation-index", type=int, default=0)
    parser.add_argument("--covariance-chunk-rows", type=int, default=256)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument(
        "--official-revision",
        default="c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= 92:
        parser.error("--layer must be in 1..92")
    if not 0 <= args.expert < 896:
        parser.error("--expert must be in 0..895")
    if args.covariance_chunk_rows <= 0:
        parser.error("--covariance-chunk-rows must be positive")
    if args.fit_modulus < 2:
        parser.error("--fit-modulus must be at least 2")
    if not 0 <= args.confirmation_index < args.fit_modulus:
        parser.error("--confirmation-index must be in the fit modulus")
    return args


def main() -> None:
    payload = run(parse_args())
    print(
        json.dumps(
            {
                "layer": payload["signature"]["layer"],
                "expert": payload["signature"]["expert"],
                "modes": len(payload["modes"]),
                "variants": sum(len(mode["variants"]) for mode in payload["modes"]),
                "complete": payload["complete"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
