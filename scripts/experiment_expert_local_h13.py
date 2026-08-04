"""Test expert-local input Hessians for official K3 ``w1``/``w3``.

The context order and all covariances use only the deterministic fit-document
fold.  ``w1`` and ``w3`` share either the layer-global H13 or a fixed
shrinkage blend toward the routed expert's conditional input covariance.
``w2`` is held fixed at the source-expert H2 blend selected by the preceding
study.  Confirmation and external-validation documents never enter an
encoding Hessian.  Official weights are streamed one expert matrix at a time;
the official model is never instantiated.
"""

from __future__ import annotations

import argparse
import json
import math
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
from kquant.mixed_exl3 import (
    RATE_TRANSFER_MODES,
    coupled_intermediate_permutation,
    expand_group_order,
    tp12_storage_group_order,
)
from kquant.source_weights import OfficialMXFP4Store
from kquant.tp_simulator import comparison_metrics
from scripts.experiment_candidate_hhat import (
    _middle,
    _parse_alphas,
    _quantize_matrix,
    _variant_name,
    _weighted_middle_nmse,
)
from scripts.experiment_codec_transfers import _load_exl3_quantizer
from scripts.experiment_context_exl3 import _parse_r_values
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


def _h13_variants(
    global_h13: torch.Tensor,
    expert_h13: torch.Tensor,
    alphas: tuple[float, ...],
):
    yield "global_teacher", 0.0, global_h13
    for alpha in alphas:
        yield (
            _variant_name("source_expert", alpha),
            alpha,
            shrink_covariance(global_h13, expert_h13, alpha),
        )


def _fit_document_contract(args: argparse.Namespace, training_report: dict):
    request_documents = _request_documents(training_report)
    documents = [
        str(document["document_hash"])
        for document in training_report["documents"]
    ]
    fit, confirmation = partition_documents(
        documents,
        modulus=args.fit_modulus,
        confirmation_index=args.confirmation_index,
    )
    fit_requests = subset_request_documents(request_documents, fit)
    confirmation_requests = subset_request_documents(request_documents, confirmation)
    manifest = _read_json(args.hessians / "manifest.json")
    if Path(manifest.get("source_capture", "")).resolve() != args.capture.resolve():
        raise ValueError("Hessian bundle does not come from the training capture")
    if manifest.get("sample_split") != "all":
        raise ValueError("fit-document Hessians must retain every fit-document row")
    if manifest.get("request_step_filter") != sorted(fit_requests):
        raise ValueError("Hessian request-step filter differs from fit documents")
    return fit, confirmation, fit_requests, confirmation_requests


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
    fit_documents, confirmation_documents, fit_requests, confirmation_requests = (
        _fit_document_contract(args, training_report)
    )
    validation_requests = _request_documents(validation_report)

    samples = load_layer_samples(args.capture, args.layer - 1)
    validation_samples = load_layer_samples(args.validation_capture, args.layer - 1)
    contexts, context_meta = _contexts_for_expert(
        samples,
        args.expert,
        RATE_TRANSFER_MODES[0].context_count,
        split=None,
        request_documents=fit_requests,
    )
    fit_inputs, fit_gates, _fit_mixture, fit_steps = _selected_expert_rows(
        samples, args.expert, None, fit_requests
    )
    confirmation_inputs, _gates, _mixture, confirmation_steps = (
        _selected_expert_rows(samples, args.expert, None, confirmation_requests)
    )
    validation_inputs, _gates, _mixture, validation_steps = _selected_expert_rows(
        validation_samples, args.expert, None, validation_requests
    )
    if not fit_inputs.shape[0] or not confirmation_inputs.shape[0] or not validation_inputs.shape[0]:
        raise ValueError("expert lacks fit, confirmation, or validation support")

    global_h13, global_h2 = load_layer_hessians(args.hessians, args.layer)
    device = torch.device(args.device)
    fit_inputs_device = fit_inputs.float().to(device)
    fit_gates_device = fit_gates.float().to(device)
    expert_h13, h13_weight_sum = weighted_covariance(
        fit_inputs_device,
        fit_gates_device.square(),
        device=device,
        chunk_rows=args.covariance_chunk_rows,
    )

    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    source = tuple(
        store.load_matrix(args.layer, args.expert, matrix).float().to(device)
        for matrix in ("w1", "w3", "w2")
    )
    source_middle = _middle(fit_inputs_device, source[0], source[1])
    source_expert_h2, h2_weight_sum = weighted_covariance(
        source_middle,
        fit_gates_device.square(),
        device=device,
        chunk_rows=args.covariance_chunk_rows,
    )
    if not math.isclose(h13_weight_sum, h2_weight_sum, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("H13 and H2 expert covariance weights differ")
    fixed_w2_hessian = shrink_covariance(
        global_h2, source_expert_h2, args.w2_local_alpha
    )

    quantize_exl3 = _load_exl3_quantizer(args.exllamav3_root)
    quantizer_module = __import__(quantize_exl3.__module__, fromlist=["*"])
    contexts = contexts.to(device)
    if len(args.r_values) != 1:
        raise ValueError("expert-local H13 study expects exactly one selected R mode")
    r = args.r_values[0]
    w2_reconstruction, w2_evidence = _quantize_matrix(
        source[2],
        contexts,
        matrix="w2",
        r=r,
        hessian=fixed_w2_hessian,
        layer=args.layer,
        device=device,
        quantizer_module=quantizer_module,
    )

    physical_group_order = tp12_storage_group_order(
        contexts, RATE_TRANSFER_MODES[0]
    )
    physical_permutation = expand_group_order(physical_group_order)
    variants = []
    byte_contract = None
    for name, alpha, hessian in _h13_variants(
        global_h13, expert_h13, args.h13_alphas
    ):
        w1_reconstruction, w1_evidence = _quantize_matrix(
            source[0],
            contexts,
            matrix="w1",
            r=r,
            hessian=hessian,
            layer=args.layer,
            device=device,
            quantizer_module=quantizer_module,
        )
        w3_reconstruction, w3_evidence = _quantize_matrix(
            source[1],
            contexts,
            matrix="w3",
            r=r,
            hessian=hessian,
            layer=args.layer,
            device=device,
            quantizer_module=quantizer_module,
        )
        reconstructed = (w1_reconstruction, w3_reconstruction, w2_reconstruction)
        physical = coupled_intermediate_permutation(
            *reconstructed, physical_permutation
        )
        canonical_output = F.linear(
            _middle(fit_inputs_device, reconstructed[0], reconstructed[1]),
            reconstructed[2],
        )
        physical_output = F.linear(
            _middle(fit_inputs_device, physical[0], physical[1]), physical[2]
        )
        closure = comparison_metrics(canonical_output, physical_output)
        if not torch.allclose(
            canonical_output, physical_output, atol=2e-4, rtol=2e-5
        ):
            raise ValueError("common physical permutation failed SiTU closure")
        current_byte_contract = (
            sum(
                evidence["trellis_descriptor"]["payload_bytes"]
                for evidence in (w1_evidence, w3_evidence, w2_evidence)
            ),
            sum(
                evidence["scale_bits"] // 8
                for evidence in (w1_evidence, w3_evidence, w2_evidence)
            ),
        )
        if byte_contract not in (None, current_byte_contract):
            raise ValueError("H13 variants changed exact codec bytes")
        byte_contract = current_byte_contract
        candidate_middle = _middle(
            fit_inputs_device, w1_reconstruction, w3_reconstruction
        )
        variants.append(
            {
                "name": name,
                "local_covariance_alpha": alpha,
                "w1": w1_evidence,
                "w3": w3_evidence,
                "candidate_middle_to_source_weighted_nmse": _weighted_middle_nmse(
                    source_middle, candidate_middle, fit_gates_device
                ),
                "total_trellis_bytes": current_byte_contract[0],
                "total_auxiliary_bytes": current_byte_contract[1],
                "coupled_activation": _coupled_activation_metrics(
                    source,
                    reconstructed,
                    training_samples=samples,
                    validation_samples=validation_samples,
                    expert=args.expert,
                    device=device,
                    training_request_documents=fit_requests,
                    training_confirmation_documents=confirmation_requests,
                    validation_request_documents=validation_requests,
                ),
                "physical_permutation_closure": closure,
            }
        )
        del (
            w1_reconstruction,
            w3_reconstruction,
            reconstructed,
            physical,
            canonical_output,
            physical_output,
            candidate_middle,
            hessian,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert byte_contract is not None
    payload = {
        "kind": "kquant_expert_local_h13_experiment",
        "schema_version": 1,
        "constraint": (
            "resident teacher and captures are interim EXL3; official matrices "
            "are streamed individually and the official model is never instantiated"
        ),
        "signature": {
            "checkpoint": str(resident_checkpoint),
            "source_checkpoint": str(store.root),
            "source_revision": store.revision,
            "capture": str(args.capture.resolve()),
            "validation_capture": str(args.validation_capture.resolve()),
            "hessians": str(args.hessians.resolve()),
            "layer": args.layer,
            "expert": args.expert,
            "r": r,
            "h13_alphas": list(args.h13_alphas),
            "w2_local_alpha": args.w2_local_alpha,
            "tp_size": 12,
            "document_split": {
                "algorithm": "blake2b-64(document_hash) modulo",
                "modulus": args.fit_modulus,
                "confirmation_index": args.confirmation_index,
                "fit_documents": len(fit_documents),
                "confirmation_documents": len(confirmation_documents),
                "fit_request_steps": sorted(fit_requests),
            },
            "provenance": provenance,
        },
        "selection_contract": {
            "covariance_fit": "fit-document routed rows only",
            "confirmation_usage": "held out from context ranking and every Hessian",
            "validation_usage": "reporting only",
            "w2_hessian": f"source-expert H2 alpha {args.w2_local_alpha:g}",
        },
        "context": context_meta,
        "covariance_support": {
            "fit_rows": int(fit_inputs.shape[0]),
            "fit_documents_routed": int(torch.unique(fit_steps).numel()),
            "confirmation_rows": int(confirmation_inputs.shape[0]),
            "confirmation_documents_routed": int(
                torch.unique(confirmation_steps).numel()
            ),
            "validation_rows": int(validation_inputs.shape[0]),
            "validation_documents_routed": int(torch.unique(validation_steps).numel()),
            "gate_square_sum": h13_weight_sum,
        },
        "covariance_diagnostics": {
            "expert_h13_vs_global": covariance_comparison(global_h13, expert_h13),
            "source_expert_h2_vs_global": covariance_comparison(
                global_h2, source_expert_h2
            ),
        },
        "mode": f"R{r}",
        "r": r,
        "w2": w2_evidence,
        "variants": variants,
        "exact_byte_contract": {
            "total_trellis_bytes": byte_contract[0],
            "total_auxiliary_bytes": byte_contract[1],
            "all_h13_variants_equal": True,
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
        "--checkpoint", type=Path, default=Path("/models/Kimi-K3-EXL3-3p09-serve")
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
        "--h13-alphas",
        type=_parse_alphas,
        default=(0.125, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--w2-local-alpha", type=float, default=0.75)
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
    if len(args.r_values) != 1:
        parser.error("--r-values must contain exactly one selected mode")
    if not math.isfinite(args.w2_local_alpha) or not 0 < args.w2_local_alpha <= 1:
        parser.error("--w2-local-alpha must be in (0, 1]")
    if args.fit_modulus < 2 or not 0 <= args.confirmation_index < args.fit_modulus:
        parser.error("invalid fit/confirmation partition")
    if args.covariance_chunk_rows <= 0:
        parser.error("--covariance-chunk-rows must be positive")
    return args


def main() -> None:
    payload = run(parse_args())
    print(
        json.dumps(
            {
                "layer": payload["signature"]["layer"],
                "expert": payload["signature"]["expert"],
                "variants": len(payload["variants"]),
                "complete": payload["complete"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
