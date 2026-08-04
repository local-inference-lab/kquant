#!/usr/bin/env python3
"""Screen K2-only SQG-Cheb tail laws on official decoded MXFP4 experts.

The graph, transform, scale search, dense-H BlockLDLQ traversal, and K3/K4
rank law are frozen.  This study changes only the continuous K2 rank-to-value
law before its single finite-E4M3 rounding step.  It is a donor-endpoint
screen; finalists still require complete rate-shifted R1/R2 counterfactual encodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from kquant import constants as C
from kquant.capture import load_layer_hessians, load_layer_samples
from kquant.exl3_loader import load_qsrt_encoder
from kquant.sqg_e4m3 import (
    sqg_cheb_normal_rank_e4m3_bytes,
    sqg_e4m3_bytes_from_rank_lut,
    sqg_k2_tail_compressed_rank_e4m3_bytes,
)
from kquant.sqg_quantizer import install_sqg_quantizer
from kquant.qsrt_candidates import (
    activation_block_contexts,
    build_expert_hessians,
    select_expert_rows,
)
from kquant.source_weights import OfficialMXFP4Store
from scripts.validate_qsrt_codebooks import (
    SQG_CHEB_NORMAL_E4M3,
    _aggregate_matrix_evidence,
    _atomic_write,
    _compact_functional_evidence,
    _coupled_activation_metrics,
    _fit_contract,
    _quantize_uniform_matrix,
    _read_json,
    _request_mask,
    _validate_corpus_contract,
)


DEFAULT_LAWS = (
    "normal",
    "asinh:2.5",
    "asinh:3",
    "asinh:4",
    "asinh:5",
    "asinh:6",
    "asinh:8",
    "rational:0.005",
    "rational:0.01",
    "rational:0.02",
    "rational:0.04",
)


@dataclass(frozen=True)
class LawSpec:
    name: str
    family: str
    parameter: float | None


def _parse_law(value: str) -> LawSpec:
    value = value.strip().lower()
    if value in ("normal", "normal-clipped", "r44-full", "r44-clipped"):
        return LawSpec(value, value, None)
    if value == "q33":
        return LawSpec("q33", "outer-slope", 33.0 / 32.0)
    try:
        family, raw_parameter = value.split(":", 1)
        parameter = float(raw_parameter)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "K2 laws must be a fixed law, q33, asinh:A, rational:LAMBDA, "
            "or outer-slope:SLOPE"
        ) from exc
    if family not in ("asinh", "rational", "outer-slope"):
        raise argparse.ArgumentTypeError(
            "K2 law family must be asinh, rational, or outer-slope"
        )
    if not math.isfinite(parameter) or parameter <= 0:
        raise argparse.ArgumentTypeError("K2 law parameter must be positive and finite")
    return LawSpec(f"{family}:{parameter:g}", family, parameter)


def _parse_laws(value: str) -> tuple[LawSpec, ...]:
    laws = tuple(_parse_law(item) for item in value.split(",") if item.strip())
    if not laws or len({law.name for law in laws}) != len(laws):
        raise argparse.ArgumentTypeError("K2 laws must be nonempty and unique")
    if laws[0].name != "normal":
        raise argparse.ArgumentTypeError("the first K2 law must be normal")
    return laws


def _parse_experts(value: str) -> tuple[int, ...]:
    try:
        experts = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("experts must be comma-separated integers") from exc
    if (
        not experts
        or len(set(experts)) != len(experts)
        or any(not 0 <= expert < C.NUM_EXPERTS for expert in experts)
    ):
        raise argparse.ArgumentTypeError("experts must be unique IDs in 0..895")
    return experts


def _lut_metadata(spec: LawSpec, rank_lut: torch.Tensor) -> dict[str, object]:
    payload = bytes(rank_lut.tolist())
    values = rank_lut.view(torch.float8_e4m3fn).float()
    return {
        "name": spec.name,
        "family": spec.family,
        "parameter": spec.parameter,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "distinct_e4m3_labels": int(torch.unique(rank_lut).numel()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "canonical_zero": torch.unique(rank_lut[values == 0]).tolist() == [0],
        "monotone": bool(torch.all(values[1:] >= values[:-1])),
    }


def _summarize(payload: dict) -> dict[str, object]:
    laws = [item["name"] for item in payload["signature"]["laws"]]
    results = list(payload["results"].values())
    summary: dict[str, object] = {}
    for law in laws:
        candidates = [result["candidates"][law] for result in results]
        matrix_rows = [
            matrix
            for candidate in candidates
            for matrix in candidate["matrix_evidence"]
        ]
        raw_error = sum(float(row["weight_squared_error"]) for row in matrix_rows)
        raw_energy = sum(float(row["weight_reference_energy"]) for row in matrix_rows)
        dense_error = sum(
            float(row["captured_dense_h_numerator"]) for row in matrix_rows
        )
        dense_energy = sum(
            float(row["captured_dense_h_denominator"]) for row in matrix_rows
        )
        folds: dict[str, object] = {}
        for fold in ("train", "confirmation", "validation"):
            metrics = [
                candidate["routed_function"][fold]
                for candidate in candidates
                if candidate["routed_function"].get(fold) is not None
            ]
            folds[fold] = {
                "experts": len(metrics),
                "route_weighted_squared_error": sum(
                    float(metric["route_weighted_squared_error"]) for metric in metrics
                ),
                "route_weighted_reference_energy": sum(
                    float(metric["route_weighted_reference_energy"]) for metric in metrics
                ),
            }
            fold_result = folds[fold]
            fold_result["route_weighted_expert_nmse"] = (
                fold_result["route_weighted_squared_error"]
                / fold_result["route_weighted_reference_energy"]
            )
        summary[law] = {
            "experts": len(candidates),
            "weight_nmse": raw_error / raw_energy,
            "captured_dense_h_nmse": dense_error / dense_energy,
            "folds": folds,
        }

    baseline = summary["normal"]
    for law in laws:
        row = summary[law]
        comparisons: dict[str, object] = {
            "weight_sse_ratio_vs_normal": row["weight_nmse"] / baseline["weight_nmse"],
            "dense_h_sse_ratio_vs_normal": (
                row["captured_dense_h_nmse"] / baseline["captured_dense_h_nmse"]
            ),
        }
        for fold in ("train", "confirmation", "validation"):
            ratios = []
            wins = 0
            for result in results:
                trial = result["candidates"][law]["routed_function"].get(fold)
                base = result["candidates"]["normal"]["routed_function"].get(fold)
                if trial is None or base is None:
                    continue
                ratio = float(trial["route_weighted_squared_error"]) / float(
                    base["route_weighted_squared_error"]
                )
                ratios.append(ratio)
                wins += ratio < 1.0
            comparisons[f"{fold}_routed_sse_ratio_vs_normal"] = (
                row["folds"][fold]["route_weighted_squared_error"]
                / baseline["folds"][fold]["route_weighted_squared_error"]
            )
            comparisons[f"{fold}_median_expert_ratio_vs_normal"] = float(
                np.median(ratios)
            )
            comparisons[f"{fold}_wins_vs_normal"] = wins
        row["comparison_vs_normal"] = comparisons
    return summary


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the production K2-law study requires CUDA")
    torch.cuda.set_device(device)
    training_report = _read_json(args.training_report)
    validation_report = _read_json(args.validation_report)
    provenance, training_requests, validation_requests = _validate_corpus_contract(
        args, training_report, validation_report
    )
    partition = _fit_contract(args, training_requests)
    training_samples = load_layer_samples(args.capture, args.layer - 1)
    validation_samples = load_layer_samples(args.validation_capture, args.layer - 1)
    global_h13, global_h2 = load_layer_hessians(args.hessians, args.layer)

    rank_luts = {
        law.name: sqg_k2_tail_compressed_rank_e4m3_bytes(
            law.family, law.parameter
        )
        for law in args.laws
    }
    base_rank_lut = sqg_cheb_normal_rank_e4m3_bytes()
    if not torch.equal(rank_luts["normal"], base_rank_lut):
        raise AssertionError("normal K2 control changed the SQG-Cheb rank law")
    luts = {
        law.name: {
            2: sqg_e4m3_bytes_from_rank_lut(
                2, rank_luts[law.name], device=device
            )
        }
        for law in args.laws
    }

    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    signature = {
        "resident_teacher": str(args.checkpoint.resolve()),
        "official_source": str(store.root.resolve()),
        "official_revision": args.official_revision,
        "capture": str(args.capture.resolve()),
        "validation_capture": str(args.validation_capture.resolve()),
        "hessians": str(args.hessians.resolve()),
        "training_report": str(args.training_report.resolve()),
        "validation_report": str(args.validation_report.resolve()),
        "layer": args.layer,
        "experts": list(args.experts),
        "laws": [_lut_metadata(law, rank_luts[law.name]) for law in args.laws],
        "ldlq_tf32": args.ldlq_tf32,
        "tailbite_context": args.tailbite_context,
        "tp_size": 12,
        "provenance": provenance,
    }
    if args.resume:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        payload = _read_json(args.output)
        if payload.get("signature") != signature:
            raise ValueError("resume output does not match this K2-law study")
        if payload.get("complete"):
            return payload
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        payload = {
            "kind": "kquant_qsrt_k2_rank_law_study",
            "schema_version": 1,
            "signature": signature,
            "controlled_variables": {
                "changed": "K2 rank-to-finite-E4M3 staircase only",
                "frozen": [
                    "SQG L16 graph and branch labelling",
                    "SQG-Cheb normal K3/K4 staircase",
                    "Hadamard/sign transforms",
                    "scale search and FP16 scale persistence",
                    "dense-H BlockLDLQ traversal",
                ],
            },
            "results": {},
            "skipped": {},
            "complete": False,
        }
        _atomic_write(args.output, payload)

    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    completed = set(payload["results"]) | set(payload["skipped"])
    for expert in args.experts:
        key = str(expert)
        if key in completed:
            continue
        all_rows = select_expert_rows(training_samples, expert, partition.all)
        fit_mask = _request_mask(all_rows.request_steps, partition.fit)
        validation_rows = select_expert_rows(
            validation_samples, expert, validation_requests
        )
        if int(fit_mask.sum()) == 0 or validation_rows.rows == 0:
            payload["skipped"][key] = {"reason": "no fit or validation rows"}
            _atomic_write(args.output, payload)
            continue

        source = tuple(
            store.load_matrix(args.layer, expert, matrix, device=device)
            .float()
            .contiguous()
            for matrix in C.EXPERT_MATRICES
        )
        fit_inputs = all_rows.inputs[fit_mask].float().to(device=device)
        fit_gates = all_rows.gates[fit_mask].float().to(device=device)
        with torch.inference_mode():
            source_middle = torch.nn.functional.silu(
                torch.nn.functional.linear(fit_inputs, source[0])
            ) * torch.nn.functional.linear(fit_inputs, source[1])
            contexts, block_scores = activation_block_contexts(
                source_middle, fit_gates
            )
            h13, h2, covariance = build_expert_hessians(
                fit_inputs,
                fit_gates,
                source_middle,
                global_h13=global_h13,
                global_h2=global_h2,
                device=device,
            )
        contexts = contexts.to(device=device, dtype=torch.long)
        candidates: dict[str, object] = {}
        for law in args.laws:
            reconstructions = []
            matrix_evidence = []
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for matrix, source_matrix in zip(
                C.EXPERT_MATRICES, source, strict=True
            ):
                reconstruction, evidence = _quantize_uniform_matrix(
                    source_matrix,
                    contexts,
                    matrix=matrix,
                    bits=2,
                    codebook=SQG_CHEB_NORMAL_E4M3,
                    hessian=h2 if matrix == "w2" else h13,
                    layer=args.layer,
                    expert=expert,
                    device=device,
                    quantizer_module=quantizer_module,
                    ldlq_tf32=args.ldlq_tf32,
                    tailbite_context=args.tailbite_context,
                    external_sqg_luts=luts[law.name],
                )
                evidence["rank_law"] = law.name
                reconstructions.append(reconstruction)
                matrix_evidence.append(evidence)
            torch.cuda.synchronize(device)
            functional = _coupled_activation_metrics(
                source,
                tuple(reconstructions),
                training_samples=training_samples,
                validation_samples=validation_samples,
                expert=expert,
                device=device,
                training_request_documents=partition.fit,
                training_confirmation_documents=partition.confirmation,
                validation_request_documents=validation_requests,
            )
            candidates[law.name] = {
                "law": law.name,
                "matrix_evidence": matrix_evidence,
                "aggregate": _aggregate_matrix_evidence(matrix_evidence),
                "routed_function": functional,
                "quantization_and_scoring_seconds": time.perf_counter() - started,
            }
            del reconstructions
            torch.cuda.empty_cache()

        if args.compact:
            _compact_functional_evidence(candidates)
        payload["results"][key] = {
            "covariance": covariance,
            "context": {
                "basis": "official_source_post_situ_fit_documents",
                "score_min": float(block_scores.min()),
                "score_median": float(block_scores.median()),
                "score_max": float(block_scores.max()),
            },
            "candidates": candidates,
        }
        payload["summary"] = _summarize(payload)
        _atomic_write(args.output, payload)
        del source, fit_inputs, fit_gates, source_middle, h13, h2, contexts
        torch.cuda.empty_cache()

    payload["summary"] = _summarize(payload)
    payload["complete"] = True
    _atomic_write(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/models/Kimi-K3-EXL3-3p09-serve"),
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=Path("/data/kquant/k3-codec-diverse-train-v2.kqcapture"),
    )
    parser.add_argument(
        "--validation-capture",
        type=Path,
        default=Path("/data/kquant/k3-codec-diverse-validation-v3-128k.kqcapture"),
    )
    parser.add_argument(
        "--hessians",
        type=Path,
        default=Path("/data/kquant/k3-codec-diverse-train-v2-fit3of4-all.kqhess"),
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=Path("out/k3-codec-diverse-train-v2-corpus.json"),
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("out/k3-codec-diverse-validation-v3-128k-corpus.json"),
    )
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", type=_parse_experts, required=True)
    parser.add_argument(
        "--laws", type=_parse_laws, default=tuple(map(_parse_law, DEFAULT_LAWS))
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ldlq-tf32", action="store_true")
    parser.add_argument("--tailbite-context", type=int, default=128)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.layer not in C.MOE_LAYERS:
        parser.error("--layer must be in 1..92")
    if not 1 <= args.tailbite_context <= 128:
        parser.error("--tailbite-context must be in 1..128")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
