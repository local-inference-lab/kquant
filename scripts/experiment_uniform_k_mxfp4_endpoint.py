"""Compare uniform-K trellis experts directly with official decoded MXFP4.

This is a numerical endpoint study for an eventual all-TrellisShift artifact.
It streams official MXFP4 matrices one expert at a time, uses only the
resident interim model's captured routes/activations for calibration, and
compares complete K3/K4 expert reconstructions against the decoded official
expert function.  The official model is never instantiated.

Unlike the earlier sampled-tile studies, every coefficient passes through the
production EXL Hadamard, scale search, dense-H BlockLDLQ traversal, stored
FP16 scale vectors, and reference procedural decoder before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from kquant import constants as C
from kquant.capture import load_layer_hessians, load_layer_samples
from kquant.exl3_reference import (
    CODEBOOK_MCG,
    CODEBOOK_MUL1_E4M3,
    EXL3_CODEBOOKS,
    decode_exl3_weight,
)
from kquant.sqg_e4m3 import (
    SQG_E4M3_CODEBOOKS,
    sqg_e4m3_bytes,
    sqg_e4m3_bytes_from_rank_lut,
    sqg_e4m3_codebook,
    sqg_mode_from_codebook,
)
from kquant.sqg_quantizer import install_sqg_quantizer
from kquant.mixed_candidates import (
    RequestPartition,
    activation_block_contexts,
    build_expert_hessians,
    partition_requests,
    request_documents,
    select_expert_rows,
)
from kquant.mixed_exl3 import RATE_TRANSFER_MODES, unpack_trellis_states
from kquant.pack.exl3 import MATRICES, SIGMA_REG, make_shared_h
from kquant.pack.mixed_exl3 import plan_mixed_matrix
from kquant.source_weights import OfficialMXFP4Store
from kquant.tp_simulator import comparison_metrics
from scripts.experiment_codec_transfers import _load_exl3_quantizer
from scripts.experiment_coupled_exl3 import (
    _coupled_activation_metrics,
)
from scripts.experiment_retained_w2_rate_transfer import (
    _read_json,
)


SQG_CHEB_NORMAL_E4M3 = "sqg-cheb-normal-e4m3"
STUDY_CODEBOOKS = tuple(
    dict.fromkeys(EXL3_CODEBOOKS + SQG_E4M3_CODEBOOKS + (SQG_CHEB_NORMAL_E4M3,))
)


def _parse_ints(value: str, *, minimum: int, maximum: int) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    if any(not minimum <= item <= maximum for item in values):
        raise argparse.ArgumentTypeError(
            f"values must be in {minimum}..{maximum}"
        )
    return values


def _parse_experts(value: str) -> tuple[int, ...]:
    return _parse_ints(value, minimum=0, maximum=C.NUM_EXPERTS - 1)


def _parse_rates(value: str) -> tuple[int, ...]:
    # The stored-state reference currently closes the production K2/K3/K4
    # syntax.  Include K2 so SQG can be judged on TrellisShift's donor arm as
    # well as the uniform K3 and recipient K4 endpoints.
    values = _parse_ints(value, minimum=2, maximum=4)
    return values


def _parse_codebooks(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("codebooks must be nonempty and unique")
    if any(item not in STUDY_CODEBOOKS for item in values):
        raise argparse.ArgumentTypeError(
            f"codebooks must be selected from {STUDY_CODEBOOKS}"
        )
    return values


def _load_sqg_rank_lut(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    """Load one shared rank-indexed finite-E4M3 reconstruction law."""

    payload = path.read_bytes()
    if len(payload) != 1 << 16:
        raise ValueError("SQG rank LUT must contain exactly 65,536 bytes")
    rank_lut = torch.tensor(list(payload), dtype=torch.uint8)
    values = rank_lut.view(torch.float8_e4m3fn).float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("SQG rank LUT must contain only finite E4M3 labels")
    return rank_lut, {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "distinct_e4m3_labels": int(torch.unique(rank_lut).numel()),
        "graph_contract": (
            "one shared rank law; unchanged baseline SQG graph for every rate"
        ),
    }


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(path)


def _request_mask(
    request_steps: torch.Tensor, requests: Mapping[int, str]
) -> torch.Tensor:
    allowed = torch.tensor(
        sorted(map(int, requests)),
        dtype=request_steps.dtype,
        device=request_steps.device,
    )
    return torch.isin(request_steps, allowed)


def _encoder_coordinates(
    source: torch.Tensor,
    hessian: torch.Tensor,
    contexts: torch.Tensor,
    *,
    matrix: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return EXL-oriented weight/H and the canonical inverse permutation."""

    plan = plan_mixed_matrix(
        contexts,
        RATE_TRANSFER_MODES[0],
        matrix=matrix,
        layout="importance_ordered",
    )
    permutation = plan.encoder_permutation
    if matrix == "w2":
        weight = source.index_select(1, permutation).T.contiguous()
        hessian_permutation = permutation.to(device=hessian.device)
        encoder_hessian = hessian.index_select(
            0, hessian_permutation
        ).index_select(1, hessian_permutation)
    else:
        weight = source.index_select(0, permutation).T.contiguous()
        encoder_hessian = hessian
    return weight, encoder_hessian, permutation


def _canonical_reconstruction(
    encoder_weight: torch.Tensor,
    source: torch.Tensor,
    permutation: torch.Tensor,
    *,
    matrix: str,
) -> torch.Tensor:
    reconstruction = torch.empty_like(source)
    if matrix == "w2":
        reconstruction[:, permutation] = encoder_weight.T
    else:
        reconstruction[permutation] = encoder_weight.T
    return reconstruction.contiguous()


def _dense_h_terms(
    source: torch.Tensor,
    reconstruction: torch.Tensor,
    hessian: torch.Tensor,
    *,
    quantizer_module,
    device: torch.device,
) -> tuple[float, float]:
    """Score the persisted reconstruction in canonical captured-H geometry."""

    source_exl = source.T.contiguous()
    error_exl = (reconstruction - source).T.contiguous()
    hessian_device = hessian.to(device=device, dtype=torch.float32)
    numerator = float(quantizer_module.block_trace(error_exl, hessian_device))
    denominator = float(quantizer_module.block_trace(source_exl, hessian_device))
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ValueError("dense-H metric is non-finite")
    if numerator < 0 or denominator <= 0:
        raise ValueError("dense-H metric has invalid energy")
    return numerator, denominator


def _quantize_uniform_matrix(
    source: torch.Tensor,
    contexts: torch.Tensor,
    *,
    matrix: str,
    bits: int,
    codebook: str,
    hessian: torch.Tensor,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    tailbite_context: int,
    external_sqg_luts: Mapping[int, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Encode one full matrix and independently decode its stored tensors."""

    weight, encoder_hessian, permutation = _encoder_coordinates(
        source, hessian, contexts, matrix=matrix
    )
    shared_h = make_shared_h(weight.shape[0], device, encoder_hessian)
    transform_seed = layer * 1_000_000 + MATRICES.index(matrix)
    quant_args: dict[str, object] = {
        "K": bits,
        "seed": transform_seed,
        "sv_seed": transform_seed + 499_979,
        "sigma_reg": SIGMA_REG,
        "devices": [str(device)],
        "device_ratios": None,
        "apply_out_scales": False,
        "ldlq_tf32": bool(ldlq_tf32),
        "tailbite_context": int(tailbite_context),
    }
    custom_codebook = None
    if codebook == SQG_CHEB_NORMAL_E4M3:
        if external_sqg_luts is None or bits not in external_sqg_luts:
            raise ValueError(f"missing external SQG rank law for K{bits}")
        raw_codebook = external_sqg_luts[bits].to(device=device)
        quant_args["sqg_e4m3_lut"] = raw_codebook
        custom_codebook = (
            raw_codebook.view(torch.float8_e4m3fn)
            .to(dtype=torch.float16)
            .contiguous()
        )
    elif codebook in SQG_E4M3_CODEBOOKS:
        mode = sqg_mode_from_codebook(codebook)
        quant_args["sqg_e4m3_lut"] = sqg_e4m3_bytes(
            bits, mode, device=device
        )
        custom_codebook = sqg_e4m3_codebook(
            bits, mode, device=device, dtype=torch.float16
        )
    elif codebook == CODEBOOK_MCG:
        quant_args["mcg"] = True
    elif codebook == CODEBOOK_MUL1_E4M3:
        quant_args["mul1_e4m3"] = True
    else:
        raise ValueError(f"unsupported codebook: {codebook}")
    if matrix in ("w1", "w3"):
        # This study asks for the numerical endpoint, so each expert/rate/
        # codebook receives its own fitted hidden-side scale profile.  A later
        # layer-shared-scale ablation can measure the small format constraint.
        quant_args["shared_input_scales_key"] = (
            "uniform-k-mxfp4-endpoint",
            layer,
            expert,
            matrix,
            bits,
            codebook,
        )
        quant_args["g_scale_into_sv"] = True

    quantized, proxy, tensors = quantizer_module.quantize_exl3(
        weight,
        shared_h,
        quant_args,
        True,
        progress_str="",
    )
    states = unpack_trellis_states(tensors["trellis"], bits)
    stored_encoder_weight = decode_exl3_weight(
        states,
        tensors["suh"],
        tensors["svh"],
        codebook=codebook,
        codebook_values=custom_codebook,
    )
    reconstruction = _canonical_reconstruction(
        stored_encoder_weight,
        source,
        permutation,
        matrix=matrix,
    )
    dense_numerator, dense_denominator = _dense_h_terms(
        source,
        reconstruction,
        hessian,
        quantizer_module=quantizer_module,
        device=device,
    )
    raw_error = (reconstruction.double() - source.double()).square().sum()
    raw_energy = source.double().square().sum()
    trellis_bytes = tensors["trellis"].numel() * tensors["trellis"].element_size()
    scale_bytes = sum(
        tensors[name].numel() * tensors[name].element_size()
        for name in ("suh", "svh")
    )
    closure = comparison_metrics(quantized, stored_encoder_weight)
    if not all(math.isfinite(value) for value in closure.values()):
        raise ValueError("stored reconstruction closure is non-finite")
    evidence: dict[str, object] = {
        "matrix": matrix,
        "bits": bits,
        "codebook": codebook,
        "weight_nmse": float(raw_error / raw_energy),
        "weight_squared_error": float(raw_error),
        "weight_reference_energy": float(raw_energy),
        "captured_dense_h_nmse": dense_numerator / dense_denominator,
        "captured_dense_h_numerator": dense_numerator,
        "captured_dense_h_denominator": dense_denominator,
        "encoder_regularized_dense_h_proxy": float(proxy),
        "trellis_bytes": trellis_bytes,
        "scale_bytes_before_layer_deduplication": scale_bytes,
        "bytes_before_layer_deduplication": trellis_bytes + scale_bytes,
        "trellis_bpw": bits,
        "trellis_and_scale_bpw_before_layer_deduplication": (
            (trellis_bytes + scale_bytes) * 8 / source.numel()
        ),
        "stored_fp16_scale_closure_vs_encoder_float_scales": closure,
        "tensor_shapes": {
            name: list(tensors[name].shape)
            for name in ("trellis", "suh", "svh")
        },
    }
    return reconstruction, evidence


def _aggregate_matrix_evidence(
    matrix_evidence: Sequence[Mapping[str, object]],
) -> dict[str, float | int]:
    raw_numerator = sum(float(value["weight_squared_error"]) for value in matrix_evidence)
    raw_denominator = sum(
        float(value["weight_reference_energy"]) for value in matrix_evidence
    )
    dense_numerator = sum(
        float(value["captured_dense_h_numerator"]) for value in matrix_evidence
    )
    dense_denominator = sum(
        float(value["captured_dense_h_denominator"]) for value in matrix_evidence
    )
    total_bytes = sum(
        int(value["bytes_before_layer_deduplication"]) for value in matrix_evidence
    )
    total_weights = sum(
        int(np.prod(value["tensor_shapes"]["suh"]))
        * int(np.prod(value["tensor_shapes"]["svh"]))
        for value in matrix_evidence
    )
    return {
        "weight_nmse": raw_numerator / raw_denominator,
        "captured_dense_h_nmse": dense_numerator / dense_denominator,
        "bytes_before_layer_deduplication": total_bytes,
        "bpw_before_layer_deduplication": total_bytes * 8 / total_weights,
    }


def _document_improvement(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    """Paired document bootstrap of candidate SSE improvement over baseline."""

    def table(metric: Mapping[str, object]) -> dict[str, float]:
        rows = metric.get("request_contributions")
        if not isinstance(rows, list):
            raise ValueError("functional metric lacks request contributions")
        result: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("invalid request contribution")
            document = str(row.get("document_hash"))
            if not document or document == "None" or document in result:
                raise ValueError("request contributions need unique document hashes")
            result[document] = float(row["route_weighted_squared_error"])
        return result

    base = table(baseline)
    trial = table(candidate)
    if set(base) != set(trial) or not base:
        raise ValueError("paired candidates changed document support")
    documents = sorted(base)
    base_values = np.asarray([base[item] for item in documents], dtype=np.float64)
    trial_values = np.asarray([trial[item] for item in documents], dtype=np.float64)
    base_total = float(base_values.sum())
    if base_total <= 0:
        raise ValueError("baseline document SSE is non-positive")
    observed = 1.0 - float(trial_values.sum()) / base_total
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(documents), size=(replicates, len(documents)))
    sampled_base = base_values[indices].sum(axis=1)
    sampled_trial = trial_values[indices].sum(axis=1)
    valid = sampled_base > 0
    samples = 1.0 - sampled_trial[valid] / sampled_base[valid]
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "documents": len(documents),
        "relative_sse_improvement": observed,
        "ci95": [float(low), float(high)],
        "valid_bootstrap_replicates": int(samples.size),
    }


def _compact_functional_evidence(candidates: Mapping[str, dict]) -> None:
    """Drop row-cluster details after all paired statistics are finalized.

    Full per-document contributions are useful for a small audit, but they make
    an all-expert result expensive to rewrite atomically after every completed
    expert.  Aggregate fold metrics and the already-computed paired bootstrap
    remain sufficient for the bulk rate-frontier study.
    """

    for candidate in candidates.values():
        routed = candidate.get("routed_function")
        if not isinstance(routed, Mapping):
            raise TypeError("candidate routed-function evidence must be a mapping")
        for fold in routed.values():
            if fold is None:
                continue
            if not isinstance(fold, dict):
                raise TypeError("routed-function fold evidence must be mutable")
            fold.pop("request_contributions", None)


def _fit_contract(
    args: argparse.Namespace,
    training_requests: Mapping[int, str],
) -> RequestPartition:
    partition = partition_requests(training_requests)
    manifest = _read_json(args.hessians / "manifest.json")
    if Path(str(manifest.get("source_capture", ""))).resolve() != args.capture.resolve():
        raise ValueError("Hessian bundle does not come from the training capture")
    if manifest.get("sample_split") != "all":
        raise ValueError("Hessian bundle must retain all selected fit-document rows")
    if manifest.get("request_step_filter") != sorted(partition.fit):
        raise ValueError("Hessian request filter differs from the fit-document split")
    return partition


def _validate_corpus_contract(
    args: argparse.Namespace,
    training_report: Mapping[str, object],
    validation_report: Mapping[str, object],
) -> tuple[dict[str, object], dict[int, str], dict[int, str]]:
    """Validate the document split while dropping repeated validation requests."""

    for name, report, capture in (
        ("training", training_report, args.capture),
        ("validation", validation_report, args.validation_capture),
    ):
        if report.get("kind") != "kquant_interim_calibration_corpus_run":
            raise ValueError(f"{name} corpus report has the wrong kind")
        if not bool(report.get("finalized")):
            raise ValueError(f"{name} corpus report is not finalized")
        if Path(str(report.get("capture_dir", ""))).resolve() != capture.resolve():
            raise ValueError(f"{name} report and capture disagree")
        if Path(str(report.get("model_dir", ""))).resolve() != args.checkpoint.resolve():
            raise ValueError(f"{name} capture used a different resident teacher")

    training_fold = training_report.get("fold", {})
    validation_fold = validation_report.get("fold", {})
    if not isinstance(training_fold, Mapping) or not isinstance(
        validation_fold, Mapping
    ):
        raise TypeError("corpus fold metadata must be mappings")
    common_fold = ("modulus", "index", "unit")
    if any(
        training_fold.get(key) != validation_fold.get(key)
        for key in common_fold
    ):
        raise ValueError("training and validation reports use different folds")
    if training_fold.get("mode") != "exclude":
        raise ValueError("training report must exclude the validation fold")
    if validation_fold.get("mode") != "include":
        raise ValueError("validation report must contain only the validation fold")

    training_requests = request_documents(training_report)
    validation_requests = request_documents(validation_report, deduplicate=True)
    overlap = set(training_requests.values()) & set(validation_requests.values())
    if overlap:
        raise ValueError(f"training/validation document overlap: {sorted(overlap)[:8]}")
    raw_validation_documents = validation_report.get("documents")
    if not isinstance(raw_validation_documents, list):
        raise TypeError("validation report documents must be a list")
    provenance: dict[str, object] = {
        "training_report": str(args.training_report.resolve()),
        "validation_report": str(args.validation_report.resolve()),
        "training_documents": len(training_requests),
        "validation_documents": len(validation_requests),
        "validation_requests_captured": len(raw_validation_documents),
        "validation_duplicate_requests_dropped": (
            len(raw_validation_documents) - len(validation_requests)
        ),
        "document_overlap": 0,
        "fold": {key: training_fold.get(key) for key in common_fold},
    }
    return provenance, training_requests, validation_requests


def _signature(
    args: argparse.Namespace,
    provenance: dict,
    source: Path,
    *,
    sqg_rank_lut: Mapping[str, object] | None,
) -> dict:
    return {
        "resident_teacher": str(args.checkpoint.resolve()),
        "official_source": str(source.resolve()),
        "official_revision": args.official_revision,
        "capture": str(args.capture.resolve()),
        "validation_capture": str(args.validation_capture.resolve()),
        "hessians": str(args.hessians.resolve()),
        "training_report": str(args.training_report.resolve()),
        "validation_report": str(args.validation_report.resolve()),
        "layer": args.layer,
        "experts": list(args.experts),
        "rates": list(args.rates),
        "codebooks": list(args.codebooks),
        "ldlq_tf32": args.ldlq_tf32,
        "tailbite_context": args.tailbite_context,
        "compact": args.compact,
        "sqg_rank_lut": dict(sqg_rank_lut) if sqg_rank_lut is not None else None,
        "tp_size": 12,
        "provenance": provenance,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the full dense-H endpoint study requires CUDA")
    torch.cuda.set_device(device)
    training_report = _read_json(args.training_report)
    validation_report = _read_json(args.validation_report)
    provenance, training_requests, validation_requests = _validate_corpus_contract(
        args, training_report, validation_report
    )
    partition = _fit_contract(args, training_requests)
    training_samples = load_layer_samples(args.capture, args.layer - 1)
    validation_samples = load_layer_samples(
        args.validation_capture, args.layer - 1
    )
    global_h13, global_h2 = load_layer_hessians(args.hessians, args.layer)
    external_sqg_luts: dict[int, torch.Tensor] | None = None
    sqg_rank_lut_metadata: dict[str, object] | None = None
    if SQG_CHEB_NORMAL_E4M3 in args.codebooks:
        rank_lut, sqg_rank_lut_metadata = _load_sqg_rank_lut(args.sqg_rank_lut)
        external_sqg_luts = {
            bits: sqg_e4m3_bytes_from_rank_lut(bits, rank_lut, device=device)
            for bits in args.rates
        }
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    signature = _signature(
        args,
        provenance,
        store.root,
        sqg_rank_lut=sqg_rank_lut_metadata,
    )
    if args.resume:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        payload = _read_json(args.output)
        if payload.get("signature") != signature:
            raise ValueError("resume output does not match this study")
        if payload.get("complete"):
            return payload
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        payload = {
            "kind": "kquant_uniform_k_mxfp4_endpoint_study",
            "schema_version": 1,
            "signature": signature,
            "metric_target": (
                "complete decoded official MXFP4 expert; captures/routes come "
                "from the resident interim teacher"
            ),
            "scale_scope": (
                "expert/rate/codebook-specific numerical ceiling; a later "
                "layer-shared scale ablation is required for final format closure"
            ),
            "results": {},
            "skipped": {},
            "complete": False,
        }
        _atomic_write(args.output, payload)

    quantize_exl3 = _load_exl3_quantizer(args.exllamav3_root)
    quantizer_module = __import__(quantize_exl3.__module__, fromlist=["*"])
    if any(
        codebook in SQG_E4M3_CODEBOOKS or codebook == SQG_CHEB_NORMAL_E4M3
        for codebook in args.codebooks
    ):
        install_sqg_quantizer(quantizer_module)
    completed = set(payload["results"]) | set(payload["skipped"])
    for expert in args.experts:
        key = str(expert)
        if key in completed:
            continue
        all_rows = select_expert_rows(training_samples, expert, partition.all)
        fit_mask = _request_mask(all_rows.request_steps, partition.fit)
        confirmation_mask = _request_mask(
            all_rows.request_steps, partition.confirmation
        )
        validation_rows = select_expert_rows(
            validation_samples, expert, validation_requests
        )
        support = {
            "fit_rows": int(fit_mask.sum()),
            "fit_documents": int(torch.unique(all_rows.request_steps[fit_mask]).numel()),
            "confirmation_rows": int(confirmation_mask.sum()),
            "confirmation_documents": int(
                torch.unique(all_rows.request_steps[confirmation_mask]).numel()
            ),
            "validation_rows": validation_rows.rows,
            "validation_documents": validation_rows.documents,
        }
        if (
            support["fit_documents"] < args.min_fit_documents
            or support["confirmation_documents"] < args.min_confirmation_documents
            or support["validation_documents"] < args.min_validation_documents
        ):
            payload["skipped"][key] = {
                "reason": "insufficient document support",
                "support": support,
            }
            _atomic_write(args.output, payload)
            continue

        source = tuple(
            store.load_matrix(
                args.layer, expert, matrix, device=device
            ).float().contiguous()
            for matrix in MATRICES
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
        candidates: dict[str, dict] = {}
        for codebook in args.codebooks:
            for bits in args.rates:
                candidate_key = f"{codebook}:K{bits}"
                reconstructions = []
                matrix_evidence = []
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                for matrix, source_matrix in zip(MATRICES, source, strict=True):
                    reconstruction, evidence = _quantize_uniform_matrix(
                        source_matrix,
                        contexts,
                        matrix=matrix,
                        bits=bits,
                        codebook=codebook,
                        hessian=h2 if matrix == "w2" else h13,
                        layer=args.layer,
                        expert=expert,
                        device=device,
                        quantizer_module=quantizer_module,
                        ldlq_tf32=args.ldlq_tf32,
                        tailbite_context=args.tailbite_context,
                        external_sqg_luts=external_sqg_luts,
                    )
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
                candidates[candidate_key] = {
                    "codebook": codebook,
                    "bits": bits,
                    "matrix_evidence": matrix_evidence,
                    "aggregate": _aggregate_matrix_evidence(matrix_evidence),
                    "routed_function": functional,
                    "quantization_and_scoring_seconds": time.perf_counter() - started,
                }
                del reconstructions
                torch.cuda.empty_cache()

        comparisons: dict[str, dict] = {}
        for codebook in args.codebooks:
            k3 = candidates.get(f"{codebook}:K3")
            k4 = candidates.get(f"{codebook}:K4")
            if k3 is None or k4 is None:
                continue
            comparison: dict[str, object] = {
                "weight_error_ratio_k4_over_k3": (
                    float(k4["aggregate"]["weight_nmse"])
                    / float(k3["aggregate"]["weight_nmse"])
                ),
                "dense_h_error_ratio_k4_over_k3": (
                    float(k4["aggregate"]["captured_dense_h_nmse"])
                    / float(k3["aggregate"]["captured_dense_h_nmse"])
                ),
            }
            for fold in ("train", "confirmation", "validation"):
                baseline = k3["routed_function"].get(fold)
                trial = k4["routed_function"].get(fold)
                if baseline is None or trial is None:
                    continue
                comparison[f"{fold}_routed_error_ratio_k4_over_k3"] = (
                    float(trial["route_weighted_squared_error"])
                    / float(baseline["route_weighted_squared_error"])
                )
                comparison[f"{fold}_paired_document_improvement"] = (
                    _document_improvement(
                        baseline,
                        trial,
                        replicates=args.bootstrap_replicates,
                        seed=args.layer * 1_000_000 + expert * 10 + len(fold),
                    )
                )
            comparisons[codebook] = comparison

        if args.compact:
            _compact_functional_evidence(candidates)

        payload["results"][key] = {
            "support": support,
            "covariance": covariance,
            "context": {
                "basis": "official_source_post_situ_fit_documents",
                "score_min": float(block_scores.min()),
                "score_median": float(block_scores.median()),
                "score_max": float(block_scores.max()),
            },
            "candidates": candidates,
            "k4_vs_k3": comparisons,
        }
        _atomic_write(args.output, payload)
        del source, fit_inputs, fit_gates, source_middle, h13, h2, contexts
        torch.cuda.empty_cache()

    payload["complete"] = True
    _atomic_write(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/models/Kimi-K3-EXL3-3p09-serve"),
        help="resident interim teacher used to produce the captures",
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
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", type=_parse_experts, required=True)
    parser.add_argument("--rates", type=_parse_rates, default=(3, 4))
    parser.add_argument(
        "--codebooks",
        type=_parse_codebooks,
        default=(CODEBOOK_MUL1_E4M3, CODEBOOK_MCG),
    )
    parser.add_argument(
        "--sqg-rank-lut",
        type=Path,
        help=(
            "shared 65,536-byte rank-indexed E4M3 law required by "
            f"{SQG_CHEB_NORMAL_E4M3}"
        ),
    )
    parser.add_argument("--device", default="cuda:11")
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--ldlq-tf32", action="store_true")
    parser.add_argument("--tailbite-context", type=int, default=128)
    parser.add_argument("--min-fit-documents", type=int, default=6)
    parser.add_argument("--min-confirmation-documents", type=int, default=4)
    parser.add_argument("--min-validation-documents", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "discard per-document fold contributions after computing paired "
            "bootstrap statistics; intended for resumable all-expert studies"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= 92:
        parser.error("--layer must be in 1..92")
    if not 1 <= args.tailbite_context <= 128:
        parser.error("--tailbite-context must be in 1..128")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    if SQG_CHEB_NORMAL_E4M3 in args.codebooks and args.sqg_rank_lut is None:
        parser.error(
            f"--sqg-rank-lut is required for {SQG_CHEB_NORMAL_E4M3}"
        )
    return args


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "complete": bool(payload["complete"]),
                "experts": sorted(payload["results"]),
                "skipped": payload["skipped"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
