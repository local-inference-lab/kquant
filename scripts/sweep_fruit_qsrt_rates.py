#!/usr/bin/env python3
"""Measure the authenticated Fruit expert curve at uniform K2/K3/K4 rates."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from kquant.exl3_loader import load_qsrt_encoder
from kquant.exl3_reference import decode_qsrt_weight
from kquant.fruit_calibration import FruitCalibrationRows, FruitCalibrationStore
from kquant.fruit_qsrt import (
    FRUIT_QSRT_CODEBOOK,
    FRUIT_QSRT_MATRICES,
    _activation_permutations,
    _calibration_support,
    _canonical_reconstruction,
    _encoder_weight,
    _functional_metric,
    _middle_rows,
    _quant_args,
    _rate_axis,
)
from kquant.fruit_source import (
    FRUIT_ANNEALED_SPEC,
    FruitCheckpointStore,
    FruitSafetensorsStore,
)
from kquant.ldlq import make_shared_h
from kquant.logical_qsrt import R0, MatrixName
from kquant.qsrt_candidates import build_expert_hessians
from kquant.sqg_quantizer import install_sqg_quantizer
from kquant.tp_simulator import comparison_metrics
from scripts.build_fruit_qsrt_model import (
    BASE_MANIFEST_SHA256,
    current_encoder_provenance,
)
from scripts.encode_fruit_qsrt import SAMPLED_ASSIGNMENTS, _assignment

_RATES = (2, 3, 4)
_SCHEMA = "kquant_fruit_uniform_rate_sweep_v1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(_canonical_json(value), encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _tensor(raw: Mapping[str, object], name: str) -> torch.Tensor:
    value = raw.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"uniform-rate encoder result omits tensor {name!r}")
    return value


def _matrix_evidence(
    source: torch.Tensor,
    reconstruction: torch.Tensor,
    raw: Mapping[str, object],
    *,
    matrix: MatrixName,
    bits: int,
    score_hessian: torch.Tensor,
    quantizer_module: object,
) -> dict[str, object]:
    error = reconstruction.double() - source.double()
    squared_error = float(error.square().sum())
    reference_energy = float(source.double().square().sum())
    if not math.isfinite(squared_error) or not math.isfinite(reference_energy):
        raise ValueError("uniform-rate weight evidence is non-finite")
    if squared_error < 0 or reference_energy <= 0:
        raise ValueError("uniform-rate weight evidence has invalid energy")

    error_exl = (reconstruction - source).T.contiguous()
    source_exl = source.T.contiguous()
    block_trace = getattr(quantizer_module, "block_trace", None)
    if block_trace is None:
        raise ValueError("QSRT encoder backend lacks block_trace")
    dense_numerator = float(block_trace(error_exl, score_hessian))
    dense_denominator = float(block_trace(source_exl, score_hessian))
    if (
        not math.isfinite(dense_numerator)
        or not math.isfinite(dense_denominator)
        or dense_numerator < 0
        or dense_denominator <= 0
    ):
        raise ValueError("uniform-rate dense-H evidence has invalid energy")

    scale_values = _tensor(raw, "suh").numel() + _tensor(raw, "svh").numel()
    trellis_bits = source.numel() * bits
    if trellis_bits % 8:
        raise ValueError("uniform-rate trellis payload is not byte aligned")
    trellis_bytes = trellis_bits // 8
    scale_bytes = scale_values * torch.float16.itemsize
    closure = comparison_metrics(_tensor(raw, "weight_q"), _tensor(raw, "decoded"))
    if not all(math.isfinite(value) for value in closure.values()):
        raise ValueError("uniform-rate stored decode closure is non-finite")
    return {
        "matrix": matrix,
        "bits": bits,
        "weight_squared_error": squared_error,
        "weight_reference_energy": reference_energy,
        "weight_nmse": squared_error / reference_energy,
        "captured_dense_h_numerator": dense_numerator,
        "captured_dense_h_denominator": dense_denominator,
        "captured_dense_h_nmse": dense_numerator / dense_denominator,
        "trellis_bytes": trellis_bytes,
        "scale_bytes_before_layer_deduplication": scale_bytes,
        "bytes_before_layer_deduplication": trellis_bytes + scale_bytes,
        "stored_fp16_scale_closure_vs_encoder": closure,
    }


def _quantize_group(
    sources: Sequence[torch.Tensor],
    matrices: Sequence[MatrixName],
    permutation: torch.Tensor,
    *,
    encoder_hessians: Sequence[torch.Tensor],
    score_hessians: Sequence[torch.Tensor],
    bits: int,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module: object,
) -> tuple[dict[MatrixName, torch.Tensor], list[dict[str, object]]]:
    if not (
        len(sources) == len(matrices) == len(encoder_hessians) == len(score_hessians)
    ):
        raise ValueError("uniform-rate matrix groups must have equal lengths")
    weights = [
        _encoder_weight(source, matrix, permutation)
        for source, matrix in zip(sources, matrices, strict=True)
    ]
    shared_h = [
        make_shared_h(weight.shape[0], device, hessian)
        for weight, hessian in zip(weights, encoder_hessians, strict=True)
    ]
    groups: list[list[dict[str, object]]] = []
    for matrix in matrices:
        arguments = _quant_args(
            matrix,
            R0,
            layer=layer,
            expert=expert,
            device=device,
        )
        arguments["K"] = bits
        mixed_tile_bits = arguments["mixed_tile_bits"]
        if not isinstance(mixed_tile_bits, tuple):
            raise TypeError("Fruit mixed_tile_bits contract changed")
        arguments["mixed_tile_bits"] = (bits,) * len(mixed_tile_bits)
        groups.append([arguments])

    batch_api = getattr(quantizer_module, "quantize_qsrt_batch", None)
    if batch_api is None:
        raise ValueError("QSRT encoder backend lacks quantize_qsrt_batch")
    raw_groups = batch_api(weights, shared_h, groups, return_weight_q=True)
    if len(raw_groups) != len(matrices) or any(len(group) != 1 for group in raw_groups):
        raise ValueError("QSRT encoder backend returned the wrong uniform-rate group")

    reconstructions: dict[MatrixName, torch.Tensor] = {}
    evidence: list[dict[str, object]] = []
    for source, matrix, raw_group, score_hessian in zip(
        sources,
        matrices,
        raw_groups,
        score_hessians,
        strict=True,
    ):
        raw = raw_group[0]
        if not isinstance(raw, dict):
            raise TypeError("QSRT encoder backend returned a malformed candidate")
        encoded = _tensor(raw, "encoded")
        rate_axis = _rate_axis(matrix)
        rate_tiles = encoded.shape[0] if rate_axis == "k" else encoded.shape[1]
        decoded = decode_qsrt_weight(
            encoded,
            _tensor(raw, "suh").to(dtype=torch.float16),
            _tensor(raw, "svh").to(dtype=torch.float16),
            rate_axis=rate_axis,
            tile_bits=(bits,) * int(rate_tiles),
            codebook=FRUIT_QSRT_CODEBOOK,
        )
        raw["decoded"] = decoded
        reconstruction = _canonical_reconstruction(
            decoded,
            source,
            matrix,
            permutation,
        ).contiguous()
        if not bool(torch.all(torch.isfinite(reconstruction))):
            raise ValueError("uniform-rate reconstruction is non-finite")
        reconstructions[matrix] = reconstruction
        evidence.append(
            _matrix_evidence(
                source,
                reconstruction,
                raw,
                matrix=matrix,
                bits=bits,
                score_hessian=score_hessian,
                quantizer_module=quantizer_module,
            )
        )
    return reconstructions, evidence


def _functional_evidence(
    rows: FruitCalibrationRows,
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, object]:
    sse, energy, counts = _functional_metric(rows, reference, candidate)
    total_sse = float(sse.double().sum())
    total_energy = float(energy.double().sum())
    if (
        not math.isfinite(total_sse)
        or not math.isfinite(total_energy)
        or total_sse < 0
        or total_energy <= 0
    ):
        raise ValueError("uniform-rate routed evidence has invalid energy")
    supported = torch.nonzero(counts, as_tuple=False).flatten()
    documents = [
        {
            "document_id": int(index),
            "routed_sse": float(sse[index]),
            "reference_energy": float(energy[index]),
            "rows": int(counts[index]),
        }
        for index in supported.tolist()
    ]
    return {
        "routed_sse": total_sse,
        "reference_energy": total_energy,
        "normalized_sse": total_sse / total_energy,
        "documents": documents,
        "document_count": len(documents),
        "row_count": int(counts.sum()),
    }


def _aggregate_matrices(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    weight_numerator = sum(float(value["weight_squared_error"]) for value in values)
    weight_denominator = sum(
        float(value["weight_reference_energy"]) for value in values
    )
    dense_numerator = sum(
        float(value["captured_dense_h_numerator"]) for value in values
    )
    dense_denominator = sum(
        float(value["captured_dense_h_denominator"]) for value in values
    )
    payload_bytes = sum(
        int(value["bytes_before_layer_deduplication"]) for value in values
    )
    weights = sum(
        int(value["trellis_bytes"]) * 8 // int(value["bits"]) for value in values
    )
    return {
        "weight_nmse": weight_numerator / weight_denominator,
        "captured_dense_h_nmse": dense_numerator / dense_denominator,
        "bytes_before_layer_deduplication": payload_bytes,
        "bpw_before_layer_deduplication": payload_bytes * 8 / weights,
    }


def _sweep_expert(
    store: FruitCheckpointStore,
    calibration,
    *,
    layer: int,
    expert: int,
    rates: Sequence[int],
    device: torch.device,
    quantizer_module: object,
) -> dict[str, object]:
    fallback_reason, support = _calibration_support(calibration)
    if fallback_reason is not None:
        return {
            "status": "skipped",
            "reason": fallback_reason,
            "support": support,
        }
    if calibration.validation.row_count == 0:
        return {
            "status": "skipped",
            "reason": "no routed validation rows",
            "support": support,
        }

    sources = {
        matrix: store.load_matrix(layer, expert, matrix, device=device)
        for matrix in FRUIT_QSRT_MATRICES
    }
    global_h13 = calibration.global_h13.to(
        device=device,
        dtype=torch.float32,
    ).contiguous()
    source_middle = {
        split: _middle_rows(rows, sources["w1"], sources["w3"], device=device)
        for split, rows in (
            ("fit", calibration.fit),
            ("confirmation", calibration.confirmation),
            ("validation", calibration.validation),
        )
    }
    permutation, _, group_scores = _activation_permutations(
        source_middle["fit"], calibration.fit.gates
    )
    identity_h2 = torch.eye(
        sources["w2"].shape[1],
        dtype=torch.float32,
        device=device,
    )
    reference_outputs = {
        split: middle @ sources["w2"].T for split, middle in source_middle.items()
    }

    result: dict[str, object] = {
        "status": "measured",
        "support": support,
        "calibration_fingerprint": calibration.fingerprint,
        "permutation_group_score": {
            "min": float(group_scores.min()),
            "median": float(group_scores.median()),
            "max": float(group_scores.max()),
        },
        "rates": {},
    }
    rate_results = result["rates"]
    assert isinstance(rate_results, dict)
    for bits in rates:
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        upstream, upstream_evidence = _quantize_group(
            (sources["w1"], sources["w3"]),
            ("w1", "w3"),
            permutation,
            encoder_hessians=(global_h13, global_h13),
            score_hessians=(global_h13, global_h13),
            bits=bits,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
        )
        fit_middle = _middle_rows(
            calibration.fit,
            upstream["w1"],
            upstream["w3"],
            device=device,
        )
        _, h2, h2_evidence = build_expert_hessians(
            calibration.fit.inputs,
            calibration.fit.gates,
            fit_middle,
            global_h13=global_h13,
            global_h2=identity_h2,
            device=device,
        )
        h2 = h2.to(device=device, dtype=torch.float32).contiguous()
        h2_encoder = h2.index_select(0, permutation).index_select(1, permutation)
        downstream, downstream_evidence = _quantize_group(
            (sources["w2"],),
            ("w2",),
            permutation,
            encoder_hessians=(h2_encoder,),
            score_hessians=(h2,),
            bits=bits,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
        )
        candidate_outputs: dict[str, torch.Tensor] = {}
        routed: dict[str, object] = {}
        for split, rows in (
            ("fit", calibration.fit),
            ("confirmation", calibration.confirmation),
            ("validation", calibration.validation),
        ):
            middle = _middle_rows(
                rows,
                upstream["w1"],
                upstream["w3"],
                device=device,
            )
            candidate_outputs[split] = middle @ downstream["w2"].T
            routed[split] = _functional_evidence(
                rows,
                reference_outputs[split],
                candidate_outputs[split],
            )
        torch.cuda.synchronize(device)
        matrix_evidence = upstream_evidence + downstream_evidence
        rate_results[f"K{bits}"] = {
            "bits": bits,
            "matrices": matrix_evidence,
            "aggregate": _aggregate_matrices(matrix_evidence),
            "routed_function": routed,
            "conditional_h2": h2_evidence,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        }
        del upstream, downstream, fit_middle, h2, h2_encoder, candidate_outputs
        torch.cuda.empty_cache()

    baseline = rate_results.get("K3")
    if isinstance(baseline, dict):
        comparisons: dict[str, object] = {}
        baseline_aggregate = baseline["aggregate"]
        baseline_routed = baseline["routed_function"]
        assert isinstance(baseline_aggregate, dict)
        assert isinstance(baseline_routed, dict)
        for name, value in rate_results.items():
            if name == "K3" or not isinstance(value, dict):
                continue
            aggregate = value["aggregate"]
            routed = value["routed_function"]
            assert isinstance(aggregate, dict)
            assert isinstance(routed, dict)
            comparisons[f"{name}_over_K3"] = {
                "weight_nmse_ratio": float(aggregate["weight_nmse"])
                / float(baseline_aggregate["weight_nmse"]),
                "captured_dense_h_nmse_ratio": float(aggregate["captured_dense_h_nmse"])
                / float(baseline_aggregate["captured_dense_h_nmse"]),
                "validation_routed_nmse_ratio": float(
                    routed["validation"]["normalized_sse"]
                )
                / float(baseline_routed["validation"]["normalized_sse"]),
            }
        result["comparisons"] = comparisons
    return result


def _parse_rates(value: str) -> tuple[int, ...]:
    try:
        rates = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "rates must be comma-separated integers"
        ) from exc
    if (
        not rates
        or len(set(rates)) != len(rates)
        or any(rate not in _RATES for rate in rates)
    ):
        raise argparse.ArgumentTypeError("rates must be a unique subset of 2,3,4")
    return rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--assignment", action="append", type=_assignment)
    selection.add_argument("--sample", action="store_true")
    parser.add_argument("--rates", type=_parse_rates, default=_RATES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exllamav3-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Fruit uniform-rate sweep requires a CUDA device")
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    assignments = (
        SAMPLED_ASSIGNMENTS
        if args.sample
        else tuple(sorted(set(args.assignment or ())))
    )
    if not assignments:
        raise ValueError("no Fruit assignments selected")

    if args.checkpoint.is_dir():
        store = FruitSafetensorsStore(
            args.checkpoint,
            spec=FRUIT_ANNEALED_SPEC,
            expected_manifest_sha256=BASE_MANIFEST_SHA256,
        )
    else:
        store = FruitCheckpointStore(
            args.checkpoint,
            spec=FRUIT_ANNEALED_SPEC,
            expected_sha256=FRUIT_ANNEALED_SPEC.checkpoint_sha256,
        )
    calibration_store = FruitCalibrationStore(args.calibration)
    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    signature = {
        "schema": _SCHEMA,
        "source": store.evidence,
        "calibration": {
            "capture_id": calibration_store.capture_id,
            "fingerprint": calibration_store.fingerprint,
            "manifest_sha256": calibration_store.manifest_sha256,
        },
        "encoder": current_encoder_provenance(
            exllamav3_root=args.exllamav3_root,
            calibration=calibration_store,
        ),
        "rates": list(args.rates),
        "assignments": [list(value) for value in assignments],
    }
    if args.resume:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            raise ValueError("resume output does not match this Fruit rate sweep")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        payload = {
            "schema": _SCHEMA,
            "signature": signature,
            "metric_target": (
                "local routed expert output on sealed document-disjoint Fruit "
                "fit/confirmation/validation rows"
            ),
            "storage_scope": (
                "uniform path-rate endpoint with per-candidate FP16 scales before "
                "layer deduplication; not a serving package"
            ),
            "results": {},
            "complete": False,
        }
        _atomic_json(args.output, payload)

    results = payload.get("results")
    if not isinstance(results, dict):
        raise TypeError("Fruit rate-sweep results must be a JSON object")
    started = time.perf_counter()
    calibration_layer = None
    calibration_layer_id = None
    with torch.inference_mode():
        for index, (layer, expert) in enumerate(assignments, start=1):
            key = f"{layer}:{expert}"
            if key in results:
                print(
                    f"[{index}/{len(assignments)}] layer {layer} expert {expert}: resume"
                )
                continue
            if calibration_layer_id != layer:
                calibration_layer = calibration_store.load_layer(layer)
                calibration_layer_id = layer
            assert calibration_layer is not None
            result = _sweep_expert(
                store,
                calibration_layer.expert_rows(expert),
                layer=layer,
                expert=expert,
                rates=args.rates,
                device=device,
                quantizer_module=quantizer_module,
            )
            results[key] = result
            payload["elapsed_seconds"] = time.perf_counter() - started
            _atomic_json(args.output, payload)
            print(
                f"[{index}/{len(assignments)}] layer {layer} expert {expert}: "
                f"{result['status']}"
            )
            torch.cuda.empty_cache()
    payload["elapsed_seconds"] = time.perf_counter() - started
    payload["complete"] = True
    _atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
