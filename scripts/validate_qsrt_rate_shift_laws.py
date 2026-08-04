#!/usr/bin/env python3
"""Validate K2-only SQG-Cheb laws in complete QSRT R0/R1/R2 encodes.

This is the authoritative follow-up to the uniform-K2 donor screen.  K3 and
K4 use the exact validated SQG-Cheb normal staircase for every candidate;
only K2 changes.  Each candidate runs through the production TP12 neuron
order, transforms, scale search, dense-H BlockLDLQ feedback, and the full
separate ``(r13, r2)`` grid before document-disjoint routed validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kquant import constants as C
from kquant.capture import load_layer_hessians, load_layer_samples
from kquant.exl3_loader import load_qsrt_encoder
from kquant.sqg_e4m3 import (
    sqg_cheb_normal_rank_e4m3_bytes,
    sqg_e4m3_bytes_from_rank_lut,
    sqg_k4_eight_stratum_e4m3_bytes_from_rank_lut,
    sqg_k2_tail_compressed_rank_e4m3_bytes,
)
from kquant.sqg_quantizer import install_sqg_quantizer
from kquant.qsrt import RATE_TRANSFER_MODES
from kquant.qsrt_candidates import (
    activation_block_contexts,
    build_expert_hessians,
    deterministic_expert_seed,
    functional_sse_by_request,
    layer_fallback_contexts,
    select_expert_rows,
    select_phase1_rate_pair,
)
from kquant.pack.qsrt_encoder import quantize_qsrt_matrix_candidates_batched
from kquant.source_weights import OfficialMXFP4Store
from scripts.validate_qsrt_codebooks import (
    _atomic_write,
    _fit_contract,
    _read_json,
    _request_mask,
    _validate_corpus_contract,
)
from scripts.validate_qsrt_k2_laws import LawSpec, _lut_metadata, _parse_experts, _parse_laws


def _expert_output(
    inputs: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    return F.linear(F.silu(F.linear(inputs, w1)) * F.linear(inputs, w3), w2)


def _metric_json(
    metric: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> dict[str, object]:
    sse, energy, counts = metric
    total_sse = float(sse.sum())
    total_energy = float(energy.sum())
    return {
        "route_weighted_squared_error": total_sse,
        "route_weighted_reference_energy": total_energy,
        "route_weighted_expert_nmse": total_sse / total_energy,
        "document_sse": sse.tolist(),
        "document_reference_energy": energy.tolist(),
        "document_rows": counts.tolist(),
    }


def _candidate_luts(
    law: LawSpec,
    device: torch.device,
    *,
    k4_eight_stratum: bool = False,
) -> dict[int, torch.Tensor | None]:
    normal = sqg_cheb_normal_rank_e4m3_bytes()
    k2 = sqg_k2_tail_compressed_rank_e4m3_bytes(law.family, law.parameter)
    if k4_eight_stratum:
        k4 = sqg_k4_eight_stratum_e4m3_bytes_from_rank_lut(
            normal, device=device
        )
    else:
        k4 = sqg_e4m3_bytes_from_rank_lut(4, normal, device=device)
    return {
        2: sqg_e4m3_bytes_from_rank_lut(2, k2, device=device),
        3: sqg_e4m3_bytes_from_rank_lut(3, normal, device=device),
        4: k4,
    }


def _profile_name(law: LawSpec, *, k4_eight_stratum: bool) -> str:
    return f"{law.name}+k4-q8" if k4_eight_stratum else law.name


def _quantize_grid(
    sources: dict[str, torch.Tensor],
    contexts: torch.Tensor,
    h13: torch.Tensor,
    h2: torch.Tensor,
    *,
    layer: int,
    device: torch.device,
    shared_scale_scope: object,
    quantizer_module: object,
    luts: dict[int, torch.Tensor | None],
    ldlq_tf32: bool,
    tailbite_context: int,
) -> tuple[dict[str, dict], dict[str, dict]]:
    modes = tuple(RATE_TRANSFER_MODES[index] for index in (0, 1, 2))
    upstream = quantize_qsrt_matrix_candidates_batched(
        {"w1": sources["w1"], "w3": sources["w3"]},
        contexts,
        modes=modes,
        hessians={"w1": h13, "w3": h13},
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        quantizer_module=quantizer_module,
        layout="importance_ordered",
        sqg_e4m3_luts_by_bits=luts,
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
    )
    down = quantize_qsrt_matrix_candidates_batched(
        {"w2": sources["w2"]},
        contexts,
        modes=(modes[0],),
        hessians={"w2": h2},
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        quantizer_module=quantizer_module,
        layout="importance_ordered",
        sqg_e4m3_luts_by_bits=luts,
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
    )
    shifted = quantize_qsrt_matrix_candidates_batched(
        {"w2": sources["w2"]},
        contexts,
        modes=modes[1:],
        hessians={"w2": h2},
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        quantizer_module=quantizer_module,
        layout="qsrt_pair_prefix_reuse",
        sqg_e4m3_luts_by_bits=luts,
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
    )
    down["w2"].update(shifted["w2"])
    return upstream, down


def _prime_shared_scales(
    store: OfficialMXFP4Store,
    *,
    layer: int,
    global_h13: torch.Tensor,
    fallback_contexts: torch.Tensor,
    device: torch.device,
    shared_scale_scope: object,
    quantizer_module: object,
    luts: dict[int, torch.Tensor | None],
    ldlq_tf32: bool,
    tailbite_context: int,
) -> None:
    sources = {
        matrix: store.load_matrix(layer, 0, matrix, device=device).float().contiguous()
        for matrix in ("w1", "w3")
    }
    quantize_qsrt_matrix_candidates_batched(
        sources,
        fallback_contexts.to(device=device, dtype=torch.long),
        modes=(RATE_TRANSFER_MODES[0],),
        hessians={"w1": global_h13, "w3": global_h13},
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        quantizer_module=quantizer_module,
        layout="importance_ordered",
        sqg_e4m3_luts_by_bits=luts,
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
    )
    del sources
    torch.cuda.empty_cache()


def _score_grid(
    sources: dict[str, torch.Tensor],
    upstream: dict[str, dict],
    down: dict[str, dict],
    *,
    training_rows,
    fit_mask: torch.Tensor,
    confirmation_mask: torch.Tensor,
    validation_rows,
    partition,
    validation_requests: dict[int, str],
    device: torch.device,
    bootstrap_replicates: int,
    layer: int,
    expert: int,
) -> dict[str, object]:
    training_inputs = training_rows.inputs.float().to(device=device)
    training_gates = training_rows.gates.float().to(device=device)
    validation_inputs = validation_rows.inputs.float().to(device=device)
    validation_gates = validation_rows.gates.float().to(device=device)
    training_reference = _expert_output(training_inputs, **sources)
    validation_reference = _expert_output(validation_inputs, **sources)

    training_middle = {
        r13: F.silu(F.linear(training_inputs, upstream["w1"][r13].reconstruction))
        * F.linear(training_inputs, upstream["w3"][r13].reconstruction)
        for r13 in (0, 1, 2)
    }
    validation_middle = {
        r13: F.silu(F.linear(validation_inputs, upstream["w1"][r13].reconstruction))
        * F.linear(validation_inputs, upstream["w3"][r13].reconstruction)
        for r13 in (0, 1, 2)
    }
    fit: dict[tuple[int, int], torch.Tensor] = {}
    confirmation: dict[tuple[int, int], torch.Tensor] = {}
    evidence: dict[str, object] = {}
    validation_by_pair: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    modes = tuple((r13, r2) for r13 in (0, 1, 2) for r2 in (0, 1, 2))
    for r13, r2 in modes:
        training_candidate = F.linear(
            training_middle[r13], down["w2"][r2].reconstruction
        )
        validation_candidate = F.linear(
            validation_middle[r13], down["w2"][r2].reconstruction
        )
        fit_metric = functional_sse_by_request(
            training_reference[fit_mask.to(device=device)],
            training_candidate[fit_mask.to(device=device)],
            training_gates[fit_mask.to(device=device)],
            training_rows.request_steps[fit_mask],
            partition.fit,
        )
        confirmation_metric = functional_sse_by_request(
            training_reference[confirmation_mask.to(device=device)],
            training_candidate[confirmation_mask.to(device=device)],
            training_gates[confirmation_mask.to(device=device)],
            training_rows.request_steps[confirmation_mask],
            partition.confirmation,
        )
        validation_metric = functional_sse_by_request(
            validation_reference,
            validation_candidate,
            validation_gates,
            validation_rows.request_steps,
            validation_requests,
        )
        fit[(r13, r2)] = fit_metric[0]
        confirmation[(r13, r2)] = confirmation_metric[0]
        validation_by_pair[(r13, r2)] = validation_metric
        evidence[f"R{r13}/R{r2}"] = {
            "fit": _metric_json(fit_metric),
            "confirmation": _metric_json(confirmation_metric),
            "validation": _metric_json(validation_metric),
            "matrix_dense_h_proxy": {
                "w1": float(upstream["w1"][r13].proxy),
                "w3": float(upstream["w3"][r13].proxy),
                "w2": float(down["w2"][r2].proxy),
            },
        }

    reference_energy = next(iter(fit.values())).new_tensor(
        evidence["R0/R0"]["fit"]["document_reference_energy"]
    )
    confirmation_energy = next(iter(confirmation.values())).new_tensor(
        evidence["R0/R0"]["confirmation"]["document_reference_energy"]
    )
    fit_counts = next(iter(fit.values())).new_tensor(
        evidence["R0/R0"]["fit"]["document_rows"], dtype=torch.int64
    )
    confirmation_counts = next(iter(confirmation.values())).new_tensor(
        evidence["R0/R0"]["confirmation"]["document_rows"], dtype=torch.int64
    )
    selection = select_phase1_rate_pair(
        fit,
        confirmation,
        fit_counts=fit_counts,
        confirmation_counts=confirmation_counts,
        modes=modes,
        min_fit_documents=6,
        min_confirmation_documents=4,
        minimum_improvement=0.0,
        bootstrap_replicates=bootstrap_replicates,
        seed=deterministic_expert_seed(layer, expert),
    )
    selected_key = f"R{selection.selected[0]}/R{selection.selected[1]}"
    return {
        "selection": asdict(selection),
        "selected_validation": evidence[selected_key]["validation"],
        "modes": evidence,
    }


def _summarize(payload: dict) -> dict[str, object]:
    laws = [item["name"] for item in payload["signature"]["laws"]]
    expert_results = list(payload["results"].values())
    summary: dict[str, object] = {}
    for law in laws:
        law_rows = [row["laws"][law] for row in expert_results]
        modes: dict[str, object] = {}
        for mode in (f"R{r13}/R{r2}" for r13 in (0, 1, 2) for r2 in (0, 1, 2)):
            validation = [row["modes"][mode]["validation"] for row in law_rows]
            sse = sum(float(item["route_weighted_squared_error"]) for item in validation)
            energy = sum(
                float(item["route_weighted_reference_energy"]) for item in validation
            )
            modes[mode] = {
                "validation_sse": sse,
                "validation_nmse": sse / energy,
            }
        selected = [row["selected_validation"] for row in law_rows]
        selected_sse = sum(
            float(item["route_weighted_squared_error"]) for item in selected
        )
        selected_energy = sum(
            float(item["route_weighted_reference_energy"]) for item in selected
        )
        selections: dict[str, int] = {}
        r13: dict[str, int] = {}
        r2: dict[str, int] = {}
        for row in law_rows:
            selected_pair = (
                int(row["selection"]["selected_r13"]),
                int(row["selection"]["selected_r2"]),
            )
            name = f"R{selected_pair[0]}/R{selected_pair[1]}"
            selections[name] = selections.get(name, 0) + 1
            r13[f"R{selected_pair[0]}"] = r13.get(f"R{selected_pair[0]}", 0) + 1
            r2[f"R{selected_pair[1]}"] = r2.get(f"R{selected_pair[1]}", 0) + 1
        summary[law] = {
            "experts": len(law_rows),
            "fixed_modes": modes,
            "selected_validation_sse": selected_sse,
            "selected_validation_nmse": selected_sse / selected_energy,
            "selected_formats": selections,
            "selected_r13": r13,
            "selected_r2": r2,
        }
        r0_rows = [row["modes"]["R0/R0"]["validation"] for row in law_rows]
        w2_exchange: dict[str, object] = {}
        for r2_mode in (1, 2):
            shifted_rows = [
                row["modes"][f"R0/R{r2_mode}"]["validation"] for row in law_rows
            ]
            ratios = np.asarray(
                [
                    float(shifted["route_weighted_squared_error"])
                    / float(base["route_weighted_squared_error"])
                    for base, shifted in zip(r0_rows, shifted_rows, strict=True)
                ],
                dtype=np.float64,
            )
            base_sse = sum(
                float(item["route_weighted_squared_error"]) for item in r0_rows
            )
            shifted_sse = sum(
                float(item["route_weighted_squared_error"])
                for item in shifted_rows
            )
            w2_exchange[f"R{r2_mode}"] = {
                "experts_beating_r0": int(np.count_nonzero(ratios < 1.0)),
                "expert_win_fraction": float(np.mean(ratios < 1.0)),
                "aggregate_validation_sse_ratio_vs_r0": shifted_sse / base_sse,
                "aggregate_validation_sse_reduction": base_sse - shifted_sse,
                "median_expert_ratio_vs_r0": float(np.median(ratios)),
                "p90_expert_ratio_vs_r0": float(np.quantile(ratios, 0.90)),
                "worst_expert_ratio_vs_r0": float(ratios.max()),
            }
        summary[law]["w2_k2_k4_exchange"] = w2_exchange

    baseline = summary["normal"]
    for law in laws:
        row = summary[law]
        comparison: dict[str, object] = {
            "selected_validation_sse_ratio_vs_normal": (
                row["selected_validation_sse"] / baseline["selected_validation_sse"]
            )
        }
        for mode in row["fixed_modes"]:
            comparison[f"{mode}_validation_sse_ratio_vs_normal"] = (
                row["fixed_modes"][mode]["validation_sse"]
                / baseline["fixed_modes"][mode]["validation_sse"]
            )
        ratios = []
        wins = 0
        for expert_row in expert_results:
            trial = expert_row["laws"][law]["selected_validation"]
            base = expert_row["laws"]["normal"]["selected_validation"]
            ratio = float(trial["route_weighted_squared_error"]) / float(
                base["route_weighted_squared_error"]
            )
            ratios.append(ratio)
            wins += ratio < 1.0
        comparison["selected_validation_median_expert_ratio_vs_normal"] = float(
            np.median(ratios)
        )
        comparison["selected_validation_wins_vs_normal"] = wins
        row["comparison_vs_normal"] = comparison
    return summary


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the QSRT rate-shift law study requires CUDA")
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
    fallback_contexts, _ = layer_fallback_contexts(training_samples, partition.fit)
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )

    base_rank = sqg_cheb_normal_rank_e4m3_bytes()
    k3 = sqg_e4m3_bytes_from_rank_lut(3, base_rank)
    k4 = sqg_e4m3_bytes_from_rank_lut(4, base_rank)
    profile_specs = [(law, False) for law in args.laws]
    if args.include_k4_eight_stratum:
        profile_specs.extend((law, True) for law in args.laws)
    law_metadata = []
    for law, k4_eight_stratum in profile_specs:
        metadata = _lut_metadata(
            law,
            sqg_k2_tail_compressed_rank_e4m3_bytes(law.family, law.parameter),
        )
        metadata["name"] = _profile_name(
            law, k4_eight_stratum=k4_eight_stratum
        )
        metadata["k2_law_name"] = law.name
        metadata["k4_graph"] = (
            "k3-aligned-eight-stratum" if k4_eight_stratum else "native-sixteen-stratum"
        )
        law_metadata.append(metadata)
    if args.include_mcg_controls:
        law_metadata.extend(
            (
                {
                    "name": "mcg-k2",
                    "family": "mcg-k2_sqg-cheb-k3-k4",
                    "parameter": None,
                },
                {
                    "name": "mcg-all",
                    "family": "mcg-k2-k3-k4",
                    "parameter": None,
                },
            )
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
        "laws": law_metadata,
        "frozen_k3_transition_lut_sha256": hashlib.sha256(bytes(k3.tolist())).hexdigest(),
        "frozen_k4_transition_lut_sha256": hashlib.sha256(bytes(k4.tolist())).hexdigest(),
        "candidate_k4_q8_transition_lut_sha256": hashlib.sha256(
            bytes(
                sqg_k4_eight_stratum_e4m3_bytes_from_rank_lut(base_rank).tolist()
            )
        ).hexdigest(),
        "layout": "production qsrt_guarded_reuse",
        "mode_ids": [0, 1, 2],
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
            raise ValueError("resume output does not match this QSRT rate-shift law study")
        if payload.get("complete"):
            return payload
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        payload = {
            "kind": "kquant_kimi_k3_qsrt_rate_shift_law_study",
            "schema_version": 1,
            "signature": signature,
            "results": {},
            "complete": False,
        }
        _atomic_write(args.output, payload)

    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    luts_by_law: dict[str, dict[int, torch.Tensor | None]] = {
        _profile_name(law, k4_eight_stratum=k4_eight_stratum): _candidate_luts(
            law, device, k4_eight_stratum=k4_eight_stratum
        )
        for law, k4_eight_stratum in profile_specs
    }
    if args.include_mcg_controls:
        luts_by_law["mcg-k2"] = {2: None, 3: k3.to(device), 4: k4.to(device)}
        luts_by_law["mcg-all"] = {2: None, 3: None, 4: None}
    shared_scale_scope = (
        "qsrt-rate-shift-law-study",
        str(args.output.resolve()),
        args.layer,
    )
    _prime_shared_scales(
        store,
        layer=args.layer,
        global_h13=global_h13,
        fallback_contexts=fallback_contexts,
        device=device,
        shared_scale_scope=shared_scale_scope,
        quantizer_module=quantizer_module,
        luts=luts_by_law["normal"],
        ldlq_tf32=args.ldlq_tf32,
        tailbite_context=args.tailbite_context,
    )

    for expert in args.experts:
        key = str(expert)
        if key in payload["results"]:
            continue
        training_rows = select_expert_rows(training_samples, expert, partition.all)
        validation_rows = select_expert_rows(
            validation_samples, expert, validation_requests
        )
        fit_mask = _request_mask(training_rows.request_steps, partition.fit)
        confirmation_mask = _request_mask(
            training_rows.request_steps, partition.confirmation
        )
        if (
            int(torch.unique(training_rows.request_steps[fit_mask]).numel()) < 6
            or int(torch.unique(training_rows.request_steps[confirmation_mask]).numel()) < 4
            or validation_rows.documents == 0
        ):
            raise ValueError(f"panel expert {args.layer}/{expert} lacks document support")

        sources = {
            matrix: store.load_matrix(args.layer, expert, matrix, device=device)
            .float()
            .contiguous()
            for matrix in C.EXPERT_MATRICES
        }
        training_inputs = training_rows.inputs.float().to(device=device)
        training_gates = training_rows.gates.float().to(device=device)
        fit_inputs = training_inputs[fit_mask.to(device=device)]
        fit_gates = training_gates[fit_mask.to(device=device)]
        source_middle = F.silu(F.linear(fit_inputs, sources["w1"])) * F.linear(
            fit_inputs, sources["w3"]
        )
        contexts, block_scores = activation_block_contexts(source_middle, fit_gates)
        h13, h2, covariance = build_expert_hessians(
            fit_inputs,
            fit_gates,
            source_middle,
            global_h13=global_h13,
            global_h2=global_h2,
            device=device,
        )
        contexts = contexts.to(device=device, dtype=torch.long)
        laws: dict[str, object] = {}
        for law_name, rate_luts in luts_by_law.items():
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            upstream, down = _quantize_grid(
                sources,
                contexts,
                h13,
                h2,
                layer=args.layer,
                device=device,
                shared_scale_scope=shared_scale_scope,
                quantizer_module=quantizer_module,
                luts=rate_luts,
                ldlq_tf32=args.ldlq_tf32,
                tailbite_context=args.tailbite_context,
            )
            result = _score_grid(
                sources,
                upstream,
                down,
                training_rows=training_rows,
                fit_mask=fit_mask,
                confirmation_mask=confirmation_mask,
                validation_rows=validation_rows,
                partition=partition,
                validation_requests=validation_requests,
                device=device,
                bootstrap_replicates=args.bootstrap_replicates,
                layer=args.layer,
                expert=expert,
            )
            torch.cuda.synchronize(device)
            result["quantization_and_scoring_seconds"] = time.perf_counter() - started
            laws[law_name] = result
            del upstream, down
            torch.cuda.empty_cache()

        normal_r0 = laws["normal"]["modes"]["R0/R0"]
        for law_name in luts_by_law:
            if law_name in ("normal", "mcg-all"):
                continue
            trial_r0 = laws[law_name]["modes"]["R0/R0"]
            for fold in ("fit", "confirmation", "validation"):
                expected = float(normal_r0[fold]["route_weighted_squared_error"])
                observed = float(trial_r0[fold]["route_weighted_squared_error"])
                if observed != expected:
                    raise ValueError(
                        f"K2-only law changed K3 R0 closure for {args.layer}/{expert} "
                        f"{law_name} {fold}: {observed} != {expected}"
                    )

        payload["results"][key] = {
            "support": {
                "fit_rows": int(fit_mask.sum()),
                "confirmation_rows": int(confirmation_mask.sum()),
                "validation_rows": validation_rows.rows,
            },
            "covariance": covariance,
            "context": {
                "score_min": float(block_scores.min()),
                "score_median": float(block_scores.median()),
                "score_max": float(block_scores.max()),
            },
            "laws": laws,
            "r0_exact_across_k2_laws": True,
        }
        payload["summary"] = _summarize(payload)
        _atomic_write(args.output, payload)
        del sources, training_inputs, training_gates, source_middle, h13, h2
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
        "--laws",
        type=_parse_laws,
        default=_parse_laws("normal,asinh:4,rational:0.02"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ldlq-tf32", action="store_true")
    parser.add_argument("--tailbite-context", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument(
        "--include-mcg-controls",
        action="store_true",
        help=(
            "also evaluate MCG at K2 only and at all rates through the exact "
            "same encoder, transforms, Hessians, layouts, and selector"
        ),
    )
    parser.add_argument(
        "--include-k4-eight-stratum",
        action="store_true",
        help=(
            "duplicate each K2 law with a K3-aligned K4 labelling that exposes "
            "two history-conditioned reconstruction choices per octile"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.layer not in C.MOE_LAYERS:
        parser.error("--layer must be in 1..92")
    if not 1 <= args.tailbite_context <= 128:
        parser.error("--tailbite-context must be in 1..128")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
