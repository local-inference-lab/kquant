#!/usr/bin/env python3
"""Evaluate an ~8 KiB learned E4M3 subcode over an L16 procedural graph.

An MCG or MUL1 decoder predicts one E4M3 anchor byte for every transition. A
learned bit plane then selects between two predictor-conditioned E4M3 bytes.
The full L16 graph is retained while reconstruction storage falls from 64 KiB
to 8 KiB for the bit plane plus either 512 bytes for two explicit outputs or
256 bytes when the primary output is the projected-MCG anchor.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from kquant import constants as C
from kquant.e4m3_subcode import (
    ConditionalBinaryE4M3Fit,
    decode_predictor_anchor_plus_alternative,
    e4m3_codes,
    fit_conditional_binary_e4m3,
    logical_conditional_binary_e4m3_bytes,
    pack_transition_bits,
)
from kquant.exl3_reference import decode_mul1_states
from kquant.source_weights import OfficialMXFP4Store
from kquant.trellis_window import (
    fp8_constrained_codebook,
    geometry,
    trellis_alias_metrics,
)
from scripts.experiment_trellis_e4m3_state_palettes import (
    _parse_floats,
    _save_tensors,
    _transition_statistics,
    _validate_groups,
    _validate_oracle_signature,
)
from scripts.experiment_trellis_learned_codebooks import _score, _slice_each_group
from scripts.experiment_trellis_window_fp8 import (
    _atomic_json,
    _comparison,
    _paired_metrics,
    _parse_ints,
    collect_corpus,
    fit_group_gains,
)


def _candidate_name(predictor_name: str, fixed_primary: bool) -> str:
    return (
        f"binary_e4m3_{predictor_name}_anchor_plus_alt"
        if fixed_primary
        else f"binary_e4m3_{predictor_name}_conditional_pair"
    )


def _fit_statistics(
    fit: ConditionalBinaryE4M3Fit,
    predictor: torch.Tensor,
    oracle: torch.Tensor,
) -> dict[str, object]:
    counts = torch.bincount(fit.transition_bits.long(), minlength=2)
    probabilities = counts.double() / counts.sum()
    entropy = -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log2()).sum()
    anchors = e4m3_codes(predictor).long()
    used_anchor = torch.bincount(anchors, minlength=256) > 0
    used_pairs = fit.pair_codes[used_anchor]
    return {
        "objective": fit.objective,
        "lloyd_iterations": fit.iterations,
        "bit_counts": [int(value) for value in counts.tolist()],
        "bit_entropy": float(entropy.item()),
        "anchor_codes_used": int(torch.count_nonzero(used_anchor).item()),
        "used_pair_rows_with_distinct_outputs": int(
            torch.count_nonzero(used_pairs[:, 0] != used_pairs[:, 1]).item()
        ),
        "numeric_outputs": int(torch.unique(fit.codebook).numel()),
        "equals_full_oracle_fraction": float(
            (fit.codebook == oracle).float().mean().item()
        ),
        "equals_projected_predictor_fraction": float(
            (fit.codebook == fp8_constrained_codebook(predictor)).float().mean().item()
        ),
    }


def _fit_candidate(
    fitting: torch.Tensor,
    fitting_group: torch.Tensor,
    selection: torch.Tensor,
    selection_group: torch.Tensor,
    *,
    gains: torch.Tensor,
    predictor: torch.Tensor,
    predictor_name: str,
    full_e4m3: torch.Tensor,
    ridge_prior: torch.Tensor,
    bootstrap_alternatives: bool,
    fixed_primary: bool,
    ridges: tuple[float, ...],
    outer_iterations: int,
    table_iterations: int,
    bits: int,
    window_bits: int,
    max_trace_mib: int,
) -> tuple[ConditionalBinaryE4M3Fit, dict[str, object]]:
    # This initialization uses only the already-fitted training artifact. It
    # does not inspect selection or evaluation coefficients.
    initial = fit_conditional_binary_e4m3(
        ridge_prior,
        torch.ones_like(ridge_prior),
        predictor,
        iterations=table_iterations,
        fixed_primary=fixed_primary,
    )
    fitting_gain = gains.index_select(0, fitting_group.long())
    fitting_targets = fitting * fitting_gain.unsqueeze(1)
    fitting_weights = fitting_gain.reciprocal().square().unsqueeze(1).expand_as(fitting)
    initial_encoded, initial_fitting_sse, initial_fitting_seconds = _score(
        fitting,
        fitting_group,
        gains=gains,
        codebook=initial.codebook,
        bits=bits,
        window_bits=window_bits,
        max_trace_mib=max_trace_mib,
        return_transitions=True,
    )
    assert initial_encoded.transition_indices is not None
    _, initial_selection_sse, initial_selection_seconds = _score(
        selection,
        selection_group,
        gains=gains,
        codebook=initial.codebook,
        bits=bits,
        window_bits=window_bits,
        max_trace_mib=max_trace_mib,
        return_transitions=False,
    )
    best_fit = initial
    best_sse = initial_selection_sse
    best_row: dict[str, object] = {
        "source": "full_learned_e4m3_projection",
        "ridge_pseudocount": None,
        "outer_iteration": -1,
        "fitting_sse": initial_fitting_sse,
        "selection_sse": initial_selection_sse,
        "fitting_seconds": initial_fitting_seconds,
        "selection_seconds": initial_selection_seconds,
        **_fit_statistics(initial, predictor, full_e4m3),
    }
    candidates = []
    for ridge in ridges:
        paths = initial_encoded.transition_indices
        previous = initial
        history = []
        for outer_iteration in range(outer_iterations):
            transition_targets, transition_weights = _transition_statistics(
                fitting_targets,
                fitting_weights,
                paths,
                ridge_prior,
                ridge_pseudocount=ridge,
            )
            fit = fit_conditional_binary_e4m3(
                transition_targets,
                transition_weights,
                predictor,
                iterations=table_iterations,
                fixed_primary=fixed_primary,
                initial_pair_codes=(
                    None
                    if bootstrap_alternatives and outer_iteration == 0
                    else previous.pair_codes
                ),
                initial_transition_bits=(
                    None
                    if bootstrap_alternatives and outer_iteration == 0
                    else previous.transition_bits
                ),
            )
            fitting_encoded, fitting_sse, fitting_seconds = _score(
                fitting,
                fitting_group,
                gains=gains,
                codebook=fit.codebook,
                bits=bits,
                window_bits=window_bits,
                max_trace_mib=max_trace_mib,
                return_transitions=True,
            )
            assert fitting_encoded.transition_indices is not None
            _, selection_sse, selection_seconds = _score(
                selection,
                selection_group,
                gains=gains,
                codebook=fit.codebook,
                bits=bits,
                window_bits=window_bits,
                max_trace_mib=max_trace_mib,
                return_transitions=False,
            )
            row = {
                "source": "path_refit",
                "ridge_pseudocount": ridge,
                "outer_iteration": outer_iteration,
                "fitting_sse": fitting_sse,
                "selection_sse": selection_sse,
                "fitting_seconds": fitting_seconds,
                "selection_seconds": selection_seconds,
                **_fit_statistics(fit, predictor, full_e4m3),
            }
            history.append(row)
            if selection_sse < best_sse:
                best_sse = selection_sse
                best_fit = replace(
                    fit,
                    codebook=fit.codebook.clone(),
                    transition_bits=fit.transition_bits.clone(),
                    pair_codes=fit.pair_codes.clone(),
                    pair_values=fit.pair_values.clone(),
                )
                best_row = dict(row)
            paths = fitting_encoded.transition_indices
            previous = fit
        candidates.append({"ridge_pseudocount": ridge, "history": history})
        ridge_best = min(history, key=lambda row: float(row["selection_sse"]))
        print(
            f"{_candidate_name(predictor_name, fixed_primary)} ridge={ridge:g}: "
            f"best outer={ridge_best['outer_iteration']} "
            f"selection_sse={ridge_best['selection_sse']:.8g}",
            flush=True,
        )
    return best_fit, {"selected": best_row, "candidates": candidates}


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the binary-subcode experiment requires CUDA")
    if args.bits != 3 or args.window_bits != 16:
        raise ValueError("this controlled study requires K3/L16")
    graph = geometry(args.bits, args.window_bits)
    if args.outer_iterations < 1 or args.table_iterations < 1:
        raise ValueError("iteration counts must be positive")

    oracle = json.loads(args.oracle_json.read_text())
    _validate_oracle_signature(oracle, args)
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    training_count = args.gain_tiles + args.fitting_tiles + args.selection_tiles
    corpus = collect_corpus(
        store,
        layers=args.layers,
        experts=args.experts,
        training_tiles_per_matrix=training_count,
        evaluation_tiles_per_matrix=args.evaluation_tiles,
        seed=args.seed,
        device=device,
    )
    fitting, fitting_group = _slice_each_group(
        corpus.training,
        corpus.training_group,
        groups=corpus.groups,
        start=args.gain_tiles,
        count=args.fitting_tiles,
    )
    selection, selection_group = _slice_each_group(
        corpus.training,
        corpus.training_group,
        groups=corpus.groups,
        start=args.gain_tiles + args.fitting_tiles,
        count=args.selection_tiles,
    )
    gain_source, gain_group = _slice_each_group(
        corpus.training,
        corpus.training_group,
        groups=corpus.groups,
        start=0,
        count=args.gain_tiles,
    )
    gain_rows = oracle["gain_fit"]["selected_gains"]
    _validate_groups(corpus, gain_rows)
    gains = torch.tensor(
        [float(row["gain"]) for row in gain_rows],
        dtype=corpus.training.dtype,
        device=device,
    )
    oracle_tensors = load_file(str(args.oracle_codebooks), device=str(device))
    mcg = oracle_tensors["procedural_mcg_fp16"].float()
    mul1 = decode_mul1_states(
        torch.arange(graph.transitions, device=device)
    ).float()
    full_e4m3 = oracle_tensors["learned_e4m3"].float()
    predictors = {"mcg": mcg, "mul1": mul1}
    projected_predictors = {
        name: fp8_constrained_codebook(predictor)
        for name, predictor in predictors.items()
    }
    mul1_gain_fit = fit_group_gains(
        gain_source,
        gain_group,
        groups=corpus.groups,
        window_bits=args.window_bits,
        gain_fit_rate=3,
        codebook=projected_predictors["mul1"],
        max_trace_mib=args.max_trace_mib,
    )
    gains_by_predictor = {
        "mcg": gains,
        "mul1": mul1_gain_fit["selected_gains_tensor"],
    }

    selected_fits: dict[tuple[str, bool], ConditionalBinaryE4M3Fit] = {}
    fit_reports = {}
    primary_modes = (True,) if args.fixed_primary_only else (True, False)
    for predictor_name, predictor in predictors.items():
        for fixed_primary in primary_modes:
            name = _candidate_name(predictor_name, fixed_primary)
            print(f"fitting {name}", flush=True)
            fit, report = _fit_candidate(
                fitting,
                fitting_group,
                selection,
                selection_group,
                gains=gains_by_predictor[predictor_name],
                predictor=predictor,
                predictor_name=predictor_name,
                full_e4m3=full_e4m3,
                ridge_prior=(
                    full_e4m3
                    if predictor_name == "mcg"
                    else projected_predictors[predictor_name]
                ),
                bootstrap_alternatives=predictor_name == "mul1",
                fixed_primary=fixed_primary,
                ridges=args.ridges,
                outer_iterations=args.outer_iterations,
                table_iterations=args.table_iterations,
                bits=args.bits,
                window_bits=args.window_bits,
                max_trace_mib=args.max_trace_mib,
            )
            selected_fits[(predictor_name, fixed_primary)] = fit
            fit_reports[name] = report

    codebooks = {
        "procedural_mcg_fp16": mcg,
        "procedural_mul1_fp16": mul1,
        "projected_mcg_e4m3": projected_predictors["mcg"],
        "projected_mul1_e4m3": projected_predictors["mul1"],
        "full_learned_e4m3": full_e4m3,
        **{
            _candidate_name(predictor_name, fixed_primary): fit.codebook
            for (predictor_name, fixed_primary), fit in selected_fits.items()
        },
    }
    metrics = {}
    for name, codebook in codebooks.items():
        predictor_name = "mul1" if "mul1" in name else "mcg"
        encoded, _, seconds = _score(
            corpus.evaluation,
            corpus.evaluation_group,
            gains=gains_by_predictor[predictor_name],
            codebook=codebook,
            bits=args.bits,
            window_bits=args.window_bits,
            max_trace_mib=args.max_trace_mib,
            return_transitions=True,
        )
        assert encoded.transition_indices is not None
        metrics[name] = {
            **_paired_metrics(
                corpus.evaluation,
                encoded.reconstructed,
                corpus.evaluation_group,
                corpus.groups,
            ),
            "seconds": seconds,
            "alias_structure": trellis_alias_metrics(
                codebook,
                args.bits,
                args.window_bits,
                transition_indices=encoded.transition_indices,
            ),
        }

    comparisons_to_mcg = {
        name: _comparison(metrics["procedural_mcg_fp16"], result)
        for name, result in metrics.items()
        if name != "procedural_mcg_fp16"
    }
    comparisons_to_full = {
        name: _comparison(metrics["full_learned_e4m3"], result)
        for name, result in metrics.items()
        if name.startswith("binary_e4m3_")
    }
    comparisons_to_mul1 = {
        name: _comparison(metrics["procedural_mul1_fp16"], result)
        for name, result in metrics.items()
        if name != "procedural_mul1_fp16"
    }
    storage = {}
    for predictor_name, fixed_primary in selected_fits:
        name = _candidate_name(predictor_name, fixed_primary)
        total = logical_conditional_binary_e4m3_bytes(
            graph.transitions, fixed_primary=fixed_primary
        )
        storage[name] = {
            "transition_bitplane_bytes": graph.transitions // 8,
            "pair_table_bytes": 256 if fixed_primary else 512,
            "total_bytes": total,
            "total_kib": total / 1024.0,
            "smem_if_pair_table_is_constant_bytes": graph.transitions // 8,
            "fraction_of_full_e4m3_table": total / graph.transitions,
        }

    payload: dict[str, object] = {
        "kind": "kquant_trellis_e4m3_binary_subcode_experiment",
        "schema_version": 2,
        "signature": {
            **oracle["signature"],
            "ridges": list(args.ridges),
            "outer_iterations": args.outer_iterations,
            "table_iterations": args.table_iterations,
            "oracle_json": str(args.oracle_json),
            "oracle_codebooks": str(args.oracle_codebooks),
            "factorization": (
                "procedural MCG or MUL1 projected to an E4M3 anchor; one learned bit per "
                "L16 transition selects a predictor-conditioned E4M3 output"
            ),
            "procedural_predictors": list(predictors),
        },
        "corpus": {
            "groups": list(corpus.groups),
            "evaluation_coefficients": int(corpus.evaluation.numel()),
        },
        "fits": fit_reports,
        "gain_fits": {
            "mcg": oracle["gain_fit"],
            "mul1_projected_e4m3": {
                key: value
                for key, value in mul1_gain_fit.items()
                if key != "selected_gains_tensor"
            },
        },
        "metrics": metrics,
        "comparisons_to_procedural_mcg": comparisons_to_mcg,
        "comparisons_to_procedural_mul1": comparisons_to_mul1,
        "comparisons_to_full_learned_e4m3": comparisons_to_full,
        "storage": {
            "full_l16_e4m3_table_bytes": graph.transitions,
            "candidates": storage,
            "note": (
                "The 256- or 512-byte pair table is global and may live in constant "
                "storage; the only mandatory 8-KiB random-access plane is the learned "
                "transition bitset. Kernel placement remains to be benchmarked."
            ),
        },
    }
    tensors = {
        "procedural_mcg_fp16": mcg,
        "procedural_mul1_fp16": mul1,
        "projected_mcg_e4m3": projected_predictors["mcg"],
        "projected_mul1_e4m3": projected_predictors["mul1"],
        "full_learned_e4m3": full_e4m3,
    }
    for (predictor_name, fixed_primary), fit in selected_fits.items():
        prefix = _candidate_name(predictor_name, fixed_primary)
        tensors[f"{prefix}.expanded"] = fit.codebook
        tensors[f"{prefix}.transition_bits"] = fit.transition_bits
        tensors[f"{prefix}.pair_codes"] = fit.pair_codes
        tensors[f"{prefix}.packed_transition_bits"] = pack_transition_bits(
            fit.transition_bits
        )
        if fixed_primary:
            alternatives = fit.pair_codes[:, 1].contiguous()
            reconstructed = decode_predictor_anchor_plus_alternative(
                predictors[predictor_name],
                tensors[f"{prefix}.packed_transition_bits"],
                alternatives,
            )
            if not torch.equal(reconstructed, fit.codebook):
                raise RuntimeError("fixed-primary packed decoder failed exact closure")
            tensors[f"{prefix}.alternative_codes"] = alternatives
    _save_tensors(args.tensors, tensors)
    _atomic_json(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=_parse_ints, default=(1, 24, 40))
    parser.add_argument(
        "--experts", type=_parse_ints, default=(0, 179, 358, 537, 716, 895)
    )
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--window-bits", type=int, default=16)
    parser.add_argument("--gain-tiles", type=int, default=8)
    parser.add_argument("--fitting-tiles", type=int, default=256)
    parser.add_argument("--selection-tiles", type=int, default=64)
    parser.add_argument("--evaluation-tiles", type=int, default=128)
    parser.add_argument("--ridges", type=_parse_floats, default=(1.0,))
    parser.add_argument("--outer-iterations", type=int, default=2)
    parser.add_argument("--table-iterations", type=int, default=16)
    parser.add_argument(
        "--fixed-primary-only",
        action="store_true",
        help="fit only the deployable one-bit anchor-plus-alternative form",
    )
    parser.add_argument("--max-trace-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument(
        "--oracle-json",
        type=Path,
        default=Path("out/trellis-learned-codebooks-l1-l24-l40-shared-l16-v2.json"),
    )
    parser.add_argument(
        "--oracle-codebooks",
        type=Path,
        default=Path(
            "out/trellis-learned-codebooks-l1-l24-l40-shared-l16-v2.safetensors"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/trellis-e4m3-binary-subcode-predictors-v1.json"),
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=Path("out/trellis-e4m3-binary-subcode-predictors-v1.safetensors"),
    )
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tensors": str(args.tensors),
                "comparisons_to_procedural_mcg": payload[
                    "comparisons_to_procedural_mcg"
                ],
                "comparisons_to_procedural_mul1": payload[
                    "comparisons_to_procedural_mul1"
                ],
                "comparisons_to_full_learned_e4m3": payload[
                    "comparisons_to_full_learned_e4m3"
                ],
                "storage": payload["storage"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
