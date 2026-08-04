#!/usr/bin/env python3
"""Fit free-FP16 and E4M3 labels on a fixed-window trellis graph.

This is the fixed-graph control requested by the reconstruction-alphabet
study.  For one selected window length it holds the rolling graph, path
syntax, transform, and matrix-local gain fixed, then alternates Viterbi path
assignment with a diagonal-metric Lloyd centroid update. Ridge and iteration
are selected on disjoint per-matrix tile samples; the final evaluation tiles
are untouched by fitting/selection. Running the same corpus seed at L10,
L12, L14, and L16 measures whether additional hidden state remains useful
after transition reconstructions are restricted to E4M3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from safetensors.torch import save_file
import torch

from kquant import constants as C
from kquant.exl3_reference import decode_mul1_states
from kquant.source_weights import OfficialMXFP4Store
from kquant.trellis_window import (
    fp8_constrained_codebook,
    geometry,
    mcg_codebook,
    refit_transition_codebook,
    tie_codebook_values,
    trellis_alias_metrics,
)
from scripts.experiment_trellis_window_fp8 import (
    TileCorpus,
    _atomic_json,
    _comparison,
    _encode,
    _paired_metrics,
    _parse_ints,
    collect_corpus,
    fit_group_gains,
)


def _parse_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result or any(item < 0.0 for item in result):
        raise argparse.ArgumentTypeError("ridge values must be nonempty and nonnegative")
    return result


def _slice_each_group(
    source: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    groups: tuple[dict[str, object], ...],
    start: int,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    samples = []
    ids = []
    for group in groups:
        group_id = int(group["group"])
        local = source[group_ids == group_id]
        stop = start + count
        if local.shape[0] < stop:
            raise ValueError(
                f"group {group_id} has {local.shape[0]} training tiles, needs {stop}"
            )
        samples.append(local[start:stop])
        ids.append(
            torch.full(
                (count,), group_id, dtype=group_ids.dtype, device=group_ids.device
            )
        )
    return torch.cat(samples), torch.cat(ids)


def _score(
    source: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    gains: torch.Tensor,
    codebook: torch.Tensor,
    bits: int,
    window_bits: int,
    max_trace_mib: int,
    return_transitions: bool,
):
    per_tile_gain = gains.index_select(0, group_ids.long())
    encoded, seconds = _encode(
        source,
        bits=bits,
        window_bits=window_bits,
        codebook=codebook,
        gain=per_tile_gain,
        max_trace_mib=max_trace_mib,
        return_transitions=return_transitions,
    )
    return encoded, float(encoded.squared_error.double().sum().item()), seconds


def _fit_family(
    fitting: torch.Tensor,
    fitting_group: torch.Tensor,
    selection: torch.Tensor,
    selection_group: torch.Tensor,
    *,
    gains: torch.Tensor,
    initial_codebook: torch.Tensor,
    reconstruction_dtype: torch.dtype,
    ridges: tuple[float, ...],
    iterations: int,
    bits: int,
    window_bits: int,
    max_trace_mib: int,
    family_name: str,
    tied_levels: int | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    fitting_gain = gains.index_select(0, fitting_group.long())
    fitting_targets = fitting * fitting_gain.unsqueeze(1)
    fitting_weights = fitting_gain.reciprocal().square().unsqueeze(1).expand_as(fitting)
    candidates: list[dict[str, object]] = []
    best_sse = float("inf")
    best_codebook: torch.Tensor | None = None
    best: dict[str, object] | None = None

    for ridge in ridges:
        codebook = initial_codebook.clone()
        history = []
        for iteration in range(iterations + 1):
            fitting_encoded, fitting_sse, fitting_seconds = _score(
                fitting,
                fitting_group,
                gains=gains,
                codebook=codebook,
                bits=bits,
                window_bits=window_bits,
                max_trace_mib=max_trace_mib,
                return_transitions=iteration < iterations,
            )
            _, selection_sse, selection_seconds = _score(
                selection,
                selection_group,
                gains=gains,
                codebook=codebook,
                bits=bits,
                window_bits=window_bits,
                max_trace_mib=max_trace_mib,
                return_transitions=False,
            )
            row = {
                "ridge_pseudocount": ridge,
                "iteration": iteration,
                "fitting_sse": fitting_sse,
                "selection_sse": selection_sse,
                "fitting_seconds": fitting_seconds,
                "selection_seconds": selection_seconds,
                "numeric_labels": int(torch.unique(codebook).numel()),
                "changed_from_initial_fraction": float(
                    (codebook != initial_codebook).float().mean().item()
                ),
            }
            history.append(row)
            if selection_sse < best_sse:
                best_sse = selection_sse
                best_codebook = codebook.clone()
                best = dict(row)
            if iteration == iterations:
                break
            assert fitting_encoded.transition_indices is not None
            fitted = refit_transition_codebook(
                fitting_targets,
                fitting_encoded.transition_indices,
                initial_codebook,
                weights=fitting_weights,
                ridge_pseudocount=ridge,
                reconstruction_dtype=(None if tied_levels is not None else reconstruction_dtype),
            )
            if tied_levels is None:
                codebook = fitted
            else:
                transition_weights = torch.full_like(
                    initial_codebook, ridge, dtype=torch.float32
                )
                transition_weights.scatter_add_(
                    0,
                    fitting_encoded.transition_indices.long().flatten(),
                    fitting_weights.float().flatten(),
                )
                codebook, _ = tie_codebook_values(
                    fitted,
                    tied_levels,
                    weights=transition_weights,
                    initial_centroids=torch.unique(initial_codebook),
                    reconstruction_dtype=reconstruction_dtype,
                )
        candidates.append({"ridge_pseudocount": ridge, "history": history})
        ridge_best = min(history, key=lambda row: row["selection_sse"])
        print(
            f"fit {family_name} ridge={ridge:g}: "
            f"best iteration={ridge_best['iteration']} "
            f"selection_sse={ridge_best['selection_sse']:.6g}",
            flush=True,
        )
    assert best_codebook is not None and best is not None
    return best_codebook, {"selected": best, "candidates": candidates}


def _save_codebooks(path: Path, codebooks: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in codebooks.items()},
        str(temporary),
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the learned-codebook experiment requires a CUDA device")
    if args.bits not in (2, 3, 4, 5) or args.window_bits not in (10, 12, 14, 16):
        raise ValueError("the controlled study accepts K2--K5 at L10/L12/L14/L16")
    if args.window_bits < 2 * args.bits:
        raise ValueError("the window must contain at least two path symbols")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if any(layer not in C.MOE_LAYERS for layer in args.layers):
        raise ValueError("all layers must be routed MoE layers")
    if any(expert < 0 or expert >= C.NUM_EXPERTS for expert in args.experts):
        raise ValueError("experts must be in 0..895")

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    training_count = args.gain_tiles + args.fitting_tiles + args.selection_tiles
    corpus: TileCorpus = collect_corpus(
        store,
        layers=args.layers,
        experts=args.experts,
        training_tiles_per_matrix=training_count,
        evaluation_tiles_per_matrix=args.evaluation_tiles,
        seed=args.seed,
        device=device,
    )
    gain_source, gain_group = _slice_each_group(
        corpus.training,
        corpus.training_group,
        groups=corpus.groups,
        start=0,
        count=args.gain_tiles,
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

    mcg = mcg_codebook(args.window_bits, device=device)
    mul1 = decode_mul1_states(
        torch.arange(1 << args.window_bits, dtype=torch.int64, device=device)
    ).float()
    e4m3_projected = fp8_constrained_codebook(mcg, dtype=torch.float8_e4m3fn)
    mul1_e4m3 = fp8_constrained_codebook(mul1, dtype=torch.float8_e4m3fn)
    procedural_codebooks = {
        "procedural_mcg_fp16": mcg,
        "procedural_mul1_fp16": mul1,
        "projected_mcg_e4m3": e4m3_projected,
        "projected_mul1_e4m3": mul1_e4m3,
    }
    gain_fit_by_family: dict[str, dict[str, object]] = {}
    gain_keys = (
        tuple(procedural_codebooks)
        if args.family_specific_gains
        else ("procedural_mcg_fp16",)
    )
    for name in gain_keys:
        gain_fit_by_family[name] = fit_group_gains(
            gain_source,
            gain_group,
            groups=corpus.groups,
            window_bits=args.window_bits,
            gain_fit_rate=args.bits,
            codebook=procedural_codebooks[name],
            max_trace_mib=args.max_trace_mib,
        )
    if not args.family_specific_gains:
        gain_fit_by_family.update(
            {
                name: gain_fit_by_family["procedural_mcg_fp16"]
                for name in procedural_codebooks
                if name != "procedural_mcg_fp16"
            }
        )
    gains_by_family = {
        name: fit["selected_gains_tensor"]
        for name, fit in gain_fit_by_family.items()
    }

    free_fp16, free_fit = _fit_family(
        fitting,
        fitting_group,
        selection,
        selection_group,
        gains=gains_by_family["procedural_mcg_fp16"],
        initial_codebook=mcg,
        reconstruction_dtype=torch.float16,
        ridges=args.ridges,
        iterations=args.iterations,
        bits=args.bits,
        window_bits=args.window_bits,
        max_trace_mib=args.max_trace_mib,
        family_name="free_fp16",
    )
    learned_e4m3, e4m3_fit = _fit_family(
        fitting,
        fitting_group,
        selection,
        selection_group,
        gains=gains_by_family["projected_mcg_e4m3"],
        initial_codebook=e4m3_projected,
        reconstruction_dtype=torch.float8_e4m3fn,
        ridges=args.ridges,
        iterations=args.iterations,
        bits=args.bits,
        window_bits=args.window_bits,
        max_trace_mib=args.max_trace_mib,
        family_name="e4m3",
    )
    mul1_free_fp16, mul1_free_fit = _fit_family(
        fitting,
        fitting_group,
        selection,
        selection_group,
        gains=gains_by_family["procedural_mul1_fp16"],
        initial_codebook=mul1,
        reconstruction_dtype=torch.float16,
        ridges=args.ridges,
        iterations=args.iterations,
        bits=args.bits,
        window_bits=args.window_bits,
        max_trace_mib=args.max_trace_mib,
        family_name="mul1_free_fp16",
    )
    mul1_learned_e4m3, mul1_e4m3_fit = _fit_family(
        fitting,
        fitting_group,
        selection,
        selection_group,
        gains=gains_by_family["projected_mul1_e4m3"],
        initial_codebook=mul1_e4m3,
        reconstruction_dtype=torch.float8_e4m3fn,
        ridges=args.ridges,
        iterations=args.iterations,
        bits=args.bits,
        window_bits=args.window_bits,
        max_trace_mib=args.max_trace_mib,
        family_name="mul1_e4m3",
    )
    tied_fp16, tied_fit = _fit_family(
        fitting,
        fitting_group,
        selection,
        selection_group,
        gains=gains_by_family["projected_mcg_e4m3"],
        initial_codebook=e4m3_projected,
        reconstruction_dtype=torch.float16,
        ridges=args.ridges,
        iterations=args.iterations,
        bits=args.bits,
        window_bits=args.window_bits,
        max_trace_mib=args.max_trace_mib,
        family_name="tied_fp16",
        tied_levels=int(torch.unique(e4m3_projected).numel()),
    )

    codebooks = {
        **procedural_codebooks,
        "learned_free_fp16": free_fp16,
        "learned_tied_fp16": tied_fp16,
        "learned_e4m3": learned_e4m3,
        "learned_mul1_free_fp16": mul1_free_fp16,
        "learned_mul1_e4m3": mul1_learned_e4m3,
    }
    codebook_gains = {
        "procedural_mcg_fp16": gains_by_family["procedural_mcg_fp16"],
        "procedural_mul1_fp16": gains_by_family["procedural_mul1_fp16"],
        "projected_mcg_e4m3": gains_by_family["projected_mcg_e4m3"],
        "projected_mul1_e4m3": gains_by_family["projected_mul1_e4m3"],
        "learned_free_fp16": gains_by_family["procedural_mcg_fp16"],
        "learned_tied_fp16": gains_by_family["projected_mcg_e4m3"],
        "learned_e4m3": gains_by_family["projected_mcg_e4m3"],
        "learned_mul1_free_fp16": gains_by_family["procedural_mul1_fp16"],
        "learned_mul1_e4m3": gains_by_family["projected_mul1_e4m3"],
    }
    metrics: dict[str, object] = {}
    for name, codebook in codebooks.items():
        encoded, _, seconds = _score(
            corpus.evaluation,
            corpus.evaluation_group,
            gains=codebook_gains[name],
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
    baseline = metrics["procedural_mcg_fp16"]
    comparisons = {
        name: _comparison(baseline, result)
        for name, result in metrics.items()
        if name != "procedural_mcg_fp16"
    }
    free_sse = float(metrics["learned_free_fp16"]["sse"])
    projected_sse = float(metrics["projected_mcg_e4m3"]["sse"])
    learned_e4m3_sse = float(metrics["learned_e4m3"]["sse"])
    tied_fp16_sse = float(metrics["learned_tied_fp16"]["sse"])
    gap = projected_sse - free_sse

    graph = geometry(args.bits, args.window_bits)
    payload: dict[str, object] = {
        "kind": "kquant_trellis_learned_codebook_experiment",
        "schema_version": 1,
        "signature": {
            "source_checkpoint": str(store.root),
            "source_revision": store.revision,
            "layers": list(args.layers),
            "experts": list(args.experts),
            "matrices": list(C.EXPERT_MATRICES),
            "bits": args.bits,
            "window_bits": args.window_bits,
            "gain_tiles_per_matrix": args.gain_tiles,
            "fitting_tiles_per_matrix": args.fitting_tiles,
            "selection_tiles_per_matrix": args.selection_tiles,
            "evaluation_tiles_per_matrix": args.evaluation_tiles,
            "ridges": list(args.ridges),
            "iterations": args.iterations,
            "family_specific_gains": bool(args.family_specific_gains),
            "seed": args.seed,
            "scale_contract": (
                "held-fixed matrix-local gains fitted on disjoint tiles; optionally one "
                "fit per procedural family; inverse gain can be folded into existing "
                "EXL outer scales; no MXFP8 block scales"
            ),
            "objective": (
                "diagonal tile SSE in the production sign/RMS/Hadamard domain; "
                "matrix gain weighting exactly undone in the centroid update"
            ),
        },
        "corpus": {
            "groups": list(corpus.groups),
            "evaluation_coefficients": int(corpus.evaluation.numel()),
        },
        "gain_fit": {
            name: {
                key: value
                for key, value in fit.items()
                if not key.endswith("_tensor")
            }
            for name, fit in gain_fit_by_family.items()
        },
        "fits": {
            "learned_free_fp16": free_fit,
            "learned_tied_fp16": tied_fit,
            "learned_e4m3": e4m3_fit,
            "learned_mul1_free_fp16": mul1_free_fit,
            "learned_mul1_e4m3": mul1_e4m3_fit,
        },
        "metrics": metrics,
        "comparisons_to_procedural_mcg": comparisons,
        "e4m3_gap_recovery_fraction": (
            (projected_sse - learned_e4m3_sse) / gap if gap > 0.0 else None
        ),
        "tied_fp16_gap_recovery_fraction": (
            (projected_sse - tied_fp16_sse) / gap if gap > 0.0 else None
        ),
        "storage_controls": {
            "path_bits_per_weight": args.bits,
            "rolling_state_bits": args.window_bits - args.bits,
            "states": graph.states,
            "transitions_per_coefficient": graph.transitions,
            "viterbi_work_relative_to_l16": graph.transitions / float(1 << 16),
            "global_fp16_table_bytes": int(free_fp16.numel() * 2),
            "global_e4m3_table_bytes": int(learned_e4m3.numel()),
            "global_tied_fp16_table_bytes": int(tied_fp16.numel() + 2 * torch.unique(tied_fp16).numel()),
            "note": (
                "Tables are global and amortizable, but their lookup cost is not free; "
                "a production E4M3 generator still requires a procedural or cache-efficient design."
            ),
        },
    }
    _save_codebooks(args.codebooks, codebooks)
    _atomic_json(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=_parse_ints, default=(24,))
    parser.add_argument("--experts", type=_parse_ints, default=(0, 179, 358, 537, 716, 895))
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--window-bits", type=int, default=16)
    parser.add_argument("--gain-tiles", type=int, default=8)
    parser.add_argument("--fitting-tiles", type=int, default=96)
    parser.add_argument("--selection-tiles", type=int, default=32)
    parser.add_argument("--evaluation-tiles", type=int, default=128)
    parser.add_argument("--ridges", type=_parse_floats, default=(0.0, 1.0, 4.0, 16.0))
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument(
        "--family-specific-gains",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-trace-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/trellis-learned-codebooks-l24.json"),
    )
    parser.add_argument(
        "--codebooks",
        type=Path,
        default=Path("out/trellis-learned-codebooks-l24.safetensors"),
    )
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({
        "output": str(args.output),
        "codebooks": str(args.codebooks),
        "comparisons": payload["comparisons_to_procedural_mcg"],
        "e4m3_gap_recovery_fraction": payload["e4m3_gap_recovery_fraction"],
        "tied_fp16_gap_recovery_fraction": payload["tied_fp16_gap_recovery_fraction"],
    }, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
