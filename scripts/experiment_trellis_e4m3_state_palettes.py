#!/usr/bin/env python3
"""Compress a learned L16 E4M3 transition table with state palettes.

The rolling K3/L16 graph remains unchanged: 8,192 origin states each expose
eight branches.  Only the reconstruction function is factorized.  Every
origin state selects one learned eight-value E4M3 palette row, so decoding is
equivalent to::

    palette = state_to_palette[origin_state]
    value = palettes[palette, branch]

This driver alternates Viterbi path assignment with weighted palette fitting.
Ridge and outer iteration are selected on disjoint tiles, and evaluation
tiles are untouched.  It deliberately reuses the matrix gains and the full
learned E4M3 table from a seeded fixed-graph experiment so the only new
variable is reconstruction-table factorization.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

from safetensors.torch import load_file, save_file
import torch

from kquant import constants as C
from kquant.source_weights import OfficialMXFP4Store
from kquant.state_palette import (
    StatePaletteFit,
    StateSubpaletteFit,
    fit_state_palette,
    fit_state_subpalettes,
    logical_state_palette_bytes,
    logical_state_subpalette_bytes,
)
from kquant.trellis_window import geometry, trellis_alias_metrics
from scripts.experiment_trellis_learned_codebooks import _score, _slice_each_group
from scripts.experiment_trellis_window_fp8 import (
    TileCorpus,
    _atomic_json,
    _comparison,
    _paired_metrics,
    _parse_ints,
    collect_corpus,
)


PaletteFit = StatePaletteFit | StateSubpaletteFit


def _candidate_name(classes: int, branch_groups: int) -> str:
    if branch_groups == 1:
        return f"state_palette_e4m3_c{classes}"
    return f"state_subpalette_e4m3_g{branch_groups}_c{classes}"


def _parse_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("ridge values must be positive and finite")
    return result


def _save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
        str(temporary),
    )
    temporary.replace(path)


def _transition_statistics(
    targets: torch.Tensor,
    weights: torch.Tensor,
    transition_indices: torch.Tensor,
    prior: torch.Tensor,
    *,
    ridge_pseudocount: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce fixed-path coefficient targets into per-transition statistics."""

    if targets.shape != weights.shape or targets.shape != transition_indices.shape:
        raise ValueError("targets, weights, and transitions must share a shape")
    if targets.device != prior.device or weights.device != prior.device:
        raise ValueError("transition statistics must share a device")
    if ridge_pseudocount <= 0.0 or not math.isfinite(ridge_pseudocount):
        raise ValueError("ridge_pseudocount must be positive and finite")
    indices = transition_indices.long().flatten()
    total_weight = torch.full_like(prior, ridge_pseudocount, dtype=torch.float32)
    weighted_sum = ridge_pseudocount * prior.float()
    total_weight.scatter_add_(0, indices, weights.float().flatten())
    weighted_sum.scatter_add_(
        0, indices, weights.float().flatten() * targets.float().flatten()
    )
    return weighted_sum / total_weight, total_weight


def _class_statistics(
    fit: PaletteFit,
    transition_indices: torch.Tensor,
    *,
    bits: int,
) -> dict[str, object]:
    assignments = fit.state_to_palette.long()
    palettes = fit.palettes
    if assignments.ndim == 1:
        assignments = assignments.unsqueeze(0)
        palettes = palettes.unsqueeze(0)
    branch_groups, states = assignments.shape
    classes = palettes.shape[1]
    branches = 1 << bits
    if branches % branch_groups:
        raise ValueError("fit branch groups do not divide trellis branches")
    branches_per_group = branches // branch_groups
    offsets = (
        torch.arange(branch_groups, device=assignments.device) * classes
    ).unsqueeze(1)
    state_counts = torch.bincount(
        (assignments + offsets).flatten(), minlength=branch_groups * classes
    )
    transitions = transition_indices.long().flatten()
    path_states = transitions >> bits
    path_groups = (transitions & (branches - 1)) // branches_per_group
    path_classes = assignments[path_groups, path_states]
    path_counts = torch.bincount(
        path_groups * classes + path_classes, minlength=branch_groups * classes
    )

    def entropy(counts: torch.Tensor) -> float:
        probabilities = counts[counts > 0].double()
        probabilities /= probabilities.sum()
        return float((-(probabilities * probabilities.log2()).sum()).item())

    return {
        "branch_groups": branch_groups,
        "states": states,
        "classes_per_branch_group": classes,
        "classes_used_by_states": int(torch.count_nonzero(state_counts).item()),
        "minimum_states_per_used_class": int(state_counts[state_counts > 0].min().item()),
        "maximum_states_per_class": int(state_counts.max().item()),
        "state_class_entropy_bits": entropy(state_counts),
        "classes_used_by_fitting_paths": int(torch.count_nonzero(path_counts).item()),
        "fitting_path_class_entropy_bits": entropy(path_counts),
        "numeric_labels": int(torch.unique(fit.palettes).numel()),
    }


def _fit_candidate(
    fitting: torch.Tensor,
    fitting_group: torch.Tensor,
    selection: torch.Tensor,
    selection_group: torch.Tensor,
    *,
    gains: torch.Tensor,
    full_e4m3: torch.Tensor,
    classes: int,
    ridges: tuple[float, ...],
    outer_iterations: int,
    palette_iterations: int,
    bits: int,
    window_bits: int,
    max_trace_mib: int,
    branch_groups: int = 1,
    initial_fit: PaletteFit | None = None,
) -> tuple[PaletteFit, dict[str, object]]:
    fitting_gain = gains.index_select(0, fitting_group.long())
    fitting_targets = fitting * fitting_gain.unsqueeze(1)
    fitting_weights = fitting_gain.reciprocal().square().unsqueeze(1).expand_as(fitting)

    oracle_encoded, oracle_fitting_sse, oracle_seconds = _score(
        fitting,
        fitting_group,
        gains=gains,
        codebook=full_e4m3,
        bits=bits,
        window_bits=window_bits,
        max_trace_mib=max_trace_mib,
        return_transitions=True,
    )
    assert oracle_encoded.transition_indices is not None

    best_sse = float("inf")
    best_fit: PaletteFit | None = None
    best_row: dict[str, object] | None = None
    initial_encoded = oracle_encoded
    if initial_fit is not None:
        initial_encoded, initial_fitting_sse, initial_fitting_seconds = _score(
            fitting,
            fitting_group,
            gains=gains,
            codebook=initial_fit.codebook,
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
            codebook=initial_fit.codebook,
            bits=bits,
            window_bits=window_bits,
            max_trace_mib=max_trace_mib,
            return_transitions=False,
        )
        best_sse = initial_selection_sse
        best_fit = replace(
            initial_fit,
            codebook=initial_fit.codebook.clone(),
            state_to_palette=initial_fit.state_to_palette.clone(),
            palettes=initial_fit.palettes.clone(),
        )
        best_row = {
            "ridge_pseudocount": None,
            "outer_iteration": -1,
            "source": "shared_initialization",
            "palette_lloyd_iterations": 0,
            "palette_objective": initial_fit.objective,
            "fitting_sse": initial_fitting_sse,
            "selection_sse": initial_selection_sse,
            "fitting_seconds": initial_fitting_seconds,
            "selection_seconds": initial_selection_seconds,
            **_class_statistics(
                initial_fit, initial_encoded.transition_indices, bits=bits
            ),
        }
    candidates = []
    for ridge in ridges:
        assert initial_encoded.transition_indices is not None
        paths = initial_encoded.transition_indices
        previous = initial_fit
        history = []
        for outer_iteration in range(outer_iterations):
            transition_targets, transition_weights = _transition_statistics(
                fitting_targets,
                fitting_weights,
                paths,
                full_e4m3,
                ridge_pseudocount=ridge,
            )
            common_fit_args = {
                "bits": bits,
                "window_bits": window_bits,
                "classes": classes,
                "reconstruction_dtype": torch.float8_e4m3fn,
                "iterations": palette_iterations,
                "initial_state_to_palette": (
                    None if previous is None else previous.state_to_palette
                ),
                "initial_palettes": None if previous is None else previous.palettes,
            }
            if branch_groups == 1:
                fit = fit_state_palette(
                    transition_targets,
                    transition_weights,
                    **common_fit_args,
                )
            else:
                fit = fit_state_subpalettes(
                    transition_targets,
                    transition_weights,
                    branch_groups=branch_groups,
                    **common_fit_args,
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
                "ridge_pseudocount": ridge,
                "outer_iteration": outer_iteration,
                "source": "scope_refit",
                "palette_lloyd_iterations": fit.iterations,
                "palette_objective": fit.objective,
                "fitting_sse": fitting_sse,
                "selection_sse": selection_sse,
                "fitting_seconds": fitting_seconds,
                "selection_seconds": selection_seconds,
                **_class_statistics(
                    fit, fitting_encoded.transition_indices, bits=bits
                ),
            }
            history.append(row)
            if selection_sse < best_sse:
                best_sse = selection_sse
                best_fit = replace(
                    fit,
                    codebook=fit.codebook.clone(),
                    state_to_palette=fit.state_to_palette.clone(),
                    palettes=fit.palettes.clone(),
                )
                best_row = dict(row)
            paths = fitting_encoded.transition_indices
            previous = fit
        candidates.append({"ridge_pseudocount": ridge, "history": history})
        ridge_best = min(history, key=lambda item: float(item["selection_sse"]))
        print(
            f"palette C{classes} ridge={ridge:g}: "
            f"best outer={ridge_best['outer_iteration']} "
            f"selection_sse={ridge_best['selection_sse']:.6g}",
            flush=True,
        )
    assert best_fit is not None and best_row is not None
    return best_fit, {
        "oracle_fitting_sse": oracle_fitting_sse,
        "oracle_fitting_seconds": oracle_seconds,
        "selected": best_row,
        "candidates": candidates,
    }


def _validate_oracle_signature(
    oracle: dict[str, object], args: argparse.Namespace
) -> None:
    signature = oracle["signature"]
    required = {
        "layers": list(args.layers),
        "experts": list(args.experts),
        "bits": args.bits,
        "window_bits": args.window_bits,
        "gain_tiles_per_matrix": args.gain_tiles,
        "fitting_tiles_per_matrix": args.fitting_tiles,
        "selection_tiles_per_matrix": args.selection_tiles,
        "evaluation_tiles_per_matrix": args.evaluation_tiles,
        "seed": args.seed,
    }
    mismatches = {
        key: {"oracle": signature.get(key), "requested": value}
        for key, value in required.items()
        if signature.get(key) != value
    }
    if mismatches:
        raise ValueError(f"oracle experiment signature mismatch: {mismatches}")


def _validate_groups(
    corpus: TileCorpus, gain_rows: list[dict[str, object]]
) -> None:
    if len(gain_rows) != len(corpus.groups):
        raise ValueError("oracle gain count does not match the reconstructed corpus")
    for group, gain in zip(corpus.groups, gain_rows, strict=True):
        identity = ("group", "layer", "expert", "matrix")
        if any(group[key] != gain[key] for key in identity):
            raise ValueError(f"oracle/corpus group mismatch: {group} versus {gain}")


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the state-palette experiment requires a CUDA device")
    if args.bits != 3 or args.window_bits != 16:
        raise ValueError("this controlled study requires K3/L16")
    graph = geometry(args.bits, args.window_bits)
    if any(classes < 2 or classes > graph.states for classes in args.classes):
        raise ValueError(f"classes must be in 2..{graph.states}")
    if (
        args.branch_groups < 1
        or graph.branches % args.branch_groups
        or args.branch_groups not in (1, 2, 4, 8)
    ):
        raise ValueError("branch_groups must be 1, 2, 4, or 8")
    if args.outer_iterations < 1 or args.palette_iterations < 1:
        raise ValueError("iteration counts must be positive")
    if any(layer not in C.MOE_LAYERS for layer in args.layers):
        raise ValueError("all layers must be routed MoE layers")
    if any(expert < 0 or expert >= C.NUM_EXPERTS for expert in args.experts):
        raise ValueError("experts must be in 0..895")

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

    gain_rows = oracle["gain_fit"]["selected_gains"]
    _validate_groups(corpus, gain_rows)
    gains = torch.tensor(
        [float(row["gain"]) for row in gain_rows],
        dtype=corpus.training.dtype,
        device=device,
    )
    oracle_tensors = load_file(str(args.oracle_codebooks), device=str(device))
    required_tensors = {"procedural_mcg_fp16", "learned_e4m3"}
    missing = required_tensors - set(oracle_tensors)
    if missing:
        raise ValueError(f"oracle codebook file is missing {sorted(missing)}")
    mcg = oracle_tensors["procedural_mcg_fp16"].float()
    full_e4m3 = oracle_tensors["learned_e4m3"].float()
    if mcg.numel() != graph.transitions or full_e4m3.numel() != graph.transitions:
        raise ValueError("oracle codebooks do not match K3/L16 geometry")

    selected_fits: dict[int, PaletteFit] = {}
    fits: dict[str, object] = {}
    for classes in args.classes:
        name = _candidate_name(classes, args.branch_groups)
        print(
            f"fitting shared L16 E4M3 state palette "
            f"G{args.branch_groups}/C{classes}",
            flush=True,
        )
        selected, fit_report = _fit_candidate(
            fitting,
            fitting_group,
            selection,
            selection_group,
            gains=gains,
            full_e4m3=full_e4m3,
            classes=classes,
            ridges=args.ridges,
            outer_iterations=args.outer_iterations,
            palette_iterations=args.palette_iterations,
            bits=args.bits,
            window_bits=args.window_bits,
            max_trace_mib=args.max_trace_mib,
            branch_groups=args.branch_groups,
        )
        selected_fits[classes] = selected
        fits[name] = fit_report

    codebooks = {
        "procedural_mcg_fp16": mcg,
        "full_learned_e4m3": full_e4m3,
        **{
            _candidate_name(classes, args.branch_groups): fit.codebook
            for classes, fit in selected_fits.items()
        },
    }
    metrics: dict[str, object] = {}
    for name, codebook in codebooks.items():
        encoded, _, seconds = _score(
            corpus.evaluation,
            corpus.evaluation_group,
            gains=gains,
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
        if name.startswith(("state_palette_", "state_subpalette_"))
    }
    storage = {}
    for classes, fit in selected_fits.items():
        size_function = (
            logical_state_palette_bytes
            if args.branch_groups == 1
            else logical_state_subpalette_bytes
        )
        size_args = {
            "states": graph.states,
            "branches": graph.branches,
            "classes": classes,
        }
        if args.branch_groups != 1:
            size_args["branch_groups"] = args.branch_groups
        packed = size_function(**size_args, packed_classes=True)
        integer_map = size_function(**size_args, packed_classes=False)
        name = _candidate_name(classes, args.branch_groups)
        storage[name] = {
            "packed_class_map_bytes": packed,
            "integer_class_map_bytes": integer_map,
            "fraction_of_full_e4m3_table_packed": packed / graph.transitions,
            "fraction_of_full_e4m3_table_integer_map": integer_map / graph.transitions,
            "state_map_entries": graph.states,
            "branch_groups": args.branch_groups,
            "palette_rows": classes,
            "palette_row_bytes": graph.branches // args.branch_groups,
            "total_palette_bytes": classes * graph.branches,
            "class_id_bits": max(1, math.ceil(math.log2(classes))),
        }

    payload: dict[str, object] = {
        "kind": "kquant_trellis_e4m3_state_palette_experiment",
        "schema_version": 1,
        "signature": {
            **oracle["signature"],
            "classes": list(args.classes),
            "branch_groups": args.branch_groups,
            "ridges": list(args.ridges),
            "outer_iterations": args.outer_iterations,
            "palette_iterations": args.palette_iterations,
            "oracle_json": str(args.oracle_json),
            "oracle_codebooks": str(args.oracle_codebooks),
            "factorization": (
                f"{args.branch_groups} independent contiguous branch-group state maps "
                "and E4M3 palette banks shared across all sampled layers, experts, and "
                "matrices; full K3/L16 state graph retained"
            ),
        },
        "corpus": {
            "groups": list(corpus.groups),
            "evaluation_coefficients": int(corpus.evaluation.numel()),
        },
        "fits": fits,
        "metrics": metrics,
        "comparisons_to_procedural_mcg": comparisons_to_mcg,
        "comparisons_to_full_learned_e4m3": comparisons_to_full,
        "storage": {
            "full_l16_e4m3_table_bytes": graph.transitions,
            "state_palettes": storage,
            "note": (
                "Logical sizes exclude alignment and optional table identifiers. The "
                "expanded codebooks saved for analysis are not part of the proposed format."
            ),
        },
    }
    tensors: dict[str, torch.Tensor] = {
        "procedural_mcg_fp16": mcg,
        "full_learned_e4m3": full_e4m3,
    }
    for classes, fit in selected_fits.items():
        prefix = _candidate_name(classes, args.branch_groups)
        tensors[f"{prefix}.expanded"] = fit.codebook
        map_dtype = torch.uint8 if classes <= 256 else torch.int16
        tensors[f"{prefix}.state_to_palette"] = fit.state_to_palette.to(map_dtype)
        tensors[f"{prefix}.palettes"] = fit.palettes
    _save_tensors(args.tensors, tensors)
    _atomic_json(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=_parse_ints, default=(1, 24, 40))
    parser.add_argument("--experts", type=_parse_ints, default=(0, 179, 358, 537, 716, 895))
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--window-bits", type=int, default=16)
    parser.add_argument("--classes", type=_parse_ints, default=(16, 32, 64, 128, 256))
    parser.add_argument("--branch-groups", type=int, default=1)
    parser.add_argument("--gain-tiles", type=int, default=8)
    parser.add_argument("--fitting-tiles", type=int, default=256)
    parser.add_argument("--selection-tiles", type=int, default=64)
    parser.add_argument("--evaluation-tiles", type=int, default=128)
    parser.add_argument("--ridges", type=_parse_floats, default=(0.25, 1.0, 4.0))
    parser.add_argument("--outer-iterations", type=int, default=4)
    parser.add_argument("--palette-iterations", type=int, default=24)
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
        default=Path("out/trellis-learned-codebooks-l1-l24-l40-shared-l16-v2.safetensors"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/trellis-e4m3-state-palettes-l1-l24-l40-v1.json"),
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=Path("out/trellis-e4m3-state-palettes-l1-l24-l40-v1.safetensors"),
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
