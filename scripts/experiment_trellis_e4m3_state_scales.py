#!/usr/bin/env python3
"""Test a 16-bit state descriptor for an L16/K3 E4M3 codebook.

The fixed rolling trellis is unchanged.  Each of its 8,192 origin states
selects a learned eight-byte E4M3 palette row and a small signed power-of-two
exponent.  With 512 rows, a simple uint16 descriptor table plus the palette
bank occupies exactly 20 KiB.  The corresponding packed forms occupy 16 KiB
with three exponent bits and 17 KiB with four.

This driver starts from the selected flat C512 palette, alternates scaled
palette fitting with fresh Viterbi paths, selects on disjoint tiles, and
reports untouched evaluation metrics against procedural MCG and the full
64-KiB learned E4M3 table.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from kquant import constants as C
from kquant.source_weights import OfficialMXFP4Store
from kquant.state_palette import (
    StateScaledPaletteFit,
    expand_state_scaled_palette,
    fit_state_scaled_palette,
    logical_state_scaled_palette_bytes,
)
from kquant.trellis_window import geometry, trellis_alias_metrics
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
)


def _candidate_name(classes: int, exponent_bits: int) -> str:
    return f"state_scaled_palette_e4m3_c{classes}_e{exponent_bits}"


def _exponent_values(bits: int) -> tuple[int, ...]:
    if bits < 1 or bits > 8:
        raise ValueError("exponent bits must be in 1..8")
    half = 1 << (bits - 1)
    return tuple(range(-half, half))


def _exponent_statistics(fit: StateScaledPaletteFit) -> dict[str, object]:
    codes = fit.state_to_exponent.long()
    counts = torch.bincount(codes, minlength=fit.exponent_values.numel())
    used = counts > 0
    probabilities = counts[used].double()
    probabilities /= probabilities.sum()
    entropy = -(probabilities * probabilities.log2()).sum()
    selected = fit.exponent_values.index_select(0, codes).float()
    return {
        "values": [int(value) for value in fit.exponent_values.tolist()],
        "counts": [int(value) for value in counts.tolist()],
        "used_levels": int(torch.count_nonzero(used).item()),
        "entropy_bits": float(entropy.item()),
        "nonzero_fraction": float((selected != 0.0).float().mean().item()),
        "minimum": int(selected.min().item()),
        "maximum": int(selected.max().item()),
    }


def _fit_candidate(
    fitting: torch.Tensor,
    fitting_group: torch.Tensor,
    selection: torch.Tensor,
    selection_group: torch.Tensor,
    *,
    gains: torch.Tensor,
    full_e4m3: torch.Tensor,
    initial_state_to_palette: torch.Tensor,
    initial_palettes: torch.Tensor,
    classes: int,
    exponent_bits: int,
    ridges: tuple[float, ...],
    outer_iterations: int,
    palette_iterations: int,
    bits: int,
    window_bits: int,
    max_trace_mib: int,
) -> tuple[StateScaledPaletteFit, dict[str, object]]:
    exponent_values = _exponent_values(exponent_bits)
    zero_code = exponent_values.index(0)
    initial_exponent = torch.full_like(initial_state_to_palette, zero_code)
    initial_codebook = expand_state_scaled_palette(
        initial_state_to_palette,
        initial_exponent,
        initial_palettes,
        torch.tensor(
            exponent_values,
            dtype=torch.float32,
            device=initial_palettes.device,
        ),
        bits=bits,
        window_bits=window_bits,
    )
    initial_fit = StateScaledPaletteFit(
        codebook=initial_codebook,
        state_to_palette=initial_state_to_palette.long().clone(),
        state_to_exponent=initial_exponent,
        palettes=initial_palettes.float().clone(),
        exponent_values=torch.tensor(
            exponent_values,
            dtype=torch.float32,
            device=initial_palettes.device,
        ),
        objective=float("nan"),
        iterations=0,
    )
    fitting_gain = gains.index_select(0, fitting_group.long())
    fitting_targets = fitting * fitting_gain.unsqueeze(1)
    fitting_weights = fitting_gain.reciprocal().square().unsqueeze(1).expand_as(fitting)
    initial_encoded, initial_fitting_sse, initial_fitting_seconds = _score(
        fitting,
        fitting_group,
        gains=gains,
        codebook=initial_codebook,
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
        codebook=initial_codebook,
        bits=bits,
        window_bits=window_bits,
        max_trace_mib=max_trace_mib,
        return_transitions=False,
    )
    best_sse = initial_selection_sse
    best_fit = initial_fit
    best_row: dict[str, object] = {
        "source": "flat_c512_initialization",
        "ridge_pseudocount": None,
        "outer_iteration": -1,
        "palette_iterations": 0,
        "fitting_sse": initial_fitting_sse,
        "selection_sse": initial_selection_sse,
        "fitting_seconds": initial_fitting_seconds,
        "selection_seconds": initial_selection_seconds,
        "exponents": _exponent_statistics(initial_fit),
    }
    candidates = []
    for ridge in ridges:
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
            fit = fit_state_scaled_palette(
                transition_targets,
                transition_weights,
                bits=bits,
                window_bits=window_bits,
                classes=classes,
                exponent_values=exponent_values,
                reconstruction_dtype=torch.float8_e4m3fn,
                iterations=palette_iterations,
                initial_state_to_palette=previous.state_to_palette,
                initial_state_to_exponent=previous.state_to_exponent,
                initial_palettes=previous.palettes,
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
                "source": "scaled_refit",
                "ridge_pseudocount": ridge,
                "outer_iteration": outer_iteration,
                "palette_iterations": fit.iterations,
                "palette_objective": fit.objective,
                "fitting_sse": fitting_sse,
                "selection_sse": selection_sse,
                "fitting_seconds": fitting_seconds,
                "selection_seconds": selection_seconds,
                "exponents": _exponent_statistics(fit),
            }
            history.append(row)
            if selection_sse < best_sse:
                best_sse = selection_sse
                best_fit = replace(
                    fit,
                    codebook=fit.codebook.clone(),
                    state_to_palette=fit.state_to_palette.clone(),
                    state_to_exponent=fit.state_to_exponent.clone(),
                    palettes=fit.palettes.clone(),
                    exponent_values=fit.exponent_values.clone(),
                )
                best_row = dict(row)
            paths = fitting_encoded.transition_indices
            previous = fit
        candidates.append({"ridge_pseudocount": ridge, "history": history})
        ridge_best = min(history, key=lambda row: float(row["selection_sse"]))
        print(
            f"scaled C{classes}/E{exponent_bits} ridge={ridge:g}: "
            f"best outer={ridge_best['outer_iteration']} "
            f"selection_sse={ridge_best['selection_sse']:.8g}",
            flush=True,
        )
    return best_fit, {"selected": best_row, "candidates": candidates}


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the state-scale experiment requires CUDA")
    if args.bits != 3 or args.window_bits != 16:
        raise ValueError("this controlled study requires K3/L16")
    graph = geometry(args.bits, args.window_bits)
    if args.classes < 2 or args.classes > graph.states:
        raise ValueError(f"classes must be in 2..{graph.states}")
    if args.outer_iterations < 1 or args.palette_iterations < 1:
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
    gain_rows = oracle["gain_fit"]["selected_gains"]
    _validate_groups(corpus, gain_rows)
    gains = torch.tensor(
        [float(row["gain"]) for row in gain_rows],
        dtype=corpus.training.dtype,
        device=device,
    )
    oracle_tensors = load_file(str(args.oracle_codebooks), device=str(device))
    mcg = oracle_tensors["procedural_mcg_fp16"].float()
    full_e4m3 = oracle_tensors["learned_e4m3"].float()
    flat_tensors = load_file(str(args.flat_palettes), device=str(device))
    flat_prefix = f"state_palette_e4m3_c{args.classes}"
    initial_state_to_palette = flat_tensors[f"{flat_prefix}.state_to_palette"].long()
    initial_palettes = flat_tensors[f"{flat_prefix}.palettes"].float()

    selected_fits: dict[int, StateScaledPaletteFit] = {}
    fit_reports = {}
    for exponent_bits in args.exponent_bits:
        print(
            f"fitting shared L16/K3 E4M3 C{args.classes} "
            f"with E{exponent_bits} state scales",
            flush=True,
        )
        fit, report = _fit_candidate(
            fitting,
            fitting_group,
            selection,
            selection_group,
            gains=gains,
            full_e4m3=full_e4m3,
            initial_state_to_palette=initial_state_to_palette,
            initial_palettes=initial_palettes,
            classes=args.classes,
            exponent_bits=exponent_bits,
            ridges=args.ridges,
            outer_iterations=args.outer_iterations,
            palette_iterations=args.palette_iterations,
            bits=args.bits,
            window_bits=args.window_bits,
            max_trace_mib=args.max_trace_mib,
        )
        selected_fits[exponent_bits] = fit
        fit_reports[_candidate_name(args.classes, exponent_bits)] = report

    codebooks = {
        "procedural_mcg_fp16": mcg,
        "full_learned_e4m3": full_e4m3,
        "flat_state_palette_e4m3_c512": flat_tensors[
            f"{flat_prefix}.expanded"
        ].float(),
        **{
            _candidate_name(args.classes, exponent_bits): fit.codebook
            for exponent_bits, fit in selected_fits.items()
        },
    }
    metrics = {}
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
        if name.startswith("state_scaled_palette_")
    }
    storage = {}
    for exponent_bits in args.exponent_bits:
        name = _candidate_name(args.classes, exponent_bits)
        levels = 1 << exponent_bits
        storage[name] = {
            "uint16_descriptor_bytes": logical_state_scaled_palette_bytes(
                states=graph.states,
                branches=graph.branches,
                classes=args.classes,
                exponent_levels=levels,
                packed_descriptor=False,
            ),
            "packed_descriptor_bytes": logical_state_scaled_palette_bytes(
                states=graph.states,
                branches=graph.branches,
                classes=args.classes,
                exponent_levels=levels,
                packed_descriptor=True,
            ),
            "palette_bytes": args.classes * graph.branches,
            "state_descriptor_bits": math.ceil(math.log2(args.classes))
            + exponent_bits,
            "decoder_helper_table_bytes": 0,
        }

    payload: dict[str, object] = {
        "kind": "kquant_trellis_e4m3_state_scale_experiment",
        "schema_version": 1,
        "signature": {
            **oracle["signature"],
            "classes": args.classes,
            "exponent_bits": list(args.exponent_bits),
            "ridges": list(args.ridges),
            "outer_iterations": args.outer_iterations,
            "palette_iterations": args.palette_iterations,
            "oracle_json": str(args.oracle_json),
            "oracle_codebooks": str(args.oracle_codebooks),
            "flat_palettes": str(args.flat_palettes),
            "factorization": (
                "one global E4M3 palette bank; each L16 origin state stores a "
                "palette-row ID and signed integer power-of-two exponent; the "
                "K3 graph, branch labels, matrix gains, and corpus remain fixed"
            ),
        },
        "corpus": {
            "groups": list(corpus.groups),
            "evaluation_coefficients": int(corpus.evaluation.numel()),
        },
        "fits": fit_reports,
        "metrics": metrics,
        "comparisons_to_procedural_mcg": comparisons_to_mcg,
        "comparisons_to_full_learned_e4m3": comparisons_to_full,
        "storage": {
            "full_l16_e4m3_table_bytes": graph.transitions,
            "candidates": storage,
            "note": (
                "Counts are global codebook bytes, not bpw payload. Integer exponent "
                "adjustment is assumed; any eventual helper LUT must be charged if the "
                "kernel cannot perform it procedurally."
            ),
        },
    }
    tensors = {
        "procedural_mcg_fp16": mcg,
        "full_learned_e4m3": full_e4m3,
    }
    for exponent_bits, fit in selected_fits.items():
        prefix = _candidate_name(args.classes, exponent_bits)
        tensors[f"{prefix}.expanded"] = fit.codebook
        tensors[f"{prefix}.state_to_palette"] = fit.state_to_palette.to(torch.uint16)
        tensors[f"{prefix}.state_to_exponent"] = fit.state_to_exponent.to(torch.uint8)
        tensors[f"{prefix}.palettes"] = fit.palettes
        tensors[f"{prefix}.exponent_values"] = fit.exponent_values
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
    parser.add_argument("--classes", type=int, default=512)
    parser.add_argument("--exponent-bits", type=_parse_ints, default=(3, 4))
    parser.add_argument("--gain-tiles", type=int, default=8)
    parser.add_argument("--fitting-tiles", type=int, default=256)
    parser.add_argument("--selection-tiles", type=int, default=64)
    parser.add_argument("--evaluation-tiles", type=int, default=128)
    parser.add_argument("--ridges", type=_parse_floats, default=(0.25, 1.0, 4.0))
    parser.add_argument("--outer-iterations", type=int, default=2)
    parser.add_argument("--palette-iterations", type=int, default=12)
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
        "--flat-palettes",
        type=Path,
        default=Path(
            "out/trellis-e4m3-state-palettes-frontier-c512-c1024-v1.safetensors"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/trellis-e4m3-state-scales-c512-v1.json"),
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=Path("out/trellis-e4m3-state-scales-c512-v1.safetensors"),
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
