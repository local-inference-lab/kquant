#!/usr/bin/env python3
"""Measure useful specialization granularity for L16 E4M3 state palettes.

This reuses the exact corpus, matrix gains, and full learned E4M3 oracle from
the shared-table experiment.  It fits the same state-paletted reconstruction
family at several scopes while keeping the K3/L16 transition graph fixed:

* ``role``: one table for fused w1/w3 and one for w2;
* ``layer``: one table shared by all expert matrices in a layer;
* ``layer_role``: one fused-w1/w3 and one w2 table per layer.

No expert needs a table identifier for these scopes: layer and matrix role
already determine the reconstruction table used by the kernel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from safetensors.torch import load_file
import torch

from kquant import constants as C
from kquant.source_weights import OfficialMXFP4Store
from kquant.state_palette import StatePaletteFit, logical_state_palette_bytes
from kquant.trellis_window import geometry, trellis_alias_metrics
from scripts.experiment_trellis_e4m3_state_palettes import (
    _fit_candidate,
    _parse_floats,
    _save_tensors,
    _validate_groups,
    _validate_oracle_signature,
)
from scripts.experiment_trellis_learned_codebooks import _score, _slice_each_group
from scripts.experiment_trellis_window_fp8 import (
    TileCorpus,
    _atomic_json,
    _comparison,
    _paired_metrics,
    _parse_ints,
    collect_corpus,
)


VALID_SCOPES = ("role", "layer", "layer_role")


def _parse_scopes(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("scopes must be nonempty and unique")
    unknown = set(result) - set(VALID_SCOPES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown scopes: {sorted(unknown)}")
    return result


def _role(matrix: object) -> str:
    return "w2" if str(matrix) == "w2" else "w13"


def _unit_key(group: dict[str, object], scope: str) -> str:
    if scope == "role":
        return _role(group["matrix"])
    if scope == "layer":
        return f"layer_{int(group['layer']):03d}"
    if scope == "layer_role":
        return f"layer_{int(group['layer']):03d}_{_role(group['matrix'])}"
    raise ValueError(f"unsupported scope {scope}")


def _scope_units(
    groups: tuple[dict[str, object], ...], scope: str
) -> dict[str, tuple[int, ...]]:
    units: dict[str, list[int]] = {}
    for group in groups:
        units.setdefault(_unit_key(group, scope), []).append(int(group["group"]))
    return {key: tuple(value) for key, value in sorted(units.items())}


def _membership(group_ids: torch.Tensor, members: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(group_ids, dtype=torch.bool)
    for member in members:
        mask.logical_or_(group_ids == member)
    return mask


def _evaluate_scoped(
    corpus: TileCorpus,
    *,
    units: dict[str, tuple[int, ...]],
    fits: dict[str, StatePaletteFit],
    gains: torch.Tensor,
    bits: int,
    window_bits: int,
    max_trace_mib: int,
) -> dict[str, object]:
    reconstruction = torch.empty_like(corpus.evaluation)
    seconds = 0.0
    aliases = []
    covered = torch.zeros_like(corpus.evaluation_group, dtype=torch.bool)
    for unit, members in units.items():
        mask = _membership(corpus.evaluation_group, members)
        if bool(torch.any(covered & mask)):
            raise ValueError("scope units overlap")
        encoded, _, elapsed = _score(
            corpus.evaluation[mask],
            corpus.evaluation_group[mask],
            gains=gains,
            codebook=fits[unit].codebook,
            bits=bits,
            window_bits=window_bits,
            max_trace_mib=max_trace_mib,
            return_transitions=True,
        )
        assert encoded.transition_indices is not None
        reconstruction[mask] = encoded.reconstructed
        covered.logical_or_(mask)
        seconds += elapsed
        aliases.append(
            {
                "unit": unit,
                "matrix_groups": list(members),
                "metrics": trellis_alias_metrics(
                    fits[unit].codebook,
                    bits,
                    window_bits,
                    transition_indices=encoded.transition_indices,
                ),
            }
        )
    if not bool(torch.all(covered)):
        raise ValueError("scope units did not cover every evaluation tile")
    return {
        **_paired_metrics(
            corpus.evaluation,
            reconstruction,
            corpus.evaluation_group,
            corpus.groups,
        ),
        "seconds": seconds,
        "alias_structure_by_table": aliases,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the palette-scope experiment requires a CUDA device")
    if args.bits != 3 or args.window_bits != 16:
        raise ValueError("this controlled study requires K3/L16")
    graph = geometry(args.bits, args.window_bits)
    if any(classes < 2 or classes > graph.states for classes in args.classes):
        raise ValueError(f"classes must be in 2..{graph.states}")
    if args.outer_iterations < 1 or args.palette_iterations < 1:
        raise ValueError("iteration counts must be positive")

    oracle = json.loads(args.oracle_json.read_text())
    _validate_oracle_signature(oracle, args)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
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

    metrics: dict[str, object] = {}
    for name, codebook in (
        ("procedural_mcg_fp16", mcg),
        ("full_shared_learned_e4m3", full_e4m3),
    ):
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

    fit_reports: dict[str, object] = {}
    selected_fits: dict[str, dict[str, StatePaletteFit]] = {}
    scope_units: dict[str, dict[str, tuple[int, ...]]] = {}
    for classes in args.classes:
        global_candidate = f"state_palette_e4m3_c{classes}_global"
        global_units = {
            "global": tuple(int(group["group"]) for group in corpus.groups)
        }
        print(f"fitting C{classes} shared initialization", flush=True)
        global_fit, global_report = _fit_candidate(
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
        )
        scope_units[global_candidate] = global_units
        selected_fits[global_candidate] = {"global": global_fit}
        fit_reports[global_candidate] = {
            "scope": "global",
            "units": {
                "global": {
                    "matrix_groups": list(global_units["global"]),
                    **global_report,
                }
            },
        }
        metrics[global_candidate] = _evaluate_scoped(
            corpus,
            units=global_units,
            fits=selected_fits[global_candidate],
            gains=gains,
            bits=args.bits,
            window_bits=args.window_bits,
            max_trace_mib=args.max_trace_mib,
        )
        for scope in args.scopes:
            candidate = f"state_palette_e4m3_c{classes}_{scope}"
            units = _scope_units(corpus.groups, scope)
            scope_units[candidate] = units
            selected_fits[candidate] = {}
            unit_reports = {}
            for unit, members in units.items():
                fitting_mask = _membership(fitting_group, members)
                selection_mask = _membership(selection_group, members)
                print(
                    f"fitting C{classes} scope={scope} unit={unit} "
                    f"groups={len(members)}",
                    flush=True,
                )
                fit, report = _fit_candidate(
                    fitting[fitting_mask],
                    fitting_group[fitting_mask],
                    selection[selection_mask],
                    selection_group[selection_mask],
                    gains=gains,
                    full_e4m3=full_e4m3,
                    classes=classes,
                    ridges=args.ridges,
                    outer_iterations=args.outer_iterations,
                    palette_iterations=args.palette_iterations,
                    bits=args.bits,
                    window_bits=args.window_bits,
                    max_trace_mib=args.max_trace_mib,
                    initial_fit=global_fit,
                )
                selected_fits[candidate][unit] = fit
                unit_reports[unit] = {
                    "matrix_groups": list(members),
                    **report,
                }
            fit_reports[candidate] = {"scope": scope, "units": unit_reports}
            metrics[candidate] = _evaluate_scoped(
                corpus,
                units=units,
                fits=selected_fits[candidate],
                gains=gains,
                bits=args.bits,
                window_bits=args.window_bits,
                max_trace_mib=args.max_trace_mib,
            )

    comparisons_to_mcg = {
        name: _comparison(metrics["procedural_mcg_fp16"], result)
        for name, result in metrics.items()
        if name != "procedural_mcg_fp16"
    }
    comparisons_to_full = {
        name: _comparison(metrics["full_shared_learned_e4m3"], result)
        for name, result in metrics.items()
        if name.startswith("state_palette_")
    }

    production_tables = {
        "global": 1,
        "role": 2,
        "layer": len(C.MOE_LAYERS),
        "layer_role": 2 * len(C.MOE_LAYERS),
    }
    resident_tables = {"global": 1, "role": 2, "layer": 1, "layer_role": 2}
    storage: dict[str, object] = {}
    for candidate, units in scope_units.items():
        classes = int(candidate.split("_c", 1)[1].split("_", 1)[0])
        scope = fit_reports[candidate]["scope"]
        packed_per_table = logical_state_palette_bytes(
            states=graph.states,
            branches=graph.branches,
            classes=classes,
            packed_classes=True,
        )
        integer_per_table = logical_state_palette_bytes(
            states=graph.states,
            branches=graph.branches,
            classes=classes,
            packed_classes=False,
        )
        storage[candidate] = {
            "scope": scope,
            "tables_in_study": len(units),
            "tables_in_full_model": production_tables[scope],
            "simultaneously_resident_tables_per_layer": resident_tables[scope],
            "packed_bytes_per_table": packed_per_table,
            "integer_map_bytes_per_table": integer_per_table,
            "packed_bytes_full_model": packed_per_table * production_tables[scope],
            "integer_map_bytes_full_model": integer_per_table * production_tables[scope],
            "packed_resident_bytes_per_layer": packed_per_table * resident_tables[scope],
            "integer_map_resident_bytes_per_layer": integer_per_table * resident_tables[scope],
            "class_id_bits": max(1, math.ceil(math.log2(classes))),
        }

    payload: dict[str, object] = {
        "kind": "kquant_trellis_e4m3_palette_scope_experiment",
        "schema_version": 1,
        "signature": {
            **oracle["signature"],
            "classes": list(args.classes),
            "scopes": list(args.scopes),
            "ridges": list(args.ridges),
            "outer_iterations": args.outer_iterations,
            "palette_iterations": args.palette_iterations,
            "oracle_json": str(args.oracle_json),
            "oracle_codebooks": str(args.oracle_codebooks),
        },
        "corpus": {
            "groups": list(corpus.groups),
            "evaluation_coefficients": int(corpus.evaluation.numel()),
        },
        "fits": fit_reports,
        "metrics": metrics,
        "comparisons_to_procedural_mcg": comparisons_to_mcg,
        "comparisons_to_full_shared_learned_e4m3": comparisons_to_full,
        "storage": storage,
    }
    tensors: dict[str, torch.Tensor] = {
        "procedural_mcg_fp16": mcg,
        "full_shared_learned_e4m3": full_e4m3,
    }
    for candidate, units in selected_fits.items():
        classes = int(candidate.split("_c", 1)[1].split("_", 1)[0])
        map_dtype = torch.uint8 if classes <= 256 else torch.int16
        for unit, fit in units.items():
            prefix = f"{candidate}.{unit}"
            tensors[f"{prefix}.expanded"] = fit.codebook
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
    parser.add_argument("--classes", type=_parse_ints, default=(256, 512))
    parser.add_argument("--scopes", type=_parse_scopes, default=("role", "layer", "layer_role"))
    parser.add_argument("--gain-tiles", type=int, default=8)
    parser.add_argument("--fitting-tiles", type=int, default=256)
    parser.add_argument("--selection-tiles", type=int, default=64)
    parser.add_argument("--evaluation-tiles", type=int, default=128)
    parser.add_argument("--ridges", type=_parse_floats, default=(0.25, 1.0, 4.0))
    parser.add_argument("--outer-iterations", type=int, default=2)
    parser.add_argument("--palette-iterations", type=int, default=16)
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
        default=Path("out/trellis-e4m3-palette-scopes-l1-l24-l40-v1.json"),
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=Path("out/trellis-e4m3-palette-scopes-l1-l24-l40-v1.safetensors"),
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
                "comparisons_to_full_shared_learned_e4m3": payload[
                    "comparisons_to_full_shared_learned_e4m3"
                ],
                "storage": payload["storage"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
