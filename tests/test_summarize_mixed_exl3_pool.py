from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from kquant.mixed_exl3 import (
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    LEGACY_LOGICAL_SCHEMA,
    MATRIX_TRELLIS_BYTES,
    SCHEMA,
    TP12TrellisDescriptor,
)
from kquant.pack.mixed_candidates import (
    CANDIDATE_POOL_KIND,
    CANDIDATE_POOL_SCHEMA_VERSION,
    OFFICIAL_SOURCE_DAMAGE_METRIC,
)
from scripts.summarize_mixed_exl3_pool import (
    summarize_arrays,
    summarize_candidate_pool,
)


def _metrics() -> dict[str, torch.Tensor]:
    fit_sse = torch.tensor(
        [
            [
                [[10.0, 10.0], [8.0, 8.0]],
                [[9.0, 9.0], [7.0, 7.0]],
            ],
            [
                [[4.0, 6.0], [3.0, 4.0]],
                [[3.5, 4.5], [4.0, 4.0]],
            ],
            [
                [[2.0, 3.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        ],
        dtype=torch.float64,
    )
    confirmation_sse = torch.tensor(
        [
            [[[5.0], [4.0]], [[4.5], [4.2]]],
            [[[2.0], [1.5]], [[1.8], [1.7]]],
            [[[1.0], [0.0]], [[0.0], [0.0]]],
        ],
        dtype=torch.float64,
    )
    selected_r13 = torch.zeros(3, dtype=torch.uint8)
    selected_r2 = torch.tensor([1, 0, 0], dtype=torch.uint8)
    rows = torch.arange(3)
    damage = (
        fit_sse[rows, selected_r13.long(), selected_r2.long()].sum(dim=1)
        + confirmation_sse[
            rows, selected_r13.long(), selected_r2.long()
        ].sum(dim=1)
    )
    return {
        "expert_ids": torch.arange(3, dtype=torch.int16),
        "mode_ids": torch.tensor([0, 1], dtype=torch.uint8),
        "selected_r13": selected_r13,
        "selected_r2": selected_r2,
        "proposed_r13": torch.zeros(3, dtype=torch.uint8),
        "proposed_r2": torch.tensor([1, 1, 0], dtype=torch.uint8),
        "format_evaluated": torch.tensor(
            [
                [[True, True], [True, True]],
                [[True, True], [True, True]],
                [[True, False], [False, False]],
            ],
            dtype=torch.bool,
        ),
        "fit_sse": fit_sse,
        "confirmation_sse": confirmation_sse,
        "fit_reference_energy": torch.full((3, 2), 100.0, dtype=torch.float64),
        "confirmation_reference_energy": torch.full(
            (3, 1), 100.0, dtype=torch.float64
        ),
        "fit_counts": torch.tensor([[2, 1], [1, 1], [0, 1]], dtype=torch.int32),
        "confirmation_counts": torch.tensor([[1], [1], [0]], dtype=torch.int32),
        "confirmation_ci95": torch.tensor(
            [[0.1, 0.3], [-0.5, 0.1], [float("nan"), float("nan")]],
            dtype=torch.float64,
        ),
        "confirmation_improvement": torch.tensor(
            [0.2, 0.25, float("nan")], dtype=torch.float64
        ),
        OFFICIAL_SOURCE_DAMAGE_METRIC: damage,
        "compressed_damage": damage.clone(),
    }


def _arrays(metrics: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "selected_r13": metrics["selected_r13"].numpy(),
        "selected_r2": metrics["selected_r2"].numpy(),
        "proposed_r13": metrics["proposed_r13"].numpy(),
        "proposed_r2": metrics["proposed_r2"].numpy(),
        "evaluated_count": metrics["format_evaluated"]
        .sum(dim=(1, 2))
        .numpy(),
        "fit_r0": metrics["fit_sse"][:, 0, 0].sum(dim=1).numpy(),
        "fit_selected": torch.tensor([16.0, 10.0, 5.0]).numpy(),
        "confirmation_r0": metrics["confirmation_sse"][:, 0, 0]
        .sum(dim=1)
        .numpy(),
        "confirmation_selected": torch.tensor([4.0, 2.0, 1.0]).numpy(),
        "fit_reference": metrics["fit_reference_energy"].sum(dim=1).numpy(),
        "confirmation_reference": metrics["confirmation_reference_energy"]
        .sum(dim=1)
        .numpy(),
        "fit_support": (metrics["fit_counts"] > 0).sum(dim=1).numpy(),
        "confirmation_support": (metrics["confirmation_counts"] > 0)
        .sum(dim=1)
        .numpy(),
        "fit_support_sufficient": torch.tensor(
            [True, True, False]
        ).numpy(),
        "confirmation_support_sufficient": torch.tensor(
            [True, True, False]
        ).numpy(),
        "confirmation_ci_low": metrics["confirmation_ci95"][:, 0].numpy(),
        "confirmation_ci_high": metrics["confirmation_ci95"][:, 1].numpy(),
        "damage": metrics[OFFICIAL_SOURCE_DAMAGE_METRIC].numpy(),
    }


def _selections(
    metrics: dict[str, torch.Tensor],
    *,
    schema: str = SCHEMA,
) -> dict[str, object]:
    geometry = {
        "w1": ("n", 224, 192),
        "w3": ("n", 224, 192),
        "w2": ("k", 192, 224),
    }
    result: dict[str, object] = {}
    trellis_bytes = 3 * MATRIX_TRELLIS_BYTES
    scale_bytes = 6 * (INTERMEDIATE_CHANNELS + LATENT_CHANNELS)
    for row, expert in enumerate(metrics["expert_ids"].tolist()):
        mode_ids = metrics["mode_ids"].tolist()
        evaluated_formats = [
            (r13, r2)
            for r13_column, r13 in enumerate(mode_ids)
            for r2_column, r2 in enumerate(mode_ids)
            if bool(metrics["format_evaluated"][row, r13_column, r2_column])
        ]
        improvement = float(metrics["confirmation_improvement"][row])
        raw_ci = [float(value) for value in metrics["confirmation_ci95"][row]]
        result[str(expert)] = {
            "selection": {
                "selected_r13": int(metrics["selected_r13"][row]),
                "selected_r2": int(metrics["selected_r2"][row]),
                "proposed_r13": int(metrics["proposed_r13"][row]),
                "proposed_r2": int(metrics["proposed_r2"][row]),
                "accepted": bool(
                    metrics["selected_r13"][row] or metrics["selected_r2"][row]
                ),
                "reason": "test evidence",
                "fit_documents": int((metrics["fit_counts"][row] > 0).sum()),
                "confirmation_documents": int(
                    (metrics["confirmation_counts"][row] > 0).sum()
                ),
                "confirmation_relative_improvement": (
                    improvement if torch.isfinite(torch.tensor(improvement)) else None
                ),
                "confirmation_ci95": [
                    value if torch.isfinite(torch.tensor(value)) else None
                    for value in raw_ci
                ],
                "bootstrap_replicates_valid": 100,
            },
            "evaluated_modes": sorted(
                {rate for rates in evaluated_formats for rate in rates}
            ),
            "mode_coding": {
                f"R{r13}/R{r2}": {
                    "trellis_bytes": trellis_bytes,
                    "scale_bytes": scale_bytes,
                    "total_bytes_before_layer_deduplication": (
                        trellis_bytes + scale_bytes
                    ),
                    "matrices": {
                        matrix: {
                            "proxy": 0.0,
                            "mode": f"R{r2 if matrix == 'w2' else r13}",
                            "mode_id": r2 if matrix == "w2" else r13,
                            "rate_axis": rate_axis,
                            "trellis_descriptor": TP12TrellisDescriptor(
                                mode_id=r2 if matrix == "w2" else r13,
                                rate_axis=rate_axis,
                                k_tiles=k_tiles,
                                n_tiles=n_tiles,
                                schema=schema,
                            ).to_manifest(),
                        }
                        for matrix, (rate_axis, k_tiles, n_tiles) in geometry.items()
                    }
                }
                for r13, r2 in evaluated_formats
            },
        }
    return result


def test_summarize_arrays_reports_population_selection_evidence() -> None:
    summary = summarize_arrays([_arrays(_metrics())], (0, 1))

    assert summary["assignments"] == 3
    assert summary["selected_format_histogram"] == {
        "R0/R0": 2,
        "R0/R1": 1,
        "R1/R0": 0,
        "R1/R1": 0,
    }
    assert summary["proposed_format_histogram"] == {
        "R0/R0": 1,
        "R0/R1": 2,
        "R1/R0": 0,
        "R1/R1": 0,
    }
    assert summary["selection"]["accepted_nonzero_proposals"] == 1
    assert summary["selection"]["rejected_nonzero_proposals"] == 1
    assert summary["selection"]["r0_r0_only_evaluated"] == 1
    assert summary["confirmation_selection_evidence"][
        "relative_improvement"
    ] == pytest.approx(1.0 - 7.0 / 8.0)
    assert summary["selection"]["accepted_confirmation_regressions"] == 0
    assert summary["support"]["mode_evaluation_gate"] == {
        "sufficient_fit_and_confirmation": 2,
        "insufficient_fit_only": 0,
        "insufficient_confirmation_only": 0,
        "insufficient_both": 1,
    }


def test_running_pool_summary_ignores_partial_layers(tmp_path: Path) -> None:
    root = tmp_path / "pool"
    candidate_dir = root / "candidates"
    candidate_dir.mkdir(parents=True)
    manifest = {
        "kind": CANDIDATE_POOL_KIND,
        "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "mode_ids": [0, 1],
    }
    (root / "mixed_exl3_candidate_manifest.json").write_text(json.dumps(manifest))

    stem = "mixed-exl3-layer-00001"
    (candidate_dir / f"{stem}.safetensors").write_bytes(b"payload")
    save_file(_metrics(), candidate_dir / f"{stem}.metrics.safetensors")
    ledger = {
        "kind": CANDIDATE_POOL_KIND,
        "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "complete": True,
        "layer": 1,
        "experts": [0, 1, 2],
        "selection_contract": {
            "mode_ids": [0, 1],
            "fit_documents": 2,
            "confirmation_documents": 1,
            "min_fit_documents": 2,
            "min_confirmation_documents": 1,
            "minimum_improvement": 0.0,
        },
        "logical_trellis_schema": LEGACY_LOGICAL_SCHEMA,
        "selected_format_histogram": {
            "R0/R0": 2,
            "R0/R1": 1,
            "R1/R0": 0,
            "R1/R1": 0,
        },
        "selections": _selections(
            _metrics(), schema=LEGACY_LOGICAL_SCHEMA
        ),
    }
    (candidate_dir / f"{stem}.selection.json").write_text(json.dumps(ledger))
    (candidate_dir / ".mixed-exl3-layer-00002.safetensors.partial").write_bytes(
        b"partial"
    )

    summary = summarize_candidate_pool(root)

    assert summary["sealed"] is False
    assert summary["content_sha256"] is None
    assert summary["progress"]["completed_layers"] == [1]
    assert summary["progress"]["in_progress_layers"] == [2]
    assert len(summary["progress"]["pending_layers"]) == 90
    assert summary["logical_trellis_schema"] == LEGACY_LOGICAL_SCHEMA
    assert summary["aggregate"]["selected_format_histogram"] == {
        "R0/R0": 2,
        "R0/R1": 1,
        "R1/R0": 0,
        "R1/R1": 0,
    }


def test_running_fieldless_pool_reports_legacy_schema_before_first_layer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pool"
    candidate_dir = root / "candidates"
    candidate_dir.mkdir(parents=True)
    (root / "mixed_exl3_candidate_manifest.json").write_text(
        json.dumps(
            {
                "kind": CANDIDATE_POOL_KIND,
                "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
                "mode_ids": [0, 1],
            }
        )
    )
    (candidate_dir / ".mixed-exl3-layer-00001.safetensors.partial").write_bytes(
        b"partial"
    )

    summary = summarize_candidate_pool(root)

    assert summary["logical_trellis_schema"] == LEGACY_LOGICAL_SCHEMA
    assert summary["progress"]["completed_layers"] == []
    assert summary["progress"]["in_progress_layers"] == [1]
    assert summary["aggregate"] is None


def test_running_pool_summary_rejects_declared_schema_drift(tmp_path: Path) -> None:
    root = tmp_path / "pool"
    candidate_dir = root / "candidates"
    candidate_dir.mkdir(parents=True)
    (root / "mixed_exl3_candidate_manifest.json").write_text(
        json.dumps(
            {
                "kind": CANDIDATE_POOL_KIND,
                "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
                "mode_ids": [0, 1],
                "logical_trellis_schema": SCHEMA,
            }
        )
    )
    stem = "mixed-exl3-layer-00001"
    (candidate_dir / f"{stem}.safetensors").write_bytes(b"payload")
    save_file(_metrics(), candidate_dir / f"{stem}.metrics.safetensors")
    (candidate_dir / f"{stem}.selection.json").write_text(
        json.dumps(
            {
                "kind": CANDIDATE_POOL_KIND,
                "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
                "complete": True,
                "layer": 1,
                "experts": [0, 1, 2],
                "logical_trellis_schema": LEGACY_LOGICAL_SCHEMA,
                "selection_contract": {
                    "mode_ids": [0, 1],
                    "fit_documents": 2,
                    "confirmation_documents": 1,
                    "min_fit_documents": 2,
                    "min_confirmation_documents": 1,
                    "minimum_improvement": 0.0,
                },
                "selections": _selections(
                    _metrics(), schema=LEGACY_LOGICAL_SCHEMA
                ),
            }
        )
    )

    with pytest.raises(ValueError, match="logical trellis schema disagrees"):
        summarize_candidate_pool(root)
