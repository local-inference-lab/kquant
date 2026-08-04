from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import kquant.pack.qsrt_pool as raw_keep_allocation
from kquant import constants as C
from kquant.qsrt import (
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    MATRIX_TRELLIS_BYTES,
    PHASE1_MODE_IDS,
    SCHEMA,
    TP12LayerLayout,
)
from kquant.pack.qsrt_pool import (
    CERTIFIED_ALLOCATION_OPTIMALITY,
    CANDIDATE_POOL_COMPLETION_FILENAME,
    RAW_KEEP_PROMOTION_BYTES,
    UNCERTIFIED_ALLOCATION_OPTIMALITY,
    QSRTCandidatePool,
    _expected_payload_metadata,
    _validate_payload_component,
    raw_keep_allocation_optimality,
    allocation_document,
    candidate_pool_completion_document,
    choose_qsrt_raw_keep_allocation,
    keep_mask_from_allocation_document,
    raw_keep_container_bytes,
    validate_candidate_pool_completion,
    validate_layer_metrics,
    validate_selection_ledger_evidence,
    write_candidate_pool_completion,
)
from kquant.pack.qsrt_candidates import (
    CANDIDATE_POOL_SCHEMA_VERSION,
    OFFICIAL_SOURCE_DAMAGE_METRIC,
)


def _metrics() -> dict[str, torch.Tensor]:
    experts = 3
    modes = (0, 1)
    fit_documents = 2
    confirmation_documents = 1
    fit_sse = torch.tensor(
        [
            [
                [[1.0, 2.0], [0.5, 1.0]],
                [[2.0, 2.0], [1.5, 1.5]],
            ],
            [
                [[3.0, 4.0], [4.0, 4.0]],
                [[5.0, 5.0], [6.0, 6.0]],
            ],
            [
                [[5.0, 6.0], [5.0, 5.0]],
                [[4.5, 5.0], [4.0, 5.0]],
            ],
        ],
        dtype=torch.float64,
    )
    confirmation_sse = torch.tensor(
        [
            [[[2.0], [1.0]], [[1.5], [1.25]]],
            [[[7.0], [8.0]], [[9.0], [10.0]]],
            [[[8.0], [7.0]], [[6.5], [6.0]]],
        ],
        dtype=torch.float64,
    )
    selected_r13 = torch.tensor([0, 0, 1], dtype=torch.uint8)
    selected_r2 = torch.tensor([1, 0, 1], dtype=torch.uint8)
    rows = torch.arange(experts)
    damage = (
        fit_sse[rows, selected_r13.long(), selected_r2.long()].sum(dim=1)
        + confirmation_sse[
            rows, selected_r13.long(), selected_r2.long()
        ].sum(dim=1)
    )
    return {
        "expert_ids": torch.arange(experts, dtype=torch.int16),
        "mode_ids": torch.tensor(modes, dtype=torch.uint8),
        "selected_r13": selected_r13,
        "selected_r2": selected_r2,
        "proposed_r13": selected_r13.clone(),
        "proposed_r2": selected_r2.clone(),
        "format_evaluated": torch.ones((experts, 2, 2), dtype=torch.bool),
        "fit_sse": fit_sse,
        "confirmation_sse": confirmation_sse,
        "fit_reference_energy": torch.ones(
            (experts, fit_documents), dtype=torch.float64
        ),
        "confirmation_reference_energy": torch.ones(
            (experts, confirmation_documents), dtype=torch.float64
        ),
        "fit_counts": torch.ones((experts, fit_documents), dtype=torch.int32),
        "confirmation_counts": torch.ones(
            (experts, confirmation_documents), dtype=torch.int32
        ),
        OFFICIAL_SOURCE_DAMAGE_METRIC: damage,
    }


def test_payload_metadata_binds_tailbite_context_when_declared() -> None:
    assert _expected_payload_metadata(
        7,
        codebook="sqg-normal-e4m3",
        tailbite_context=128,
    ) == {
        "kind": "kquant_kimi_k3_qsrt_candidate_pool",
        "schema_version": str(CANDIDATE_POOL_SCHEMA_VERSION),
        "layer": "7",
        "codebook": "sqg-normal-e4m3",
        "tailbite_context": "128",
    }


def test_layer_metrics_recompute_selected_official_source_damage() -> None:
    validated = validate_layer_metrics(
        _metrics(),
        mode_ids=(0, 1),
        fit_documents=2,
        confirmation_documents=1,
        expert_ids=(0, 1, 2),
    )

    torch.testing.assert_close(
        validated["damage"], torch.tensor([2.5, 14.0, 15.0], dtype=torch.float64)
    )
    assert validated["evaluated_format_count"].tolist() == [4, 4, 4]


def test_layer_metrics_reject_damage_that_does_not_close() -> None:
    metrics = _metrics()
    metrics[OFFICIAL_SOURCE_DAMAGE_METRIC][0] += 1
    with pytest.raises(ValueError, match=r"selected fit\+confirmation SSE"):
        validate_layer_metrics(
            metrics,
            mode_ids=(0, 1),
            fit_documents=2,
            confirmation_documents=1,
            expert_ids=(0, 1, 2),
        )


def test_layer_metrics_rederive_confirmation_proposal_and_gate() -> None:
    metrics = _metrics()
    metrics["confirmation_improvement"] = torch.tensor(
        [0.5, float("nan"), 0.25], dtype=torch.float64
    )
    metrics["confirmation_ci95"] = torch.tensor(
        [[0.5, 0.5], [float("nan"), float("nan")], [0.25, 0.25]],
        dtype=torch.float64,
    )

    validate_layer_metrics(
        metrics,
        mode_ids=(0, 1),
        fit_documents=2,
        confirmation_documents=1,
        expert_ids=(0, 1, 2),
        min_fit_documents=1,
        min_confirmation_documents=1,
        minimum_improvement=0.0,
    )

    metrics["selected_r2"][0] = 0
    with pytest.raises(ValueError, match="confirmation gate"):
        validate_layer_metrics(
            metrics,
            mode_ids=(0, 1),
            fit_documents=2,
            confirmation_documents=1,
            expert_ids=(0, 1, 2),
            min_fit_documents=1,
            min_confirmation_documents=1,
            minimum_improvement=0.0,
        )


def _format_coding(r13: int, r2: int) -> dict:
    rates = {"w1": ("n", r13), "w3": ("n", r13), "w2": ("k", r2)}
    trellis_bytes = 3 * MATRIX_TRELLIS_BYTES
    scale_bytes = 6 * (INTERMEDIATE_CHANNELS + LATENT_CHANNELS)
    return {
        "trellis_bytes": trellis_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes_before_layer_deduplication": trellis_bytes + scale_bytes,
        "matrices": {
            matrix: {
                "proxy": 0.0,
                "mode": f"R{rate}",
                "mode_id": rate,
                "rate_axis": rate_axis,
            }
            for matrix, (rate_axis, rate) in rates.items()
        }
    }


def _selection_ledger(
    metrics: dict[str, torch.Tensor],
) -> dict:
    selections = {}
    mode_ids = metrics["mode_ids"].tolist()
    for row, expert in enumerate(metrics["expert_ids"].tolist()):
        evaluated_formats = [
            (r13, r2)
            for r13_column, r13 in enumerate(mode_ids)
            for r2_column, r2 in enumerate(mode_ids)
            if bool(metrics["format_evaluated"][row, r13_column, r2_column])
        ]
        improvement = float(metrics["confirmation_improvement"][row])
        ci = [float(value) for value in metrics["confirmation_ci95"][row]]
        selections[str(expert)] = {
            "selection": {
                "selected_r13": int(metrics["selected_r13"][row]),
                "selected_r2": int(metrics["selected_r2"][row]),
                "proposed_r13": int(metrics["proposed_r13"][row]),
                "proposed_r2": int(metrics["proposed_r2"][row]),
                "accepted": bool(
                    metrics["selected_r13"][row]
                    or metrics["selected_r2"][row]
                ),
                "reason": "test evidence",
                "fit_documents": int((metrics["fit_counts"][row] > 0).sum()),
                "confirmation_documents": int(
                    (metrics["confirmation_counts"][row] > 0).sum()
                ),
                "confirmation_relative_improvement": (
                    improvement if np.isfinite(improvement) else None
                ),
                "confirmation_ci95": [
                    value if np.isfinite(value) else None for value in ci
                ],
                "bootstrap_replicates_valid": 100,
            },
            "evaluated_modes": sorted(
                {rate for rate_pair in evaluated_formats for rate in rate_pair}
            ),
            "mode_coding": {
                f"R{r13}/R{r2}": _format_coding(r13, r2)
                for r13, r2 in evaluated_formats
            },
        }
    return {"selections": selections}


def test_selection_json_is_bound_to_tensor_evidence_and_descriptors() -> None:
    metrics = _metrics()
    metrics["confirmation_improvement"] = torch.tensor(
        [0.5, float("nan"), 0.25], dtype=torch.float64
    )
    metrics["confirmation_ci95"] = torch.tensor(
        [[0.5, 0.5], [float("nan"), float("nan")], [0.25, 0.25]],
        dtype=torch.float64,
    )
    ledger = _selection_ledger(metrics)

    validate_selection_ledger_evidence(
        ledger,
        metrics,
        mode_ids=(0, 1),
        expert_ids=(0, 1, 2),
    )

    ledger["selections"]["0"]["selection"]["selected_r2"] = 0
    with pytest.raises(ValueError, match="selected_r2 disagrees"):
        validate_selection_ledger_evidence(
            ledger,
            metrics,
            mode_ids=(0, 1),
            expert_ids=(0, 1, 2),
        )


def test_selection_json_preserves_qsrt_candidate_schema() -> None:
    metrics = _metrics()
    metrics["confirmation_improvement"] = torch.tensor(
        [0.5, float("nan"), 0.25], dtype=torch.float64
    )
    metrics["confirmation_ci95"] = torch.tensor(
        [[0.5, 0.5], [float("nan"), float("nan")], [0.25, 0.25]],
        dtype=torch.float64,
    )
    ledger = _selection_ledger(metrics)

    assert validate_selection_ledger_evidence(
        ledger,
        metrics,
        mode_ids=(0, 1),
        expert_ids=(0, 1, 2),
        logical_trellis_schema=SCHEMA,
    ) == SCHEMA


def test_selection_json_rejects_matrix_rate_drift() -> None:
    metrics = _metrics()
    metrics["confirmation_improvement"] = torch.tensor(
        [0.5, float("nan"), 0.25], dtype=torch.float64
    )
    metrics["confirmation_ci95"] = torch.tensor(
        [[0.5, 0.5], [float("nan"), float("nan")], [0.25, 0.25]],
        dtype=torch.float64,
    )
    ledger = _selection_ledger(metrics)
    matrix = ledger["selections"]["0"]["mode_coding"]["R0/R1"]["matrices"]["w2"]
    matrix["rate_axis"] = "n"

    with pytest.raises(ValueError, match="w2 rate_axis drifted"):
        validate_selection_ledger_evidence(
            ledger,
            metrics,
            mode_ids=(0, 1),
            expert_ids=(0, 1, 2),
        )


def test_candidate_payload_component_geometry_is_exact() -> None:
    _validate_payload_component(
        "w1.trellis",
        {"dtype": "I16", "shape": [2_064_384]},
        matrix="w1",
        part="trellis",
    )
    _validate_payload_component(
        "w1.suh",
        {"dtype": "F16", "shape": [3_584]},
        matrix="w1",
        part="suh",
    )
    _validate_payload_component(
        "w2.svh",
        {"dtype": "F16", "shape": [3_584]},
        matrix="w2",
        part="svh",
    )
    with pytest.raises(ValueError, match="expected F16"):
        _validate_payload_component(
            "w2.svh",
            {"dtype": "F32", "shape": [3_584]},
            matrix="w2",
            part="svh",
        )


def _damage() -> np.ndarray:
    return np.zeros((C.NUM_MOE_LAYERS, C.NUM_EXPERTS), dtype=np.float64)


def test_candidate_pool_completion_binds_every_atomic_triplet(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(raw_keep_allocation.C, "MOE_LAYERS", (1,))
    monkeypatch.setattr(raw_keep_allocation.C, "NUM_MOE_LAYERS", 1)
    (tmp_path / "candidates").mkdir()
    (tmp_path / "qsrt-candidate-manifest.json").write_text(
        '{"codebook":"sqg-normal-e4m3"}'
    )
    stem = tmp_path / "candidates" / "qsrt-layer-00001"
    payload = stem.with_suffix(".safetensors")
    metrics = stem.with_suffix(".metrics.safetensors")
    selection = stem.with_suffix(".selection.json")
    payload.write_bytes(b"p")
    metrics.write_bytes(b"m")
    selection.write_bytes(b"s")

    document = candidate_pool_completion_document(tmp_path, jobs=2)
    assert document["logical_trellis_schema"] == SCHEMA
    write_candidate_pool_completion(tmp_path, document)

    validated = validate_candidate_pool_completion(
        tmp_path,
        verify_hashes=True,
    )
    assert validated["content_sha256"] == document["content_sha256"]
    assert (tmp_path / CANDIDATE_POOL_COMPLETION_FILENAME).is_file()

    payload.write_bytes(b"x")
    with pytest.raises(ValueError, match="identity drifted"):
        validate_candidate_pool_completion(tmp_path)


def test_count_allocation_keeps_exact_global_top_damage_with_stable_ties() -> None:
    damage = _damage()
    damage[0, 5] = 9
    damage[4, 8] = 8
    damage[2, 3] = 7

    allocation = choose_qsrt_raw_keep_allocation(damage, keep_count=3)

    assert np.argwhere(allocation.keep_mask).tolist() == [[0, 5], [2, 3], [4, 8]]
    assert allocation.retained_experts == 3
    assert allocation.container_bytes == raw_keep_container_bytes(allocation.keep_mask)
    assert raw_keep_allocation_optimality(allocation) == CERTIFIED_ALLOCATION_OPTIMALITY


def test_exact_byte_target_repairs_layer_alignment_at_minimum_local_loss() -> None:
    damage = _damage()
    damage[0, :3] = [10, 9, 7]
    damage[1, 0] = 8
    all_compressed = C.NUM_MOE_LAYERS * TP12LayerLayout(
        C.NUM_EXPERTS, 0
    ).disk_bytes
    # Three promotions fit only if all three land in one layer: its rank-scale
    # padding is 6,144 bytes, versus 55,296 bytes for a 2+1 distribution.
    target = all_compressed + 3 * RAW_KEEP_PROMOTION_BYTES + 6_144

    allocation = choose_qsrt_raw_keep_allocation(
        damage, target_container_bytes=target
    )

    assert allocation.retained_experts == 3
    assert allocation.keep_mask[0, :3].tolist() == [True, True, True]
    assert not allocation.keep_mask[1, 0]
    assert allocation.container_bytes == target
    assert allocation.alignment_repair_swaps == 1
    assert allocation.alignment_repair_damage_cost == 1.0
    assert (
        raw_keep_allocation_optimality(allocation)
        == UNCERTIFIED_ALLOCATION_OPTIMALITY
    )


def test_exact_byte_target_drops_one_when_alignment_cannot_be_repaired() -> None:
    damage = _damage()
    damage[0, 0] = 1
    all_compressed = C.NUM_MOE_LAYERS * TP12LayerLayout(
        C.NUM_EXPERTS, 0
    ).disk_bytes

    allocation = choose_qsrt_raw_keep_allocation(
        damage,
        target_container_bytes=all_compressed + RAW_KEEP_PROMOTION_BYTES,
    )

    assert allocation.retained_experts == 0
    assert allocation.container_bytes == all_compressed
    assert (
        raw_keep_allocation_optimality(allocation)
        == UNCERTIFIED_ALLOCATION_OPTIMALITY
    )


def test_allocation_document_carries_final_format_codes_and_damage_ledger() -> None:
    damage = _damage()
    damage[0, 7] = 4
    r13 = np.zeros_like(damage, dtype=np.uint8)
    r2 = np.zeros_like(damage, dtype=np.uint8)
    r13[0, 8] = 1
    r2[0, 8] = 2
    pool = QSRTCandidatePool(
        root=Path("/candidate").resolve(),
        manifest={
            "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
            "source_revision": "revision",
        },
        damage=damage,
        selected_r13=r13,
        selected_r2=r2,
        proposed_r13=r13.copy(),
        proposed_r2=r2.copy(),
        evaluated_format_counts=np.ones_like(r13),
        format_histogram={
            f"R{left}/R{right}": 0
            for left in PHASE1_MODE_IDS
            for right in PHASE1_MODE_IDS
        },
    )
    allocation = choose_qsrt_raw_keep_allocation(damage, keep_count=1)

    document = allocation_document(pool, allocation)

    assert document["layers"]["1"]["format_codes"][7] == 0xFF
    assert document["layers"]["1"]["format_codes"][8] == 0x12
    assert document["meta"]["retained_damage_avoided"] == 4.0
    assert document["meta"]["remaining_compressed_damage"] == 0.0
    assert (
        document["meta"]["raw_keep_allocation_optimality"]
        == CERTIFIED_ALLOCATION_OPTIMALITY
    )
    parsed = keep_mask_from_allocation_document(document)
    assert np.array_equal(parsed, allocation.keep_mask)


def test_allocation_document_identifies_external_damage_source() -> None:
    damage = _damage()
    r13 = np.zeros_like(damage, dtype=np.uint8)
    r2 = np.zeros_like(damage, dtype=np.uint8)
    pool = QSRTCandidatePool(
        root=Path("/candidate").resolve(),
        manifest={
            "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
            "source_revision": "revision",
        },
        damage=damage,
        selected_r13=r13,
        selected_r2=r2,
        proposed_r13=r13.copy(),
        proposed_r2=r2.copy(),
        evaluated_format_counts=np.ones_like(r13),
        format_histogram={
            f"R{left}/R{right}": 0
            for left in PHASE1_MODE_IDS
            for right in PHASE1_MODE_IDS
        },
        damage_metric="held_out_damage",
        damage_weighting="document-disjoint routes",
        damage_provenance={"validation_scores": "/scores"},
    )

    document = allocation_document(
        pool, choose_qsrt_raw_keep_allocation(damage, keep_count=0)
    )

    assert document["meta"]["damage_metric"] == "held_out_damage"
    assert document["meta"]["damage_weighting"] == "document-disjoint routes"
    assert document["meta"]["damage_provenance"] == {
        "validation_scores": "/scores"
    }
