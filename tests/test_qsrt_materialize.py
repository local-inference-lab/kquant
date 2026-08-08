from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kquant import constants as C
from kquant.exl3_reference import CODEBOOK_QSRT_E4M3
from kquant.qsrt import PHASE1_MODE_IDS
from kquant.pack.qsrt_pool import QSRTCandidatePool
from kquant.pack.qsrt_allocation import (
    choose_qsrt_lagrangian,
    qsrt_allocation_document,
)
from kquant.pack.qsrt_materialize import (
    qsrt_materialization_build_document,
    validate_qsrt_materialization_allocation,
)
from kquant.pack.x4t_index import (
    X4T_COST_COMPLETION_FILENAME,
    X4T_COST_MANIFEST_FILENAME,
    X4TCostIndex,
)


def _fixtures(tmp_path: Path) -> tuple[QSRTCandidatePool, X4TCostIndex]:
    shape = (C.NUM_MOE_LAYERS, C.NUM_EXPERTS)
    damage = np.zeros(shape, dtype=np.float64)
    damage[0, 7] = 10.0
    selected_r13 = np.zeros(shape, dtype=np.uint8)
    selected_r2 = np.zeros(shape, dtype=np.uint8)
    selected_r13[0, 8] = 1
    selected_r2[0, 8] = 2
    histogram = {
        f"R{r13}/R{r2}": 0
        for r13 in PHASE1_MODE_IDS
        for r2 in PHASE1_MODE_IDS
    }
    histogram["R0/R0"] = damage.size - 1
    histogram["R1/R2"] = 1
    pool = QSRTCandidatePool(
        root=(tmp_path / "pool").resolve(),
        manifest={"schema_version": 1, "source_revision": C.REVISION},
        damage=damage,
        selected_r13=selected_r13,
        selected_r2=selected_r2,
        proposed_r13=selected_r13.copy(),
        proposed_r2=selected_r2.copy(),
        evaluated_format_counts=np.ones(shape, dtype=np.uint8),
        format_histogram=histogram,
        completion={},
        content_sha256="ab" * 32,
        mode_ids=PHASE1_MODE_IDS,
        codebook=CODEBOOK_QSRT_E4M3,
    )
    matrix_costs = np.full((*shape, 3), 5_566_464, dtype=np.int64)
    expert_costs = matrix_costs.sum(axis=2)
    index = X4TCostIndex(
        root=(tmp_path / "x4t").resolve(),
        manifest={"source_revision": C.REVISION},
        matrix_storage_bytes=matrix_costs,
        expert_storage_bytes=expert_costs,
        content_sha256="cd" * 32,
    )
    return pool, index


def test_qsrt_materialization_closes_all_tiers_and_bytes(tmp_path) -> None:
    pool, index = _fixtures(tmp_path)
    allocation = choose_qsrt_lagrangian(
        pool.damage,
        index.expert_storage_bytes,
        lagrange_lambda=3e-7,
    )
    document = qsrt_allocation_document(pool, index, allocation)

    plan = validate_qsrt_materialization_allocation(document, pool, index)

    assert plan.x4t_experts == 1
    assert plan.layers[0].x4t == (7,)
    assert plan.layers[0].compressed[7] == 8
    assert plan.total_container_bytes == document["meta"]["container_bytes"]
    assert plan.total_container_bytes == (
        plan.qsrt_atom_bytes + plan.x4t_bytes
    )


def test_qsrt_materialization_rejects_byte_and_mode_drift(tmp_path) -> None:
    pool, index = _fixtures(tmp_path)
    allocation = choose_qsrt_lagrangian(
        pool.damage,
        index.expert_storage_bytes,
        lagrange_lambda=3e-7,
    )
    document = qsrt_allocation_document(pool, index, allocation)
    document["layers"]["1"]["x4t_layer_bytes"] += 4096
    with pytest.raises(ValueError, match="X4T bytes"):
        validate_qsrt_materialization_allocation(document, pool, index)

    document = qsrt_allocation_document(pool, index, allocation)
    document["layers"]["1"]["format_codes"][8] = 0
    with pytest.raises(ValueError, match="selected trellis candidate"):
        validate_qsrt_materialization_allocation(document, pool, index)


def test_qsrt_build_freezes_allocation_damage_provenance(tmp_path) -> None:
    pool, index = _fixtures(tmp_path)
    pool.root.mkdir()
    index.root.mkdir()
    for path in (
        pool.root / "qsrt-candidate-manifest.json",
        pool.root / "qsrt-candidate-completion.json",
        index.root / X4T_COST_MANIFEST_FILENAME,
        index.root / X4T_COST_COMPLETION_FILENAME,
    ):
        path.write_text("{}\n")
    allocation = choose_qsrt_lagrangian(
        pool.damage,
        index.expert_storage_bytes,
        lagrange_lambda=3e-7,
    )
    allocation_document = qsrt_allocation_document(pool, index, allocation)
    allocation_path = tmp_path / "allocation.json"
    allocation_path.write_text(json.dumps(allocation_document))
    plan = validate_qsrt_materialization_allocation(
        allocation_document,
        pool,
        index,
    )

    build = qsrt_materialization_build_document(
        pool=pool,
        x4t_index=index,
        allocation_path=allocation_path,
        official_root=tmp_path / "official",
        official_revision=C.REVISION,
        plan=plan,
    )

    assert build["allocation_damage_metric"] == pool.damage_metric
    assert build["allocation_damage_weighting"] == pool.damage_weighting
    assert build["allocation_damage_provenance"] == pool.damage_provenance
