from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import kquant.pack.mixed_materialize as mixed_materialize
from kquant import constants as C
from kquant.mixed_exl3 import (
    FORMAT_SECTION_BYTES,
    INTERMEDIATE_CHANNELS,
    KEEP_STORAGE_EXTERNAL_X4,
    LATENT_CHANNELS,
    LAYER_HEADER_BYTES,
    LEGACY_LOGICAL_SCHEMA,
    MATRIX_TRELLIS_BYTES,
    PAIR_BYTES,
    PHASE1_MODE_IDS,
    SCHEMA,
    ExpertFormatSpec,
    TP12LayerHeader,
    TP12LayerLayout,
    pack_tp12_format_section,
    pack_tp12_shared_scale_section,
    tp12_logical_pair_index,
)
from kquant.pack.mixed_allocation import (
    MixedCandidatePool,
    allocation_document,
    choose_mixed_allocation,
)
from kquant.pack.mixed_candidates import (
    CANDIDATE_POOL_SCHEMA_VERSION,
    candidate_tensor_name,
)
from kquant.pack.mixed_materialize import (
    LayerMaterializationSpec,
    MATERIALIZATION_BUILD_FILENAME,
    MATERIALIZED_ALLOCATION_FILENAME,
    TP12LayerReader,
    candidate_local_scale,
    candidate_trellis_pair,
    expected_logical_source_bytes,
    load_layer_closure_receipt,
    materialization_build_document,
    mxfp4_rank_matrix_components,
    prepare_destination,
    pwrite_exact,
    validate_materialization_allocation,
    validate_materialized_layer,
    validate_materialized_layer_payloads,
    write_layer_closure_receipt,
)
from kquant.pack.mixed_validation import (
    VALIDATION_DAMAGE_METRIC,
    VALIDATION_DAMAGE_WEIGHTING,
)
from kquant.source_weights import PackedMXFP4Matrix
from kquant.x4 import X4LayerWriter


def _pool(tmp_path: Path) -> MixedCandidatePool:
    shape = (C.NUM_MOE_LAYERS, C.NUM_EXPERTS)
    selected_r13 = np.zeros(shape, dtype=np.uint8)
    selected_r2 = np.zeros(shape, dtype=np.uint8)
    selected_r13[0, 8] = 1
    selected_r2[0, 8] = 2
    damage = np.zeros(shape, dtype=np.float64)
    damage[0, 7] = 5.0
    return MixedCandidatePool(
        root=tmp_path.resolve(),
        manifest={
            "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
            "source_revision": C.REVISION,
        },
        damage=damage,
        selected_r13=selected_r13,
        selected_r2=selected_r2,
        proposed_r13=selected_r13.copy(),
        proposed_r2=selected_r2.copy(),
        evaluated_format_counts=np.ones(shape, dtype=np.uint8),
        format_histogram={
            f"R{left}/R{right}": 0
            for left in PHASE1_MODE_IDS
            for right in PHASE1_MODE_IDS
        },
    )


def test_materialization_allocation_closes_modes_tiers_and_bytes(tmp_path) -> None:
    pool = _pool(tmp_path)
    allocation = choose_mixed_allocation(pool.damage, keep_count=1)
    document = allocation_document(pool, allocation)

    plan = validate_materialization_allocation(document, pool)

    layer = plan.layers[0]
    assert layer.kept == (7,)
    assert layer.formats[7].is_mxfp4
    assert layer.formats[8] == ExpertFormatSpec.compressed(1, 2)
    assert plan.retained_experts == 1
    assert plan.compressed_experts == C.NUM_MOE_LAYERS * C.NUM_EXPERTS - 1
    assert plan.total_container_bytes == document["meta"]["container_bytes"]


def test_materialization_build_records_legacy_source_but_emits_v3(
    tmp_path,
    monkeypatch,
) -> None:
    pool = replace(
        _pool(tmp_path),
        content_sha256="a" * 64,
        trellis_schema=LEGACY_LOGICAL_SCHEMA,
    )
    (tmp_path / "mixed_exl3_candidate_manifest.json").write_text("{}\n")
    document = allocation_document(
        pool,
        choose_mixed_allocation(pool.damage, keep_count=1),
    )
    allocation_path = tmp_path / "allocation.json"
    allocation_path.write_text(json.dumps(document))
    mode_summary_path = tmp_path / "mode-summary.json"
    mode_summary_path.write_text('{"pass": true}\n')
    teacher_proxy_summary_path = tmp_path / "teacher-proxy-summary.json"
    teacher_proxy_summary_path.write_text('{"status": "pass"}\n')
    performance_summary_path = tmp_path / "performance-summary.json"
    performance_summary_path.write_text('{"status": "pass"}\n')
    monkeypatch.setattr(
        mixed_materialize,
        "validate_materialization_mode_gate",
        lambda summary, allocation, candidate_pool: {"status": "pass"},
    )
    monkeypatch.setattr(
        mixed_materialize,
        "validate_materialization_teacher_proxy_gate",
        lambda summary: {"status": "pass"},
    )
    monkeypatch.setattr(
        mixed_materialize,
        "validate_materialization_performance_gate",
        lambda summary, candidate_pool: {"status": "pass"},
    )
    plan = validate_materialization_allocation(document, pool)

    build = materialization_build_document(
        pool=pool,
        allocation_path=allocation_path,
        mode_validation_summary_path=mode_summary_path,
        teacher_proxy_summary_path=teacher_proxy_summary_path,
        performance_summary_path=performance_summary_path,
        official_root=tmp_path,
        official_revision=C.REVISION,
        plan=plan,
    )

    assert build["candidate_logical_trellis_schema"] == LEGACY_LOGICAL_SCHEMA
    assert build["format_schema"] == SCHEMA


def test_materialization_allocation_rejects_candidate_mode_drift(tmp_path) -> None:
    pool = _pool(tmp_path)
    document = allocation_document(
        pool, choose_mixed_allocation(pool.damage, keep_count=1)
    )
    document["layers"]["1"]["format_codes"][8] = 0x11

    with pytest.raises(ValueError, match="selected phase-1 candidate"):
        validate_materialization_allocation(document, pool)


def test_materialization_reauthenticates_external_validation_damage(
    tmp_path,
    monkeypatch,
) -> None:
    pool = _pool(tmp_path / "candidate")
    damage = np.zeros_like(pool.damage)
    damage[0, 5] = 9.0
    score_root = (tmp_path / "scores").resolve()
    score_set_sha256 = "12" * 32
    validation = SimpleNamespace(
        root=score_root,
        manifest={
            "validation_capture": "/capture",
            "validation_report": "/report",
            "validation_documents": 27,
        },
        damage=damage,
        content_sha256=score_set_sha256,
    )
    provenance = {
        "validation_scores": str(score_root),
        "validation_score_set_sha256": score_set_sha256,
        "validation_capture": "/capture",
        "validation_report": "/report",
        "validation_documents": 27,
        "selection_data_used": False,
    }
    scored_pool = replace(
        pool,
        damage=damage,
        damage_metric=VALIDATION_DAMAGE_METRIC,
        damage_weighting=VALIDATION_DAMAGE_WEIGHTING,
        damage_provenance=provenance,
    )
    document = allocation_document(
        scored_pool,
        choose_mixed_allocation(damage, keep_count=1),
    )
    monkeypatch.setattr(
        mixed_materialize,
        "load_mixed_validation_scores",
        lambda root, candidate_pool: validation,
    )

    plan = validate_materialization_allocation(document, pool)

    assert plan.layers[0].kept == (5,)
    document["meta"]["damage_provenance"][
        "validation_score_set_sha256"
    ] = "34" * 32
    with pytest.raises(ValueError, match="provenance drifted"):
        validate_materialization_allocation(document, pool)


def test_materialization_rejects_damage_ledger_drift(tmp_path) -> None:
    pool = _pool(tmp_path)
    document = allocation_document(
        pool,
        choose_mixed_allocation(pool.damage, keep_count=1),
    )
    document["meta"]["retained_damage_avoided"] += 1.0

    with pytest.raises(ValueError, match="retained_damage_avoided"):
        validate_materialization_allocation(document, pool)


def test_materialization_mode_gate_requires_validation_backed_allocation(
    tmp_path,
) -> None:
    pool = _pool(tmp_path)
    document = allocation_document(
        pool,
        choose_mixed_allocation(pool.damage, keep_count=1),
    )
    summary = tmp_path / "mode-summary.json"
    summary.write_text("{}")

    with pytest.raises(ValueError, match="requires validation damage"):
        mixed_materialize.validate_materialization_mode_gate(
            summary,
            document,
            pool,
        )


def test_materialization_mode_gate_records_authenticated_pass(
    tmp_path,
    monkeypatch,
) -> None:
    pool = replace(_pool(tmp_path / "pool"), content_sha256="11" * 32)
    validation = SimpleNamespace(
        root=(tmp_path / "validation").resolve(),
        content_sha256="22" * 32,
    )
    scores = SimpleNamespace(
        root=(tmp_path / "mode-scores").resolve(),
        content_sha256="33" * 32,
        audited=np.array([[True, False]], dtype=np.bool_),
    )
    summary_document = {
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 20260801,
    }
    summary = tmp_path / "mode-summary.json"
    summary.write_text(json.dumps(summary_document))
    monkeypatch.setattr(
        mixed_materialize,
        "_allocation_validation_scores",
        lambda meta, candidate_pool: validation,
    )
    monkeypatch.setattr(
        mixed_materialize,
        "load_mixed_mode_validation_summary",
        lambda path, candidate_pool, selected, require_pass: (
            summary_document,
            scores,
        ),
    )

    result = mixed_materialize.validate_materialization_mode_gate(
        summary,
        {"meta": {}},
        pool,
    )

    assert result["status"] == "pass"
    assert result["audited_assignments"] == 1
    assert result["selected_validation_score_set_sha256"] == "22" * 32
    assert result["mode_validation_score_set_sha256"] == "33" * 32


def test_prepare_destination_publishes_an_immutable_build_pair(tmp_path) -> None:
    allocation = tmp_path / "source-allocation.json"
    allocation.write_text('{"allocation": 1}')
    mode_summary = tmp_path / "mode-summary.json"
    mode_summary.write_text('{"pass": true}')
    teacher_proxy_summary = tmp_path / "teacher-proxy-summary.json"
    teacher_proxy_summary.write_text('{"pass": true}')
    performance_summary = tmp_path / "performance-summary.json"
    performance_summary.write_text('{"pass": true}')
    destination = tmp_path / "artifact"
    build = {"kind": "unit-build", "version": 1}

    result = prepare_destination(
        destination,
        build_document=build,
        allocation_path=allocation,
        mode_validation_summary_path=mode_summary,
        teacher_proxy_summary_path=teacher_proxy_summary,
        performance_summary_path=performance_summary,
        resume=False,
    )

    assert result == destination.resolve()
    assert (result / MATERIALIZATION_BUILD_FILENAME).is_file()
    assert (
        result / MATERIALIZED_ALLOCATION_FILENAME
    ).read_bytes() == allocation.read_bytes()
    assert (
        result / mixed_materialize.MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME
    ).read_bytes() == mode_summary.read_bytes()
    assert (
        result / mixed_materialize.MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME
    ).read_bytes() == teacher_proxy_summary.read_bytes()
    assert (
        result / mixed_materialize.MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME
    ).read_bytes() == performance_summary.read_bytes()
    assert prepare_destination(
        destination,
        build_document=build,
        allocation_path=allocation,
        mode_validation_summary_path=mode_summary,
        teacher_proxy_summary_path=teacher_proxy_summary,
        performance_summary_path=performance_summary,
        resume=True,
    ) == result
    wrong_summary = tmp_path / "wrong-mode-summary.json"
    wrong_summary.write_text('{"pass": false}')
    with pytest.raises(ValueError, match="mode-validation summary"):
        prepare_destination(
            destination,
            build_document=build,
            allocation_path=allocation,
            mode_validation_summary_path=wrong_summary,
            teacher_proxy_summary_path=teacher_proxy_summary,
            performance_summary_path=performance_summary,
            resume=True,
        )
    with pytest.raises(ValueError, match="build contract"):
        prepare_destination(
            destination,
            build_document={**build, "version": 2},
            allocation_path=allocation,
            mode_validation_summary_path=mode_summary,
            teacher_proxy_summary_path=teacher_proxy_summary,
            performance_summary_path=performance_summary,
            resume=True,
        )


def test_candidate_pair_and_local_scale_split_exactly_by_rank() -> None:
    words_per_pair = PAIR_BYTES // torch.int16.itemsize
    trellis = torch.zeros(
        MATRIX_TRELLIS_BYTES // torch.int16.itemsize, dtype=torch.int16
    )
    for rank in range(12):
        trellis[rank * words_per_pair] = rank + 1
    scale = torch.arange(INTERMEDIATE_CHANNELS, dtype=torch.float16)

    for rank in (0, 5, 11):
        pair = candidate_trellis_pair(trellis, rank)
        local = candidate_local_scale(scale, rank)
        assert pair.numel() == words_per_pair
        assert int(pair[0]) == rank + 1
        begin = rank * 256
        assert torch.equal(local, scale[begin : begin + 256])


@pytest.mark.parametrize("matrix", ["w1", "w2"])
def test_official_mxfp4_rank_payload_is_codes_then_scales(matrix: str) -> None:
    out_features, in_features = C.EXPERT_SHAPES[matrix]
    packed = torch.zeros((out_features, in_features // 2), dtype=torch.uint8)
    scale = torch.zeros(
        (out_features, in_features // C.MXFP4_BLOCK), dtype=torch.uint8
    )
    if matrix == "w1":
        packed[11 * 256 :, :].fill_(17)
        scale[11 * 256 :, :].fill_(29)
    else:
        packed[:, 11 * 128 :].fill_(17)
        scale[:, 11 * 8 :].fill_(29)
    raw = PackedMXFP4Matrix(packed, scale)

    rank_packed, rank_scale = mxfp4_rank_matrix_components(raw, matrix, 11)

    assert rank_packed.is_contiguous() and rank_scale.is_contiguous()
    assert rank_packed.numel() == 256 * 3584 // 2
    assert rank_scale.numel() == 256 * 3584 // 32
    assert bool(torch.all(rank_packed == 17))
    assert bool(torch.all(rank_scale == 29))


def test_pwrite_exact_preserves_explicit_offsets(tmp_path) -> None:
    path = tmp_path / "offsets.bin"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, 12)
        pwrite_exact(descriptor, b"abc", 2)
        tensor = torch.tensor([0x6465, 0x6667], dtype=torch.int16)
        pwrite_exact(descriptor, memoryview(tensor.numpy()).cast("B"), 6)
    finally:
        os.close(descriptor)

    assert path.read_bytes() == b"\x00\x00abc\x00edgf\x00\x00"


def test_sparse_layer_structure_validator_closes_fixed_sections(tmp_path) -> None:
    formats = tuple(ExpertFormatSpec.compressed(0) for _ in range(C.NUM_EXPERTS))
    layout = TP12LayerLayout(C.NUM_EXPERTS, 0)
    spec = LayerMaterializationSpec(
        layer=1,
        formats=formats,
        compressed=tuple(range(C.NUM_EXPERTS)),
        kept=(),
        layout=layout,
    )
    path = tmp_path / "layer.bin"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, layout.disk_bytes)
        header = TP12LayerHeader(1, layout)
        pwrite_exact(descriptor, header.to_bytes(), 0)
        pwrite_exact(
            descriptor,
            memoryview(pack_tp12_format_section(formats).numpy()),
            LAYER_HEADER_BYTES,
        )
        shared_w1 = torch.arange(LATENT_CHANNELS, dtype=torch.float16)
        shared_w3 = torch.zeros(LATENT_CHANNELS, dtype=torch.float16)
        shared_w2 = torch.zeros(LATENT_CHANNELS, dtype=torch.float16)
        shared = pack_tp12_shared_scale_section(
            shared_w1,
            shared_w3,
            shared_w2,
        )
        pwrite_exact(
            descriptor,
            memoryview(shared.numpy()),
            LAYER_HEADER_BYTES + FORMAT_SECTION_BYTES,
        )
        expert = 17
        source_parts: dict[str, dict[str, torch.Tensor]] = {}
        for matrix_index, matrix in enumerate(C.EXPERT_MATRICES):
            trellis = torch.zeros(
                MATRIX_TRELLIS_BYTES // torch.int16.itemsize,
                dtype=torch.int16,
            )
            local = torch.arange(INTERMEDIATE_CHANNELS, dtype=torch.float16)
            local.add_(matrix_index)
            for rank in range(12):
                logical_pair = tp12_logical_pair_index(1, expert, rank)
                pair = candidate_trellis_pair(trellis, logical_pair)
                pair[0] = 100 * matrix_index + logical_pair + 1
                pwrite_exact(
                    descriptor,
                    memoryview(pair.numpy()),
                    layout.trellis_pair_offset(rank, expert, matrix),
                )
                pwrite_exact(
                    descriptor,
                    memoryview(
                        candidate_local_scale(local, logical_pair).numpy()
                    ),
                    layout.local_scale_offset(rank, expert, matrix),
                )
            if matrix == "w1":
                source_parts[matrix] = {
                    "trellis": trellis,
                    "suh": shared_w1,
                    "svh": local,
                }
            elif matrix == "w3":
                source_parts[matrix] = {
                    "trellis": trellis,
                    "suh": shared_w3,
                    "svh": local,
                }
            else:
                source_parts[matrix] = {
                    "trellis": trellis,
                    "suh": local,
                    "svh": shared_w2,
                }
    finally:
        os.close(descriptor)

    result = validate_materialized_layer(path, spec)

    assert result["disk_bytes"] == layout.disk_bytes
    assert result["compressed_experts"] == C.NUM_EXPERTS
    assert path.stat().st_blocks * 512 < 64 * 1024 * 1024
    with TP12LayerReader(path) as reader:
        restored = reader.read_compressed_matrix(expert, "w1")
        for part, expected in source_parts["w1"].items():
            assert torch.equal(restored[part], expected)
        with pytest.raises(ValueError, match="not in the MXFP4 tier"):
            reader.read_kept_matrix(expert, "w1")

    candidate_dir = tmp_path / "candidate" / "candidates"
    candidate_dir.mkdir(parents=True)
    candidate_tensors = {
        candidate_tensor_name(1, expert, matrix, part): value
        for matrix, parts in source_parts.items()
        for part, value in parts.items()
    }
    save_file(
        candidate_tensors,
        str(candidate_dir / "mixed-exl3-layer-00001.safetensors"),
    )
    closure = validate_materialized_layer_payloads(
        path,
        spec,
        candidate_dir.parent,
        object(),  # No retained experts are requested in this closure.
        compressed_experts=(expert,),
        kept_experts=(),
    )
    assert closure["payload_closure"] == "bit_exact"
    assert closure["compressed_experts_verified"] == 1

    full_closure = {
        **result,
        "payload_closure": "bit_exact",
        "compressed_experts_verified": len(spec.compressed),
        "kept_experts_verified": len(spec.kept),
        "logical_source_bytes_verified": expected_logical_source_bytes(spec),
    }
    build = {"kind": "unit-build", "source": "immutable"}
    receipt = write_layer_closure_receipt(
        path,
        spec,
        full_closure,
        build_document=build,
    )
    assert receipt.is_file()
    assert load_layer_closure_receipt(
        path,
        spec,
        build_document=build,
    ) == full_closure
    with pytest.raises(ValueError, match="build_sha256"):
        load_layer_closure_receipt(
            path,
            spec,
            build_document={**build, "source": "changed"},
        )


def test_layer_reader_reassembles_kept_w2_across_tp12_columns(tmp_path) -> None:
    expert = 31
    formats = [ExpertFormatSpec.compressed(0) for _ in range(C.NUM_EXPERTS)]
    formats[expert] = ExpertFormatSpec.mxfp4()
    formats = tuple(formats)
    compressed = tuple(value for value in range(C.NUM_EXPERTS) if value != expert)
    layout = TP12LayerLayout(len(compressed), 1)
    spec = LayerMaterializationSpec(
        layer=1,
        formats=formats,
        compressed=compressed,
        kept=(expert,),
        layout=layout,
    )
    out_features, in_features = C.EXPERT_SHAPES["w2"]
    packed = torch.zeros((out_features, in_features // 2), dtype=torch.uint8)
    scale = torch.zeros(
        (out_features, in_features // C.MXFP4_BLOCK), dtype=torch.uint8
    )
    for rank in range(12):
        packed[:, rank * 128 : (rank + 1) * 128].fill_(rank + 1)
        scale[:, rank * 8 : (rank + 1) * 8].fill_(rank + 33)
    raw = PackedMXFP4Matrix(packed, scale)

    path = tmp_path / "kept-layer.bin"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, layout.disk_bytes)
        header = TP12LayerHeader(1, layout)
        pwrite_exact(descriptor, header.to_bytes(), 0)
        pwrite_exact(
            descriptor,
            memoryview(pack_tp12_format_section(formats).numpy()),
            header.format_offset,
        )
        shared = pack_tp12_shared_scale_section(
            *[torch.zeros(LATENT_CHANNELS, dtype=torch.float16) for _ in range(3)]
        )
        pwrite_exact(
            descriptor,
            memoryview(shared.numpy()),
            header.shared_scale_offset,
        )
        for rank in range(12):
            rank_packed, rank_scale = mxfp4_rank_matrix_components(raw, "w2", rank)
            offset = layout.kept_matrix_offset(rank, 0, "w2")
            pwrite_exact(descriptor, memoryview(rank_packed.numpy()), offset)
            pwrite_exact(
                descriptor,
                memoryview(rank_scale.numpy()),
                offset + rank_packed.numel(),
            )
    finally:
        os.close(descriptor)

    with TP12LayerReader(path) as reader:
        restored = reader.read_kept_matrix(expert, "w2")
        assert torch.equal(restored.packed, packed)
        assert torch.equal(restored.scale, scale)
        with pytest.raises(ValueError, match="not in the compressed tier"):
            reader.read_compressed_matrix(expert, "w2")


def test_external_x4_sidecar_closes_exact_kept_payload(tmp_path) -> None:
    expert = 7
    formats = [ExpertFormatSpec.compressed(0) for _ in range(C.NUM_EXPERTS)]
    formats[expert] = ExpertFormatSpec.mxfp4()
    formats = tuple(formats)
    compressed = tuple(value for value in range(C.NUM_EXPERTS) if value != expert)
    layout = TP12LayerLayout(
        len(compressed),
        1,
        keep_storage=KEEP_STORAGE_EXTERNAL_X4,
    )
    spec = LayerMaterializationSpec(
        layer=1,
        formats=formats,
        compressed=compressed,
        kept=(expert,),
        layout=layout,
    )
    slab = tmp_path / "sqrt-c-layer.bin"
    descriptor = os.open(slab, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, layout.disk_bytes)
        header = TP12LayerHeader(1, layout)
        pwrite_exact(descriptor, header.to_bytes(), 0)
        pwrite_exact(
            descriptor,
            memoryview(pack_tp12_format_section(formats).numpy()),
            header.format_offset,
        )
        shared = pack_tp12_shared_scale_section(
            *[torch.zeros(LATENT_CHANNELS, dtype=torch.float16) for _ in range(3)]
        )
        pwrite_exact(descriptor, memoryview(shared.numpy()), header.shared_scale_offset)
    finally:
        os.close(descriptor)

    source: dict[str, PackedMXFP4Matrix] = {}
    sidecar = tmp_path / "x4-layer.bin"
    with X4LayerWriter(sidecar, layer=1) as writer:
        for matrix in C.EXPERT_MATRICES:
            out_features, in_features = C.EXPERT_SHAPES[matrix]
            packed = torch.zeros(
                (out_features, in_features // 2), dtype=torch.uint8
            )
            scale = torch.full(
                (out_features, in_features // C.MXFP4_BLOCK),
                120,
                dtype=torch.uint8,
            )
            scale[:, 1::2] = 121
            source[matrix] = PackedMXFP4Matrix(packed, scale)
            writer.add(expert, matrix, packed, scale)

    class Store:
        def load_packed_matrix(self, layer, requested_expert, matrix):
            assert layer == 1 and requested_expert == expert
            return source[matrix]

    closure = validate_materialized_layer_payloads(
        slab,
        spec,
        tmp_path / "unused-candidate-root",
        Store(),
        compressed_experts=(),
        kept_experts=(expert,),
        x4_path=sidecar,
    )

    assert closure["payload_closure"] == "bit_exact"
    assert closure["kept_experts_verified"] == 1
    with TP12LayerReader(slab) as reader:
        with pytest.raises(ValueError, match="external X4"):
            reader.read_kept_matrix(expert, "w1")


def test_layer_reader_emits_b12x_native_compressed_rank_payload(tmp_path) -> None:
    expert = 17
    rank = 5
    formats = [ExpertFormatSpec.mxfp4() for _ in range(C.NUM_EXPERTS)]
    formats[expert] = ExpertFormatSpec.compressed(3, 7)
    formats = tuple(formats)
    kept = tuple(value for value in range(C.NUM_EXPERTS) if value != expert)
    layout = TP12LayerLayout(1, len(kept))
    spec = LayerMaterializationSpec(
        layer=1,
        formats=formats,
        compressed=(expert,),
        kept=kept,
        layout=layout,
    )
    shared_scales = tuple(
        torch.full((LATENT_CHANNELS,), value, dtype=torch.float16)
        for value in (1.0, 2.0, 3.0)
    )
    pair_values = (101, 202, 303)
    local_values = (11.0, 22.0, 33.0)

    path = tmp_path / "rank-runtime-layer.bin"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, layout.disk_bytes)
        header = TP12LayerHeader(1, layout)
        pwrite_exact(descriptor, header.to_bytes(), 0)
        pwrite_exact(
            descriptor,
            memoryview(pack_tp12_format_section(formats).numpy()),
            header.format_offset,
        )
        shared = pack_tp12_shared_scale_section(*shared_scales)
        pwrite_exact(
            descriptor,
            memoryview(shared.numpy()),
            header.shared_scale_offset,
        )
        words = PAIR_BYTES // torch.int16.itemsize
        for matrix, pair_value, local_value in zip(
            C.EXPERT_MATRICES, pair_values, local_values, strict=True
        ):
            pair = torch.full((words,), pair_value, dtype=torch.int16)
            local = torch.full((256,), local_value, dtype=torch.float16)
            pwrite_exact(
                descriptor,
                memoryview(pair.numpy()),
                layout.trellis_pair_offset(rank, 0, matrix),
            )
            pwrite_exact(
                descriptor,
                memoryview(local.numpy()),
                layout.local_scale_offset(rank, 0, matrix),
            )
    finally:
        os.close(descriptor)

    with TP12LayerReader(path) as reader:
        payload = reader.read_compressed_rank(rank)

    assert spec.layout == layout
    assert payload.num_experts == 1
    assert payload.expert_ids.tolist() == [expert]
    assert tuple(payload.w13_trellis.shape) == (2, 1, words)
    assert bool(torch.all(payload.w13_trellis[0, 0] == pair_values[0]))
    assert bool(torch.all(payload.w13_trellis[1, 0] == pair_values[1]))
    assert bool(torch.all(payload.w2_trellis[0] == pair_values[2]))
    assert payload.fc1_pair_modes.tolist() == [0]
    assert payload.fc2_pair_modes.tolist() == [1]
    assert torch.equal(payload.gate_suh[0], shared_scales[0])
    assert torch.equal(payload.up_suh[0], shared_scales[1])
    assert torch.equal(payload.down_svh[0], shared_scales[2])
    assert payload.intermediate_rotations[0, :256].tolist() == [11.0] * 256
    assert payload.intermediate_rotations[0, 256:512].tolist() == [22.0] * 256
    assert payload.intermediate_rotations[0, 512:].tolist() == [33.0] * 256
    kwargs = payload.b12x_prepare_kwargs()
    assert kwargs["w1_fp4"] is payload.w13_trellis
    assert kwargs["w2_fp4"] is payload.w2_trellis
    assert kwargs["trellis_fc1_pair_modes"] is payload.fc1_pair_modes
    assert kwargs["trellis_fc2_pair_modes"] is payload.fc2_pair_modes
