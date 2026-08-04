from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kquant.teacher_proxy import (
    DEFAULT_FLOAT_STAGES,
    align_routed_post_situ,
    compare_teacher_proxy_traces,
)
from kquant.trace_compare import TraceFormatError, TraceKey, TraceRun
from scripts.stream_k3_pytorch import LayerCaptures


def _run(
    root: Path,
    *,
    ids: torch.Tensor,
    weights: torch.Tensor,
    offset: float,
    post_situ: torch.Tensor | None = None,
) -> TraceRun:
    tensors: dict[TraceKey, dict[int, torch.Tensor]] = {
        TraceKey(1, 0, "canonical_topk_ids"): {0: ids},
        TraceKey(1, 0, "canonical_topk_weights"): {0: weights},
    }
    for index, stage in enumerate(DEFAULT_FLOAT_STAGES, start=1):
        value = torch.tensor(
            [[float(index), float(index + 1)]], dtype=torch.bfloat16
        )
        tensors[TraceKey(1, 0, stage)] = {0: value + offset}
    if post_situ is not None:
        tensors[TraceKey(1, 0, "routed_post_situ")] = {0: post_situ}
    return TraceRun(
        root=root,
        world_size=1,
        captured_ranks=(0,),
        manifest={"max_tokens": 2, "checkpoint": str(root / "checkpoint")},
        tensors=tensors,
    )


def test_align_routed_post_situ_reverses_official_expert_grouping() -> None:
    ids = torch.tensor([[2, 1], [2, 1], [1, 2]], dtype=torch.int32)
    flat = ids.flatten().long()
    grouped_positions = flat.argsort()
    values: dict[int, torch.Tensor] = {}
    cursor = 0
    for expert_id in sorted(set(flat.tolist())):
        count = int((flat == expert_id).sum())
        positions = grouped_positions[cursor : cursor + count]
        values[expert_id] = torch.stack(
            (positions.float(), positions.float() + 10), dim=1
        )
        cursor += count

    aligned = align_routed_post_situ(ids, values)

    expected_positions = torch.arange(flat.numel(), dtype=torch.float32)
    assert aligned.shape == (3, 2, 2)
    assert torch.equal(aligned[..., 0].flatten(), expected_positions)
    assert torch.equal(aligned[..., 1].flatten(), expected_positions + 10)


def test_stream_layer_capture_emits_route_aligned_post_situ() -> None:
    ids = torch.tensor([[2, 1], [2, 1], [1, 2]], dtype=torch.int64)
    weights = torch.full((3, 2), 0.5)

    class Gate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3, 2))

        def forward(
            self, unused_hidden_states: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return ids, weights

    class Expert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.act_fn = torch.nn.Identity()

    class Moe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = Gate()
            self.experts = torch.nn.ModuleList([Expert() for _ in range(3)])
            self.routed_expert_down_proj = torch.nn.Identity()
            self.routed_expert_up_proj = torch.nn.Identity()

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = torch.nn.Identity()
            self.block_sparse_moe = Moe()

    layer = Layer()
    captures = LayerCaptures(layer, capture_routed_post_situ=True)
    layer.block_sparse_moe.gate(torch.ones(3, 2))
    flat = ids.flatten()
    grouped_positions = flat.argsort()
    cursor = 0
    for expert_id in (1, 2):
        count = int((flat == expert_id).sum())
        positions = grouped_positions[cursor : cursor + count]
        values = torch.stack(
            (positions.float(), positions.float() + 10), dim=1
        )
        layer.block_sparse_moe.experts[expert_id].act_fn(values)
        cursor += count
    captures.finalize()

    aligned = captures.values["routed_post_situ"]
    expected_positions = torch.arange(flat.numel(), dtype=torch.float32)
    assert aligned.shape == (3, 2, 2)
    assert torch.equal(aligned[..., 0].flatten(), expected_positions)
    captures.close()


def test_teacher_proxy_report_uses_route_sets_and_sparse_gate_alignment(
    tmp_path: Path,
) -> None:
    official = _run(
        tmp_path / "official",
        ids=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        weights=torch.tensor([[0.6, 0.4], [0.7, 0.3]]),
        offset=0.0,
    )
    interim = _run(
        tmp_path / "interim",
        ids=torch.tensor([[0, 4], [3, 2]], dtype=torch.int32),
        weights=torch.tensor([[0.55, 0.45], [0.3, 0.7]]),
        offset=0.125,
    )

    report = compare_teacher_proxy_traces(
        official, interim, num_experts=5
    )

    routes = report["route_summary"]
    assert report["status"] == "descriptive_only"
    assert report["post_situ_summary"]["available"] is False
    assert routes["route_assignments"] == 4
    assert routes["common_route_assignments"] == 3
    assert routes["route_assignment_retention_fraction"] == pytest.approx(0.75)
    assert routes["exact_route_set_row_fraction"] == pytest.approx(0.5)
    assert routes["route_position_match_fraction"] == pytest.approx(0.25)
    assert routes["row_jaccard_mean"] == pytest.approx(2 / 3)
    assert routes["sparse_applied_gate"]["elements"] == 10
    assert routes["common_route_gate"]["elements"] == 3
    assert report["float_stage_summary"]["router_logits"]["relative_l2"] > 0


def test_teacher_proxy_compares_post_situ_record_rankings(tmp_path: Path) -> None:
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
    record_low = torch.ones(128)
    record_high = torch.full((128,), 2.0)
    official_post_situ = torch.cat((record_low, record_high)).repeat(2, 2, 1)
    interim_post_situ = official_post_situ * 2
    official = _run(
        tmp_path / "official",
        ids=ids,
        weights=weights,
        offset=0.0,
        post_situ=official_post_situ,
    )
    interim = _run(
        tmp_path / "interim",
        ids=ids.clone(),
        weights=weights.clone(),
        offset=0.0,
        post_situ=interim_post_situ,
    )

    report = compare_teacher_proxy_traces(
        official,
        interim,
        num_experts=4,
        min_expert_post_situ_routes=1,
    )

    post_situ = report["post_situ_summary"]
    assert post_situ["available"] is True
    assert post_situ["direct_common_route_activation"]["relative_l2"] == 1
    assert post_situ["layer_record_rank_spearman"]["mean"] == 1
    assert post_situ["expert_record_rank_spearman"]["count"] == 4
    assert (
        post_situ["layers"][0]["rate_ladder_record_overlap"]["R1"]
        ["donor_record_overlap_fraction"]
        == 1
    )


def test_teacher_proxy_rejects_duplicate_route_ids(tmp_path: Path) -> None:
    official = _run(
        tmp_path / "official",
        ids=torch.tensor([[0, 0]], dtype=torch.int32),
        weights=torch.tensor([[0.5, 0.5]]),
        offset=0.0,
    )
    interim = _run(
        tmp_path / "interim",
        ids=torch.tensor([[0, 1]], dtype=torch.int32),
        weights=torch.tensor([[0.5, 0.5]]),
        offset=0.0,
    )

    with pytest.raises(TraceFormatError, match="duplicate experts"):
        compare_teacher_proxy_traces(official, interim, num_experts=2)
