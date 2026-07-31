from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from kquant.trace_compare import (
    TraceFormatError,
    compare_traces,
    load_trace,
    select_trace_call,
)


def _digest(tensor: torch.Tensor) -> str:
    data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _write_trace(
    root: Path,
    *,
    world_size: int,
    stages: dict[str, list[torch.Tensor]],
    call: int = 0,
) -> None:
    ranks = list(range(world_size))
    for rank in ranks:
        rank_dir = root / f"tp-rank-{rank:03d}"
        rank_dir.mkdir(parents=True)
        (rank_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "pid": 1000 + rank,
                    "tp_rank": rank,
                    "tp_world_size": world_size,
                    "layers": [1],
                    "ranks": ranks,
                    "start_call": call,
                    "max_calls": 1,
                    "max_tokens": 16,
                }
            )
        )
        for stage, values in stages.items():
            tensor = values[rank]
            metadata = {
                "schema_version": 1,
                "layer": 1,
                "call": call,
                "stage": stage,
                "tp_rank": rank,
                "tp_world_size": world_size,
                "original_shape": list(tensor.shape),
                "saved_shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "source_device": f"cuda:{rank}",
                "sha256": _digest(tensor),
            }
            torch.save(
                {"metadata": metadata, "tensor": tensor},
                rank_dir / f"layer-001.call-{call:06d}.{stage}.pt",
            )


def test_compare_traces_aggregates_partials_and_checks_routes(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    ids = torch.tensor([[2, 7]], dtype=torch.int32)
    left_output = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    right_output = torch.tensor([[1.01, 1.99]], dtype=torch.bfloat16)
    _write_trace(
        left,
        world_size=2,
        stages={
            "moe_output": [left_output, left_output.clone()],
            "canonical_topk_ids": [ids, ids.clone()],
            "routed_latent_partial": [
                torch.tensor([[0.25, 0.5]], dtype=torch.bfloat16),
                torch.tensor([[0.75, 1.5]], dtype=torch.bfloat16),
            ],
        },
    )
    _write_trace(
        right,
        world_size=3,
        stages={
            "moe_output": [right_output, right_output.clone(), right_output.clone()],
            "canonical_topk_ids": [ids, ids.clone(), ids.clone()],
            "routed_latent_partial": [
                torch.tensor([[0.25, 0.25]], dtype=torch.bfloat16),
                torch.tensor([[0.25, 0.75]], dtype=torch.bfloat16),
                torch.tensor([[0.5, 1.0]], dtype=torch.bfloat16),
            ],
        },
    )

    report = compare_traces(
        load_trace(left),
        load_trace(right),
        max_relative_l2=0.02,
        min_cosine=0.999,
        max_route_mismatch_fraction=0,
    )

    assert report["pass"] is True
    partial = next(
        item
        for item in report["comparisons"]
        if item["stage"] == "routed_latent_partial"
    )
    assert partial["reduction"] == "fp32_rank_sum"
    assert partial["relative_l2"] == 0


def test_compare_traces_rejects_rank_replication_drift(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    base = torch.tensor([[1.0]], dtype=torch.bfloat16)
    _write_trace(
        left,
        world_size=2,
        stages={"moe_output": [base, base + 1]},
    )
    _write_trace(
        right,
        world_size=2,
        stages={"moe_output": [base, base.clone()]},
    )

    report = compare_traces(load_trace(left), load_trace(right))

    assert report["pass"] is False
    assert report["replicated_rank_mismatches"][0]["mismatched_ranks"] == [1]


def test_load_trace_rejects_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    tensor = torch.tensor([[1.0]], dtype=torch.bfloat16)
    _write_trace(root, world_size=1, stages={"moe_output": [tensor]})
    path = root / "tp-rank-000/layer-001.call-000000.moe_output.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["tensor"] += 1
    torch.save(payload, path)

    with pytest.raises(TraceFormatError, match="digest mismatch"):
        load_trace(root)


def test_compare_traces_enforces_route_mismatch_limit(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    right_ids = torch.tensor([[1, 8, 3, 9]], dtype=torch.int32)
    _write_trace(
        left,
        world_size=1,
        stages={"canonical_topk_ids": [left_ids]},
    )
    _write_trace(
        right,
        world_size=1,
        stages={"canonical_topk_ids": [right_ids]},
    )

    report = compare_traces(
        load_trace(left),
        load_trace(right),
        max_route_mismatch_fraction=0.25,
    )

    assert report["pass"] is False
    assert report["comparisons"][0]["membership_mismatch_fraction"] == 0.5


def test_compare_traces_ignores_route_order_and_aligns_weights(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_trace(
        left,
        world_size=1,
        stages={
            "canonical_topk_ids": [
                torch.tensor([[2, 7, 5]], dtype=torch.int32)
            ],
            "canonical_topk_weights": [
                torch.tensor([[0.2, 0.7, 0.5]], dtype=torch.float32)
            ],
        },
    )
    _write_trace(
        right,
        world_size=1,
        stages={
            "canonical_topk_ids": [
                torch.tensor([[7, 5, 2]], dtype=torch.int32)
            ],
            "canonical_topk_weights": [
                torch.tensor([[0.7, 0.5, 0.2]], dtype=torch.float32)
            ],
        },
    )

    report = compare_traces(load_trace(left), load_trace(right))

    assert report["pass"] is True
    ids = next(
        item
        for item in report["comparisons"]
        if item["stage"] == "canonical_topk_ids"
    )
    weights = next(
        item
        for item in report["comparisons"]
        if item["stage"] == "canonical_topk_weights"
    )
    assert ids["position_mismatch_count"] == 3
    assert ids["membership_mismatch_count"] == 0
    assert weights["exact"] is True
    assert weights["reduction"] == "replicated_aligned_by_expert_id"


def test_select_trace_call_normalizes_invocations_and_allows_extra_keys(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    value = torch.tensor([[1.0]], dtype=torch.bfloat16)
    _write_trace(
        left,
        world_size=1,
        stages={"layer_boundary": [value]},
        call=0,
    )
    _write_trace(
        right,
        world_size=1,
        stages={
            "layer_boundary": [value.clone()],
            "implementation_detail": [value.clone()],
        },
        call=3,
    )

    left_run = select_trace_call(load_trace(left), 0)
    right_run = select_trace_call(load_trace(right), 3)
    strict = compare_traces(left_run, right_run)
    shared = compare_traces(
        left_run,
        right_run,
        allow_key_coverage_difference=True,
    )

    assert strict["pass"] is False
    assert shared["pass"] is True
    assert shared["comparisons"][0]["stage"] == "layer_boundary"
    with pytest.raises(TraceFormatError, match="available=\\[3\\]"):
        select_trace_call(load_trace(right), 2)
