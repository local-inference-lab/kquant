from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from kquant import constants as C
from kquant.io.safetensors_stream import TensorSpec
from kquant.exl3_reference import CODEBOOK_SQG_NORMAL_E4M3
from kquant.qsrt import SCHEMA
import scripts.pack_qsrt_candidates_tp12 as pack_script
from scripts.pack_qsrt_candidates_tp12 import (
    _ScheduleJob,
    _balanced_expert_schedule,
    _balanced_layer_schedule,
    _bind_candidate_manifest,
    _manifest,
    _allocate_metrics,
    _merge_scheduled_layer,
    _layer_rotation_plan,
    _schedule_from_report,
    _schedule_report,
)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        official_revision=C.REVISION,
        exllamav3_root=tmp_path / "exllamav3",
        teacher_checkpoint=tmp_path / "teacher",
        capture=tmp_path / "train.kqcapture",
        training_report=tmp_path / "train.json",
        hessians=tmp_path / "train.kqhess",
        hessian_policy="captured_blend",
        mode_ids=(0, 1, 2),
        min_fit_documents=6,
        min_confirmation_documents=4,
        minimum_improvement=0.0,
        bootstrap_replicates=2_000,
        logical_trellis_schema=SCHEMA,
        codebook=CODEBOOK_SQG_NORMAL_E4M3,
        layout="qsrt_guarded_reuse",
        tailbite_context=128,
    )


def test_new_candidate_manifest_freezes_explicit_logical_schema(tmp_path) -> None:
    args = _args(tmp_path)
    path = tmp_path / "qsrt-candidate-manifest.json"

    actual = _bind_candidate_manifest(args, path, create=True)

    assert actual == _manifest(args)
    assert actual["logical_trellis_schema"] == SCHEMA
    assert actual["codebook"] == CODEBOOK_SQG_NORMAL_E4M3
    assert actual["layout"] == "qsrt_guarded_reuse"
    assert actual["permutation_policy"] == "h2_reverse"
    assert actual["hessian_policy"] == "captured_blend"
    assert actual["h2_contract"]["indexed_by"] == "expert_r13"
    assert actual["h2_contract"]["down_candidate_grid"] == "w2_r13_r2"
    assert actual["h2_contract"]["prior"] == "expert_local_trace_scaled_identity"
    assert actual["h2_contract"]["unsupported_expert_fallback"] == "identity"
    assert actual["rotation_draws"] == {
        "source": "fixed_cli_draws",
        "fixed": {
            "residual_draw": 0,
            "intermediate_default": 0,
            "intermediate_overrides": {},
        },
    }
    assert json.loads(path.read_text()) == actual


def test_candidate_manifest_embeds_rotation_plan_contents(tmp_path) -> None:
    args = _args(tmp_path)
    args.rotation_plan = tmp_path / "rotations.json"
    args.rotation_plan.write_text(
        json.dumps(
            {
                "kind": "kquant_qsrt_rotation_plan",
                "schema_version": 1,
                "layers": {
                    "12": {
                        "residual_draw": 6,
                        "intermediate_overrides": {"102": 7},
                    }
                },
            }
        )
    )

    manifest = _manifest(args)
    layer_plan = _layer_rotation_plan(args, 12)

    assert manifest["rotation_draws"]["source"] == "model_rotation_plan"
    assert manifest["rotation_draws"]["plan"]["layers"]["12"] == {
        "residual_draw": 6,
        "intermediate_default": 0,
        "intermediate_overrides": {"102": 7},
    }
    assert layer_plan.residual_draw == 6
    assert layer_plan.intermediate_draw(102) == 7
    assert layer_plan.intermediate_draw(103) == 0


def test_candidate_manifest_rejects_non_schema_drift(tmp_path) -> None:
    args = _args(tmp_path)
    path = tmp_path / "qsrt-candidate-manifest.json"
    drifted = _manifest(args)
    drifted["mode_ids"] = [0, 1]
    path.write_text(json.dumps(drifted))

    with pytest.raises(ValueError, match="does not match"):
        _bind_candidate_manifest(args, path, create=False)


def test_balanced_layer_schedule_is_complete_deterministic_and_balanced() -> None:
    costs = {layer: cost for layer, cost in enumerate((9, 8, 7, 6, 5, 4), start=1)}

    schedule = _balanced_layer_schedule(costs, list(costs), workers=3)

    assert schedule == ((1, 6), (2, 5), (3, 4))
    assert sorted(layer for worker in schedule for layer in worker) == list(costs)
    assert [sum(costs[layer] for layer in worker) for worker in schedule] == [13, 13, 13]


def test_balanced_layer_schedule_rejects_duplicate_layers() -> None:
    with pytest.raises(ValueError, match="unique"):
        _balanced_layer_schedule({1: 1}, [1, 1], workers=2)


def test_expert_balanced_schedule_splits_only_enough_to_close_worker_tail() -> None:
    layers = list(range(1, 8))
    expert_costs = {
        layer: tuple(
            2 if (expert + layer) % 5 else 1
            for expert in range(C.NUM_EXPERTS)
        )
        for layer in layers
    }

    schedule = _balanced_expert_schedule(
        expert_costs,
        layers,
        workers=3,
        expert_batch_size=20,
    )

    ownership = [
        (job.layer, expert)
        for jobs in schedule
        for job in jobs
        for expert in job.experts
    ]
    assert len(ownership) == len(set(ownership)) == len(layers) * C.NUM_EXPERTS
    loads = [sum(job.estimated_work for job in jobs) for jobs in schedule]
    target = sum(loads) / len(loads)
    assert max(loads) <= target + 40
    assert any(
        sum(job.layer == layer for jobs in schedule for job in jobs) > 1
        for layer in layers
    )
    assert all(len({job.layer for job in jobs}) == len(jobs) for jobs in schedule)


def test_expert_schedule_report_round_trips_complete_model() -> None:
    expert_costs = {
        layer: tuple(
            136 if (expert + layer) % 11 else 72
            for expert in range(C.NUM_EXPERTS)
        )
        for layer in C.MOE_LAYERS
    }
    schedule = _balanced_expert_schedule(
        expert_costs,
        list(C.MOE_LAYERS),
        expert_batch_size=20,
    )
    full_grid = {
        layer: sum(cost == 136 for cost in expert_costs[layer])
        for layer in C.MOE_LAYERS
    }

    report = _schedule_report(
        schedule,
        expert_batch_size=20,
        r0_work=72,
        full_grid_work=136,
        full_grid_experts_by_layer=full_grid,
    )

    assert _schedule_from_report(report) == schedule
    assert report["max_over_mean"] < 1.01
    assert report["split_layer_count"] > 0


def test_split_layer_merge_publishes_one_canonical_atomic_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(C, "NUM_EXPERTS", 4)

    def payload_specs(layer: int, experts: list[int]):
        return tuple(
            TensorSpec(f"layer{layer}.expert{expert}", torch.int16, (2,))
            for expert in experts
        )

    def metrics(experts, modes, fit_documents, confirmation_documents):
        count = len(experts)
        return {
            "expert_ids": torch.tensor(experts, dtype=torch.int16),
            "mode_ids": torch.tensor(modes, dtype=torch.uint8),
            "selected_r13": torch.zeros(count, dtype=torch.uint8),
            "selected_r2": torch.zeros(count, dtype=torch.uint8),
        }

    monkeypatch.setattr(pack_script, "_candidate_payload_specs", payload_specs)
    monkeypatch.setattr(pack_script, "_allocate_metrics", metrics)
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    jobs = [
        _ScheduleJob(1, (0, 1), 2),
        _ScheduleJob(1, (2, 3), 2),
    ]
    for job in jobs:
        stem = pack_script._layer_stem(1, list(job.experts))
        save_file(
            {
                f"layer1.expert{expert}": torch.tensor(
                    [expert, expert + 10], dtype=torch.int16
                )
                for expert in job.experts
            },
            str(candidate_dir / f"{stem}.safetensors"),
            metadata={"codebook": "sqg-normal-e4m3", "tailbite_context": "128"},
        )
        save_file(
            metrics(list(job.experts), (0, 1), 1, 1),
            str(candidate_dir / f"{stem}.metrics.safetensors"),
        )
        (candidate_dir / f"{stem}.selection.json").write_text(
            json.dumps(
                {
                    "experts": list(job.experts),
                    "seconds": 1.0,
                    "selections": {
                        str(expert): {"selected_r13": 0, "selected_r2": 0}
                        for expert in job.experts
                    },
                }
            )
        )
    args = SimpleNamespace(
        dest=tmp_path,
        resume=False,
        codebook="sqg-normal-e4m3",
        mode_ids=(0, 1),
        tailbite_context=128,
    )
    partition = SimpleNamespace(fit=(0,), confirmation=(1,))

    _merge_scheduled_layer(args, 1, jobs, partition)

    stem = "qsrt-layer-00001"
    payload = load_file(str(candidate_dir / f"{stem}.safetensors"))
    assert set(payload) == {
        f"layer1.expert{expert}" for expert in range(C.NUM_EXPERTS)
    }
    assert torch.equal(
        load_file(str(candidate_dir / f"{stem}.metrics.safetensors"))["expert_ids"],
        torch.arange(C.NUM_EXPERTS, dtype=torch.int16),
    )
    ledger = json.loads((candidate_dir / f"{stem}.selection.json").read_text())
    assert ledger["all_experts"] is True
    assert ledger["selected_format_histogram"]["R0/R0"] == C.NUM_EXPERTS
