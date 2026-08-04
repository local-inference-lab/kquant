from __future__ import annotations

import pytest
import torch

from kquant.mixed_exl3 import (
    CONTEXT_GROUP_CHANNELS,
    INTERMEDIATE_CHANNELS,
    RATE_TRANSFER_MODES,
    RECORDS_PER_EXPERT,
    tp12_record_bits,
)
from kquant.pack.mixed_exl3 import _mixed_quant_args, plan_mixed_matrix


def _scrambled_record_contexts() -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    contexts = torch.arange(RECORDS_PER_EXPERT).repeat_interleave(32)
    return contexts[torch.randperm(contexts.numel(), generator=generator)]


@pytest.mark.parametrize(
    ("matrix", "rate_axis"), (("w1", "n"), ("w3", "n"), ("w2", "k"))
)
@pytest.mark.parametrize("r", [0, 1, 3, 5, 12])
def test_mixed_matrix_plan_closes_the_physical_tp12_schedule(
    matrix: str, rate_axis: str, r: int
) -> None:
    contexts = _scrambled_record_contexts()
    mode = RATE_TRANSFER_MODES[r]

    plan = plan_mixed_matrix(contexts, mode, matrix=matrix)

    assert plan.rate_axis == rate_axis
    assert plan.encoder_permutation.shape == (INTERMEDIATE_CHANNELS,)
    assert plan.physical_permutation.shape == (INTERMEDIATE_CHANNELS,)
    assert sorted(plan.record_repack_order.tolist()) == list(
        range(RECORDS_PER_EXPERT)
    )
    assert plan.physical_tile_bits == tuple(
        bits for bits in tp12_record_bits(mode) for _ in range(8)
    )
    assert plan.physical_tp12_rank_bpw == (3.0,) * 12


def test_common_physical_permutation_is_matrix_independent() -> None:
    contexts = _scrambled_record_contexts()
    plans = [
        plan_mixed_matrix(contexts, RATE_TRANSFER_MODES[3], matrix=matrix)
        for matrix in ("w1", "w3", "w2")
    ]

    assert torch.equal(plans[0].physical_permutation, plans[1].physical_permutation)
    assert torch.equal(plans[0].physical_permutation, plans[2].physical_permutation)


def test_r05_pair_prefix_order_grows_one_exact_trie_branch_per_pair() -> None:
    contexts = torch.arange(RECORDS_PER_EXPERT).repeat_interleave(32)
    expected_records = (4, 19, 3, 20, 2, 21, 1, 22, 0, 23, *range(5, 19))
    plans = [
        plan_mixed_matrix(
            contexts,
            RATE_TRANSFER_MODES[r],
            matrix="w2",
            layout="r05_pair_prefix_reuse",
        )
        for r in range(6)
    ]
    for plan in plans:
        group_indices = plan.encoder_permutation[::128] // CONTEXT_GROUP_CHANNELS
        records = contexts.index_select(0, group_indices)
        assert tuple(records.tolist()) == expected_records
        assert torch.equal(
            plan.encoder_permutation, plans[0].encoder_permutation
        )


def test_mixed_matrix_plan_rejects_unbalanced_contexts() -> None:
    contexts = _scrambled_record_contexts()
    contexts[0] = 1

    with pytest.raises(ValueError, match="equal population"):
        plan_mixed_matrix(contexts, RATE_TRANSFER_MODES[3], matrix="w2")


def test_mixed_quant_args_select_exactly_one_procedural_codebook() -> None:
    plan = plan_mixed_matrix(
        _scrambled_record_contexts(), RATE_TRANSFER_MODES[1], matrix="w2"
    )
    mcg = _mixed_quant_args(
        plan,
        matrix="w2",
        layer=1,
        device=torch.device("cuda", 0),
        shared_scale_scope=None,
        codebook="mcg",
    )
    e4m3 = _mixed_quant_args(
        plan,
        matrix="w2",
        layer=1,
        device=torch.device("cuda", 0),
        shared_scale_scope=None,
        codebook="mul1-e4m3",
    )
    assert mcg["mcg"] is True and "mul1_e4m3" not in mcg
    assert e4m3["mul1_e4m3"] is True and "mcg" not in e4m3

    with pytest.raises(ValueError, match="unsupported EXL3 codebook"):
        _mixed_quant_args(
            plan,
            matrix="w2",
            layer=1,
            device=torch.device("cuda", 0),
            shared_scale_scope=None,
            codebook="unknown",
        )
