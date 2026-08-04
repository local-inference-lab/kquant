from __future__ import annotations

import numpy as np
import pytest

from kquant import constants as C
from kquant.mixed_exl3 import TP12LayerLayout
from kquant.pack.mixed_allocation import RAW_KEEP_PROMOTION_BYTES
from scripts.summarize_mixed_exl3_validation import (
    _average_ranks,
    _spearman,
    bootstrap_keep_stability,
    compare_damage_rankings,
)


def test_average_ranks_and_spearman_handle_ties() -> None:
    ranks = _average_ranks(np.array([3.0, 1.0, 1.0, 2.0]))
    assert ranks.tolist() == [3.0, 0.5, 0.5, 2.0]
    correlation = _spearman(
        np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])
    )
    assert correlation == pytest.approx(-1.0)


def test_ranking_comparison_reports_held_out_allocation_regret() -> None:
    selection = np.zeros((C.NUM_MOE_LAYERS, C.NUM_EXPERTS), dtype=np.float64)
    validation = np.zeros_like(selection)
    selection[0, 0] = 10.0
    selection[0, 1] = 9.0
    validation[0, 0] = 1.0
    validation[0, 1] = 11.0
    all_compressed = C.NUM_MOE_LAYERS * TP12LayerLayout(
        C.NUM_EXPERTS, 0
    ).disk_bytes
    # Allow alignment slack for one promotion in the same layer.
    target = all_compressed + RAW_KEEP_PROMOTION_BYTES + 64 * 1024

    result = compare_damage_rankings(
        selection, validation, target_container_bytes=target
    )

    assert result["keep_set"]["selection_retained"] == 1
    assert result["keep_set"]["validation_retained"] == 1
    assert result["keep_set"]["intersection"] == 0
    assert result["validation_ranking_gain_over_selection_ranking"][
        "additional_validation_damage_avoided"
    ] == pytest.approx(10.0)


def test_document_bootstrap_reports_stable_exact_budget_keep() -> None:
    damage = np.zeros(
        (C.NUM_MOE_LAYERS, C.NUM_EXPERTS, 3),
        dtype=np.float64,
    )
    damage[0, 0] = [10.0, 11.0, 9.0]
    damage[0, 1] = [1.0, 2.0, 1.5]
    all_compressed = C.NUM_MOE_LAYERS * TP12LayerLayout(
        C.NUM_EXPERTS, 0
    ).disk_bytes
    target = all_compressed + RAW_KEEP_PROMOTION_BYTES + 64 * 1024

    result = bootstrap_keep_stability(
        damage,
        target_container_bytes=target,
        replicates=20,
        seed=7,
    )

    assert result["base_retained_experts"] == 1
    assert result["exact_base_keep_set_frequency"] == 1.0
    assert result["uncertified_alignment_repair_frequency"] == 0.0
    assert result["ambiguous_assignments_probability_10_to_90"] == 0
    assert result["replicate_keep_set_jaccard_quantiles"]["p00"] == 1.0
