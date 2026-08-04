from __future__ import annotations

import torch

from kquant.trellis_window import trellis_alias_metrics
from scripts.experiment_trellis_e4m3_qd_k5_search import (
    Parameters,
    _candidate,
    _parameter_grid,
)


def test_k5_qd_search_grid_is_unique_and_contains_prior_candidate() -> None:
    grid = _parameter_grid()
    assert len(grid) == 1280
    assert len({parameters.name for parameters in grid}) == len(grid)
    assert Parameters((), (13, 23), 3, 7, 1.25) in grid


def test_selected_k5_qd_shape_is_finite_dense_and_locally_diverse() -> None:
    codebook, metadata = _candidate(
        Parameters((15,), (13, 23), 5, 0, 1.25)
    )
    assert codebook.shape == (1 << 16,)
    assert bool(torch.isfinite(codebook).all())
    assert metadata["numeric_labels"] == 146
    aliases = trellis_alias_metrics(codebook, 5, 16)["outgoing"]
    assert aliases["mean_distinct_labels_per_state"] > 29.0
    assert aliases["min_distinct_labels_per_state"] >= 25

