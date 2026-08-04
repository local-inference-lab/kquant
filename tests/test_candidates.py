from __future__ import annotations

import numpy as np
import pytest
import torch

from kquant import constants as C
from kquant.pack.candidates import _conditional_damage, choose_allocation
from kquant.pack.exl3 import make_shared_h


def test_shared_h_keeps_untouched_error_hessian() -> None:
    source = torch.tensor([[2.0, 0.5], [0.5, 3.0]])
    h_data = make_shared_h(2, torch.device("cpu"), source)

    h_data["H"].zero_()

    torch.testing.assert_close(h_data["error_H"], source)


def test_conditional_damage_uses_matching_activation_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "NUM_EXPERTS", 2)
    monkeypatch.setattr(C, "LATENT", 3)
    monkeypatch.setattr(C, "EXPERT_INTER", 2)
    w1 = np.array([[1, 2, 3], [2, 0, 1]], dtype=np.float64)
    w3 = np.array([[4, 5, 6], [1, 3, 2]], dtype=np.float64)
    w2 = np.array([[7, 8], [4, 5]], dtype=np.float64)
    act_in = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float64)
    act_mid = np.array([[8, 9], [10, 11]], dtype=np.float64)

    actual = _conditional_damage(w1, w3, w2, act_in, act_mid)
    expected = ((w1 + w3) * act_in).sum(axis=1) + (w2 * act_mid).sum(axis=1)
    np.testing.assert_allclose(actual, expected)


def test_choose_allocation_keeps_exact_highest_damage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "NUM_MOE_LAYERS", 2)
    monkeypatch.setattr(C, "NUM_EXPERTS", 3)
    monkeypatch.setattr(C, "MOE_LAYERS", (1, 2))
    scores = np.array([[1.0, 6.0, 2.0], [5.0, 3.0, 4.0]])

    allocation = choose_allocation(scores, keep_frac=0.5)

    assert allocation == {
        "1": {"keep": [1], "exl3": [0, 2]},
        "2": {"keep": [0, 2], "exl3": [1]},
    }


def test_choose_allocation_rejects_nonfinite_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "NUM_MOE_LAYERS", 1)
    monkeypatch.setattr(C, "NUM_EXPERTS", 2)
    monkeypatch.setattr(C, "MOE_LAYERS", (1,))
    with pytest.raises(ValueError, match="finite non-negative"):
        choose_allocation(np.array([[1.0, np.nan]]), keep_frac=0.5)
