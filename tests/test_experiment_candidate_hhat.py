import argparse

import pytest
import torch

from scripts.experiment_candidate_hhat import (
    _covariance_variants,
    _parse_alphas,
    _variant_name,
    _weighted_middle_nmse,
)


def test_alpha_parser_and_variant_names_are_stable():
    assert _parse_alphas("0.125,0.5,1") == (0.125, 0.5, 1.0)
    assert _variant_name("candidate_hhat", 0.125) == "candidate_hhat_a0p125"
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_alphas("0,0.5")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_alphas("0.5,0.5")


def test_covariance_variants_include_one_global_and_paired_shrinkage():
    global_h = torch.eye(2)
    source_h = 2 * torch.eye(2)
    candidate_h = 3 * torch.eye(2)

    variants = list(
        _covariance_variants(global_h, source_h, candidate_h, (0.25, 1.0))
    )

    assert [variant[0] for variant in variants] == [
        "global_teacher",
        "source_expert_a0p25",
        "candidate_hhat_a0p25",
        "source_expert_a1",
        "candidate_hhat_a1",
    ]
    torch.testing.assert_close(variants[1][3], 1.25 * torch.eye(2))
    torch.testing.assert_close(variants[2][3], 1.5 * torch.eye(2))
    torch.testing.assert_close(variants[-1][3], candidate_h)


def test_weighted_middle_nmse_uses_gate_square_weights():
    reference = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    candidate = torch.zeros_like(reference)
    gates = torch.tensor([0.5, 1.0])

    assert _weighted_middle_nmse(reference, candidate, gates) == pytest.approx(1.0)
