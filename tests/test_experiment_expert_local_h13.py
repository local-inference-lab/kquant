import torch

from scripts.experiment_expert_local_h13 import _h13_variants


def test_h13_variants_keep_global_control_and_fixed_shrinkage_ladder():
    global_h = torch.eye(2)
    expert_h = 3 * torch.eye(2)

    variants = list(_h13_variants(global_h, expert_h, (0.25, 0.75, 1.0)))

    assert [name for name, _alpha, _hessian in variants] == [
        "global_teacher",
        "source_expert_a0p25",
        "source_expert_a0p75",
        "source_expert_a1",
    ]
    torch.testing.assert_close(variants[1][2], 1.5 * torch.eye(2))
    torch.testing.assert_close(variants[2][2], 2.5 * torch.eye(2))
    torch.testing.assert_close(variants[3][2], expert_h)
