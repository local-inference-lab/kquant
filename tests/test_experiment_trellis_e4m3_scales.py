import torch

from scripts.experiment_trellis_e4m3_scales import _fit_scales


def test_fit_scales_recovers_continuous_and_power_of_two_block_scale():
    source = torch.full((1, 32), 2.0)
    transitions = torch.zeros((1, 32), dtype=torch.int32)
    codebook = torch.ones(1)
    base = torch.ones(1)

    continuous, log_ratio = _fit_scales(
        source,
        transitions,
        codebook,
        base,
        exponent_min=None,
        exponent_max=None,
        exponent_step=1.0,
    )
    assert torch.equal(continuous, source)
    assert torch.equal(log_ratio, torch.ones((1, 1)))

    power_two, exponent = _fit_scales(
        source,
        transitions,
        codebook,
        base,
        exponent_min=-4,
        exponent_max=3,
        exponent_step=1.0,
    )
    assert torch.equal(power_two, source)
    assert torch.equal(exponent, torch.ones((1, 1)))


def test_fit_scales_honors_packed_delta_range():
    source = torch.full((1, 32), 16.0)
    transitions = torch.zeros((1, 32), dtype=torch.int32)
    codebook = torch.ones(1)
    scale, exponent = _fit_scales(
        source,
        transitions,
        codebook,
        torch.ones(1),
        exponent_min=-4,
        exponent_max=3,
        exponent_step=1.0,
    )
    assert torch.equal(scale, torch.full((1, 32), 8.0))
    assert torch.equal(exponent, torch.full((1, 1), 3.0))


def test_fit_scales_supports_fractional_log2_steps():
    desired = 2.0 ** (3.0 / 32.0)
    source = torch.full((1, 32), desired)
    scale, exponent = _fit_scales(
        source,
        torch.zeros((1, 32), dtype=torch.int32),
        torch.ones(1),
        torch.ones(1),
        exponent_min=-4,
        exponent_max=3,
        exponent_step=1.0 / 32.0,
    )
    assert torch.allclose(scale, source)
    assert torch.allclose(exponent, torch.full((1, 1), 3.0 / 32.0))
