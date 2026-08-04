import torch

from scripts.select_w2_source_sample import _stratified_sample


def test_stratified_sample_is_deterministic_and_balanced():
    eligible = list(range(32))
    traffic = torch.arange(896, dtype=torch.float64)
    first, strata = _stratified_sample(
        eligible,
        traffic,
        layer=24,
        sample_size=16,
        strata=4,
        seed=123,
    )
    second, second_strata = _stratified_sample(
        eligible,
        traffic,
        layer=24,
        sample_size=16,
        strata=4,
        seed=123,
    )
    assert first == second
    assert strata == second_strata
    assert len(first) == len(set(first)) == 16
    assert [list(strata.values()).count(index) for index in range(4)] == [4] * 4
