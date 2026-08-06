import pytest
import torch

from kquant.candidate_hessian import (
    adaptive_identity_shrinkage,
    covariance_comparison,
    partition_documents,
    request_mask,
    shrink_covariance,
    subset_request_documents,
    weighted_covariance,
    weighted_effective_sample_size,
)


def test_weighted_covariance_matches_direct_definition():
    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 2.0]])
    weights = torch.tensor([1.0, 2.0, 3.0])

    actual, denominator = weighted_covariance(
        rows, weights, device=torch.device("cpu"), chunk_rows=2
    )
    expected = rows.T @ (rows * weights[:, None]) / weights.sum()

    assert denominator == 6.0
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, actual.T)


def test_covariance_shrinkage_endpoints_and_diagnostics():
    global_h = torch.eye(3)
    local_h = torch.diag(torch.tensor([2.0, 3.0, 4.0]))

    torch.testing.assert_close(shrink_covariance(global_h, local_h, 0), global_h)
    torch.testing.assert_close(shrink_covariance(global_h, local_h, 1), local_h)
    torch.testing.assert_close(
        shrink_covariance(global_h, local_h, 0.25),
        0.75 * global_h + 0.25 * local_h,
    )
    diagnostics = covariance_comparison(global_h, local_h)
    assert diagnostics["trace_ratio"] == 3.0
    assert diagnostics["relative_frobenius_difference"] > 0
    assert diagnostics["candidate_symmetry_max_abs"] == 0


@pytest.mark.parametrize(
    "rows,weights,error",
    [
        (torch.empty(0, 2), torch.empty(0), "nonempty"),
        (torch.ones(2, 2), torch.ones(3), "shapes"),
        (torch.ones(2, 2), torch.tensor([1.0, -1.0]), "nonnegative"),
        (torch.ones(2, 2), torch.zeros(2), "positive"),
    ],
)
def test_weighted_covariance_rejects_bad_samples(rows, weights, error):
    with pytest.raises(ValueError, match=error):
        weighted_covariance(rows, weights, device=torch.device("cpu"))


def test_shrink_covariance_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        shrink_covariance(torch.eye(2), torch.eye(2), 1.1)


def test_weighted_effective_sample_size_tracks_weight_concentration():
    assert weighted_effective_sample_size(torch.ones(8)) == pytest.approx(8.0)
    concentrated = weighted_effective_sample_size(
        torch.tensor([10.0, 1.0, 1.0, 1.0])
    )
    assert 1.0 < concentrated < 4.0


def test_adaptive_identity_shrinkage_trace_matches_and_respects_cap():
    rows = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 2.0], [-1.0, 1.0]]
    )
    weights = torch.ones(4)
    local, _ = weighted_covariance(
        rows, weights, device=torch.device("cpu"), return_cpu=False
    )
    actual, evidence = adaptive_identity_shrinkage(
        local, weights, max_local_alpha=0.6
    )

    assert evidence["effective_sample_size"] == pytest.approx(4.0)
    assert 0.0 <= evidence["local_alpha"] <= 0.6
    assert evidence["identity_scale"] == pytest.approx(
        float(torch.trace(local) / 2)
    )
    torch.testing.assert_close(torch.trace(actual), torch.trace(local))
    torch.testing.assert_close(actual, actual.T)


def test_adaptive_identity_shrinkage_has_no_external_covariance_prior():
    rows = torch.tensor(
        [[3.0, 0.0], [0.0, 1.0], [2.0, -1.0], [-2.0, 1.0]]
    )
    weights = torch.tensor([1.0, 2.0, 1.0, 0.5])
    local, _ = weighted_covariance(
        rows, weights, device=torch.device("cpu"), return_cpu=False
    )

    actual, evidence = adaptive_identity_shrinkage(local, weights)
    identity_scale = float(torch.trace(local) / local.shape[0])
    expected = torch.lerp(
        torch.eye(2) * identity_scale,
        local,
        evidence["local_alpha"],
    )
    torch.testing.assert_close(actual, expected)


def test_document_partition_and_request_mask_are_exact():
    documents = [f"document-{index}" for index in range(16)]
    fit, confirmation = partition_documents(
        documents, modulus=4, confirmation_index=0
    )
    assert set(fit).isdisjoint(confirmation)
    assert set(fit) | set(confirmation) == set(documents)
    assert partition_documents(
        documents, modulus=4, confirmation_index=0
    ) == (fit, confirmation)

    request_documents = {index + 1: document for index, document in enumerate(documents)}
    fit_requests = subset_request_documents(request_documents, fit)
    observations = torch.tensor(
        [(1 << 32) | 7, (2 << 32) | 9, (99 << 32) | 1], dtype=torch.int64
    )
    expected = torch.tensor([1 in fit_requests, 2 in fit_requests, False])
    torch.testing.assert_close(request_mask(observations, fit_requests), expected)


def test_document_helpers_reject_incomplete_inputs():
    with pytest.raises(ValueError):
        partition_documents(["same", "same"], modulus=4, confirmation_index=0)
    with pytest.raises(ValueError):
        subset_request_documents({1: "present"}, ["missing"])
    with pytest.raises(ValueError):
        request_mask(torch.zeros((1, 1), dtype=torch.int64), {1: "doc"})
