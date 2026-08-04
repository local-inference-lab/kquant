import itertools

import pytest

from kquant.mixed_exl3 import RECORDS_PER_EXPERT
from kquant.rate_transfer_search import (
    RecordRateAssignment,
    order_assignment_for_ldlq,
    top_rate_transfer_assignments,
)


def _brute_force(donor_deltas, recipient_deltas, r):
    candidates = []
    records = range(RECORDS_PER_EXPERT)
    for donors in itertools.combinations(records, r):
        remaining = tuple(record for record in records if record not in donors)
        for recipients in itertools.combinations(remaining, r):
            candidates.append(
                RecordRateAssignment(
                    donors,
                    recipients,
                    sum(donor_deltas[record] for record in donors)
                    + sum(recipient_deltas[record] for record in recipients),
                )
            )
    return sorted(
        candidates,
        key=lambda item: (
            item.additive_delta,
            item.donor_records,
            item.recipient_records,
        ),
    )


def test_top_rate_transfer_assignments_matches_brute_force_top_k():
    # Make only the first six records competitive, while preserving the real
    # 24-record format contract.  Brute force at r=1 is still compact enough
    # to be a transparent exact oracle.
    donor = [float(record + 20) for record in range(RECORDS_PER_EXPERT)]
    recipient = [float(record + 20) for record in range(RECORDS_PER_EXPERT)]
    donor[:6] = [3.0, 0.5, 2.0, 4.0, 1.0, 5.0]
    recipient[:6] = [-1.0, -6.0, -2.0, -5.0, -3.0, -4.0]

    expected = _brute_force(donor, recipient, 1)[:12]
    actual = top_rate_transfer_assignments(donor, recipient, r=1, limit=12)

    assert actual == tuple(expected)
    assert all(
        set(item.donor_records).isdisjoint(item.recipient_records)
        for item in actual
    )


def test_zero_transfer_has_one_empty_assignment():
    zeros = [0.0] * RECORDS_PER_EXPERT
    assert top_rate_transfer_assignments(zeros, zeros, r=0, limit=4) == (
        RecordRateAssignment((), (), 0.0),
    )


def test_ldlq_order_places_low_cost_donor_low_and_high_gain_recipient_high():
    donor = [10.0] * RECORDS_PER_EXPERT
    recipient = [0.0] * RECORDS_PER_EXPERT
    donor[2], donor[7], donor[9] = 2.0, 1.0, 3.0
    recipient[4], recipient[8], recipient[12] = -2.0, -5.0, -3.0
    assignment = RecordRateAssignment((2, 7, 9), (4, 8, 12), -4.0)

    donor_order, recipient_pair_order = order_assignment_for_ldlq(
        assignment, donor, recipient
    )

    assert donor_order == (7, 2, 9)
    assert recipient_pair_order == (8, 12, 4)


@pytest.mark.parametrize(
    "donor,recipient,r,limit,error",
    [
        ([0.0] * 23, [0.0] * 24, 1, 1, "exactly 24"),
        ([0.0] * 24, [0.0] * 23, 1, 1, "exactly 24"),
        ([0.0] * 23 + [float("nan")], [0.0] * 24, 1, 1, "finite"),
        ([0.0] * 24, [0.0] * 24, 13, 1, "r must"),
        ([0.0] * 24, [0.0] * 24, 1, 0, "limit"),
    ],
)
def test_search_rejects_malformed_inputs(donor, recipient, r, limit, error):
    with pytest.raises(ValueError, match=error):
        top_rate_transfer_assignments(
            donor, recipient, r=r, limit=limit
        )
