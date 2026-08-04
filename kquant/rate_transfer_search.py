"""Offline proposal search for equal-count K2/K4 record exchanges.

The serving format needs only an ``R_r`` mode and one common intermediate
permutation.  This module searches the *sets* of 128-channel records that
should donate and receive rate under an additive proposal model.  Full dense-H
LDLQ remains authoritative: proposal scores only decide which complete
counterfactual encodes are worth evaluating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from kquant.mixed_exl3 import PAIRS_PER_EXPERT, RECORDS_PER_EXPERT


@dataclass(frozen=True)
class RecordRateAssignment:
    """One equal-count donor/recipient assignment under an additive proxy."""

    donor_records: tuple[int, ...]
    recipient_records: tuple[int, ...]
    additive_delta: float


def _validate_deltas(values: Sequence[float], name: str) -> tuple[float, ...]:
    if len(values) != RECORDS_PER_EXPERT:
        raise ValueError(
            f"{name} must contain exactly {RECORDS_PER_EXPERT} record deltas"
        )
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def top_rate_transfer_assignments(
    donor_deltas: Sequence[float],
    recipient_deltas: Sequence[float],
    *,
    r: int,
    limit: int,
) -> tuple[RecordRateAssignment, ...]:
    """Return the exact top assignments under a separable proposal model.

    ``donor_deltas[j]`` is the proposed cost of changing record ``j`` from K3
    to K2.  ``recipient_deltas[j]`` is the proposed signed distortion change
    from K3 to K4 (normally negative).  Every result selects exactly ``r``
    disjoint donors and recipients.  Dynamic programming retains the best
    ``limit`` partial paths per count state; this is exact for the top
    ``limit`` final paths because future transitions depend only on the two
    counts, not on the path used to reach them.

    These assignments are proposals, not distortion claims.  Dense-H LDLQ is
    stateful and nonseparable, so a complete encode must validate each result.
    """

    donors = _validate_deltas(donor_deltas, "donor_deltas")
    recipients = _validate_deltas(recipient_deltas, "recipient_deltas")
    if isinstance(r, bool) or not isinstance(r, int) or not 0 <= r <= PAIRS_PER_EXPERT:
        raise ValueError(f"r must be an integer in 0..{PAIRS_PER_EXPERT}")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    # Values are (cost, donor tuple, recipient tuple).  Tuples are naturally
    # sorted because records are considered in ascending order, which also
    # makes tie-breaking deterministic.
    states: dict[
        tuple[int, int], list[tuple[float, tuple[int, ...], tuple[int, ...]]]
    ] = {(0, 0): [(0.0, (), ())]}
    for record in range(RECORDS_PER_EXPERT):
        next_states: dict[
            tuple[int, int], list[tuple[float, tuple[int, ...], tuple[int, ...]]]
        ] = {}
        for (donor_count, recipient_count), paths in states.items():
            for cost, donor_records, recipient_records in paths:
                next_states.setdefault((donor_count, recipient_count), []).append(
                    (cost, donor_records, recipient_records)
                )
                if donor_count < r:
                    next_states.setdefault(
                        (donor_count + 1, recipient_count), []
                    ).append(
                        (
                            cost + donors[record],
                            donor_records + (record,),
                            recipient_records,
                        )
                    )
                if recipient_count < r:
                    next_states.setdefault(
                        (donor_count, recipient_count + 1), []
                    ).append(
                        (
                            cost + recipients[record],
                            donor_records,
                            recipient_records + (record,),
                        )
                    )
        states = {
            state: sorted(paths, key=lambda path: (path[0], path[1], path[2]))[
                :limit
            ]
            for state, paths in next_states.items()
            if state[0] <= r and state[1] <= r
        }

    return tuple(
        RecordRateAssignment(donor_records, recipient_records, cost)
        for cost, donor_records, recipient_records in states[(r, r)]
    )


def order_assignment_for_ldlq(
    assignment: RecordRateAssignment,
    donor_deltas: Sequence[float],
    recipient_deltas: Sequence[float],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Order selected sets from low to high proposed sensitivity.

    Donors are returned in increasing K2 cost, so the cheapest donor occupies
    the lowest logical position.  Recipients are returned in increasing
    signed K4 delta (largest expected gain first).  The latter tuple is the
    *pair order* accepted by :func:`rate_transfer_record_order`; that helper
    reverses it in the logical suffix, placing the largest-gain recipient at
    the highest position and therefore first in EXL's backward LDLQ traversal.
    """

    donors = _validate_deltas(donor_deltas, "donor_deltas")
    recipients = _validate_deltas(recipient_deltas, "recipient_deltas")
    donor_order = tuple(
        sorted(assignment.donor_records, key=lambda record: (donors[record], record))
    )
    recipient_order = tuple(
        sorted(
            assignment.recipient_records,
            key=lambda record: (recipients[record], -record),
        )
    )
    return donor_order, recipient_order
