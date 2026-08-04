from scripts.summarize_w2_record_map import _select_candidates


def _candidate(r, train_sse, validation_sse):
    def metric(value):
        return {
            "weighted_squared_error": value,
            "weighted_reference_energy": 10.0,
        }

    return {
        "mode": f"R{r}",
        "r": r,
        "captured_activation": {
            "train": metric(train_sse),
            "validation": metric(validation_sse),
        },
    }


def test_candidate_selection_never_uses_validation_error():
    canonical = _candidate(0, 10.0, 10.0)
    monotone_r1 = _candidate(1, 8.0, 30.0)
    monotone_r2 = _candidate(2, 9.0, 1.0)
    map_r1 = _candidate(1, 7.0, 40.0)
    map_r2 = _candidate(2, 8.0, 0.5)
    matched_r1 = _candidate(0, 9.5, 11.0)
    matched_r2 = _candidate(0, 9.2, 8.0)
    payload = {
        "canonical_r0": canonical,
        "monotone": [monotone_r1, monotone_r2],
        "maps": [
            {
                "proposal_rank": 2,
                "mixed": map_r1,
                "matched_order_r0": matched_r1,
            },
            {
                "proposal_rank": 1,
                "mixed": map_r2,
                "matched_order_r0": matched_r2,
            },
        ],
    }

    selected = _select_candidates(payload)

    assert selected["monotone"] is monotone_r1
    assert selected["map_mixed"] is map_r1
    assert selected["map_matched_r0"] is matched_r1
    assert selected["map_entry"] is payload["maps"][0]
    assert selected["map_kind"] == "arbitrary"


def test_canonical_remains_available_to_both_selectors():
    canonical = _candidate(0, 1.0, 1.0)
    payload = {
        "canonical_r0": canonical,
        "monotone": [_candidate(1, 2.0, 0.1)],
        "maps": [
            {
                "proposal_rank": 1,
                "mixed": _candidate(1, 3.0, 0.01),
                "matched_order_r0": _candidate(0, 1.5, 0.5),
            }
        ],
    }

    selected = _select_candidates(payload)

    assert selected["monotone"] is canonical
    assert selected["map_mixed"] is canonical
    assert selected["map_matched_r0"] is canonical
    assert selected["map_entry"] is None
    assert selected["map_kind"] == "canonical"


def test_record_map_family_falls_back_to_best_monotone_candidate():
    canonical = _candidate(0, 10.0, 10.0)
    monotone = _candidate(2, 7.0, 8.0)
    payload = {
        "canonical_r0": canonical,
        "monotone": [monotone],
        "maps": [
            {
                "proposal_rank": 1,
                "mixed": _candidate(2, 8.0, 7.0),
                "matched_order_r0": _candidate(0, 9.0, 9.0),
            }
        ],
    }

    selected = _select_candidates(payload)

    assert selected["map_mixed"] is monotone
    assert selected["map_matched_r0"] is canonical
    assert selected["map_entry"] is None
    assert selected["map_kind"] == "monotone"
