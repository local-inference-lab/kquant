from scripts.summarize_candidate_hhat import (
    _global_split_argmin,
    _global_training_argmin,
    _training_argmin,
)


def _variant(name, train_sse, validation_sse):
    def metric(value):
        return {
            "route_weighted_squared_error": value,
            "route_weighted_reference_energy": 10.0,
            "request_contributions": [
                {
                    "document_hash": "doc-a",
                    "route_weighted_squared_error": value,
                    "route_weighted_reference_energy": 10.0,
                    "teacher_mixture_energy": 20.0,
                }
            ],
        }

    return {
        "name": name,
        "coupled_activation": {
            "train": metric(train_sse),
            "validation": metric(validation_sse),
        },
    }


def _payload(variants):
    return {"modes": [{"variants": variants}]}


def test_training_argmin_does_not_use_validation():
    payload = _payload(
        [
            _variant("global_teacher", 10.0, 1.0),
            _variant("candidate_hhat_a0p25", 8.0, 50.0),
            _variant("candidate_hhat_a0p5", 9.0, 0.1),
        ]
    )

    assert _training_argmin(
        payload,
        ["global_teacher", "candidate_hhat_a0p25", "candidate_hhat_a0p5"],
    ) == "candidate_hhat_a0p25"


def test_global_training_argmin_sums_experts_before_selecting():
    records = {
        (1, 1): _payload(
            [
                _variant("global_teacher", 10.0, 1.0),
                _variant("candidate_hhat_a0p25", 7.0, 99.0),
                _variant("candidate_hhat_a0p5", 8.0, 0.1),
            ]
        ),
        (1, 2): _payload(
            [
                _variant("global_teacher", 20.0, 1.0),
                _variant("candidate_hhat_a0p25", 19.0, 99.0),
                _variant("candidate_hhat_a0p5", 15.0, 0.1),
            ]
        ),
    }

    assert _global_training_argmin(
        records,
        ["global_teacher", "candidate_hhat_a0p25", "candidate_hhat_a0p5"],
    ) == "candidate_hhat_a0p5"


def test_global_split_argmin_can_select_on_confirmation():
    payload = _payload(
        [
            _variant("global_teacher", 1.0, 1.0),
            _variant("candidate_hhat_a0p5", 2.0, 0.5),
        ]
    )
    for variant in payload["modes"][0]["variants"]:
        variant["coupled_activation"]["confirmation"] = variant[
            "coupled_activation"
        ]["validation"]

    assert _global_split_argmin(
        {(1, 1): payload},
        ["global_teacher", "candidate_hhat_a0p5"],
        "confirmation",
    ) == "candidate_hhat_a0p5"
