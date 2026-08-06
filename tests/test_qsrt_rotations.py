from __future__ import annotations

import json

import pytest

from kquant.qsrt_rotations import (
    QSRTRotationPlan,
    QSRTLayerRotationPlan,
    load_qsrt_rotation_plan,
)


def test_rotation_plan_defaults_to_draw_zero() -> None:
    plan = QSRTRotationPlan()

    assert plan.for_layer(24) == QSRTLayerRotationPlan()
    assert plan.for_layer(24).intermediate_draw(127) == 0


def test_rotation_plan_round_trips_sparse_private_overrides(tmp_path) -> None:
    expected = QSRTRotationPlan(
        (
            (
                12,
                QSRTLayerRotationPlan(
                    residual_draw=6,
                    intermediate_overrides=((102, 7), (326, 5)),
                ),
            ),
        )
    )
    path = tmp_path / "rotations.json"
    path.write_text(json.dumps(expected.to_json()))

    actual = load_qsrt_rotation_plan(path)

    assert actual == expected
    assert actual.for_layer(12).intermediate_draw(102) == 7
    assert actual.for_layer(12).intermediate_draw(160) == 0
    assert actual.for_layer(24) == QSRTLayerRotationPlan()


@pytest.mark.parametrize(
    "value, match",
    (
        ({"kind": "wrong", "schema_version": 1, "layers": {}}, "kind"),
        (
            {
                "kind": "kquant_qsrt_rotation_plan",
                "schema_version": 1,
                "layers": {"0": {}},
            },
            "1..92",
        ),
        (
            {
                "kind": "kquant_qsrt_rotation_plan",
                "schema_version": 1,
                "layers": {"12": {"intermediate_overrides": {"896": 1}}},
            },
            "0..895",
        ),
    ),
)
def test_rotation_plan_rejects_malformed_values(value, match) -> None:
    with pytest.raises(ValueError, match=match):
        QSRTRotationPlan.from_json(value)
