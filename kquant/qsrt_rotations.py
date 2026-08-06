"""Validated transform-draw plans for QSRT candidate construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kquant import constants as C


ROTATION_PLAN_KIND = "kquant_qsrt_rotation_plan"
ROTATION_PLAN_SCHEMA_VERSION = 1


def _draw(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class QSRTLayerRotationPlan:
    """One layer-shared residual draw and optional private expert draws."""

    residual_draw: int = 0
    intermediate_default: int = 0
    intermediate_overrides: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        _draw(self.residual_draw, field="residual_draw")
        _draw(self.intermediate_default, field="intermediate_default")
        experts = tuple(expert for expert, _draw_id in self.intermediate_overrides)
        if experts != tuple(sorted(set(experts))):
            raise ValueError("intermediate overrides must have sorted unique experts")
        for expert, draw_id in self.intermediate_overrides:
            if (
                isinstance(expert, bool)
                or not isinstance(expert, int)
                or not 0 <= expert < C.NUM_EXPERTS
            ):
                raise ValueError("rotation-plan experts must lie in 0..895")
            _draw(draw_id, field=f"intermediate override for expert {expert}")

    def intermediate_draw(self, expert: int) -> int:
        if (
            isinstance(expert, bool)
            or not isinstance(expert, int)
            or not 0 <= expert < C.NUM_EXPERTS
        ):
            raise ValueError("expert must lie in 0..895")
        return dict(self.intermediate_overrides).get(
            expert, self.intermediate_default
        )

    def to_json(self) -> dict[str, object]:
        return {
            "residual_draw": self.residual_draw,
            "intermediate_default": self.intermediate_default,
            "intermediate_overrides": {
                str(expert): draw_id
                for expert, draw_id in self.intermediate_overrides
            },
        }

    @classmethod
    def from_json(cls, value: object) -> "QSRTLayerRotationPlan":
        if not isinstance(value, dict):
            raise ValueError("each rotation-plan layer must be an object")
        unknown = set(value) - {
            "residual_draw",
            "intermediate_default",
            "intermediate_overrides",
        }
        if unknown:
            raise ValueError(f"unknown layer rotation-plan fields: {sorted(unknown)}")
        overrides_raw = value.get("intermediate_overrides", {})
        if not isinstance(overrides_raw, dict):
            raise ValueError("intermediate_overrides must be an object")
        overrides: list[tuple[int, int]] = []
        for expert_text, draw_id_raw in overrides_raw.items():
            if not isinstance(expert_text, str) or not expert_text.isdecimal():
                raise ValueError("rotation-plan expert keys must be decimal strings")
            expert = int(expert_text)
            overrides.append(
                (
                    expert,
                    _draw(
                        draw_id_raw,
                        field=f"intermediate override for expert {expert}",
                    ),
                )
            )
        return cls(
            residual_draw=_draw(
                value.get("residual_draw", 0), field="residual_draw"
            ),
            intermediate_default=_draw(
                value.get("intermediate_default", 0),
                field="intermediate_default",
            ),
            intermediate_overrides=tuple(sorted(overrides)),
        )


@dataclass(frozen=True)
class QSRTRotationPlan:
    """Sparse model-wide plan; omitted layers use production draw zero."""

    layers: tuple[tuple[int, QSRTLayerRotationPlan], ...] = ()

    def __post_init__(self) -> None:
        layer_ids = tuple(layer for layer, _plan in self.layers)
        if layer_ids != tuple(sorted(set(layer_ids))):
            raise ValueError("rotation-plan layers must be sorted and unique")
        if any(layer not in C.MOE_LAYERS for layer in layer_ids):
            raise ValueError("rotation-plan layers must lie in 1..92")

    def for_layer(self, layer: int) -> QSRTLayerRotationPlan:
        if layer not in C.MOE_LAYERS:
            raise ValueError("Kimi-K3 MoE layer must lie in 1..92")
        return dict(self.layers).get(layer, QSRTLayerRotationPlan())

    def to_json(self) -> dict[str, object]:
        return {
            "kind": ROTATION_PLAN_KIND,
            "schema_version": ROTATION_PLAN_SCHEMA_VERSION,
            "layers": {
                str(layer): plan.to_json() for layer, plan in self.layers
            },
        }

    @classmethod
    def from_json(cls, value: object) -> "QSRTRotationPlan":
        if not isinstance(value, dict):
            raise ValueError("rotation plan must be an object")
        if value.get("kind") != ROTATION_PLAN_KIND:
            raise ValueError(f"rotation plan kind must be {ROTATION_PLAN_KIND!r}")
        if value.get("schema_version") != ROTATION_PLAN_SCHEMA_VERSION:
            raise ValueError(
                "unsupported QSRT rotation-plan schema version"
            )
        unknown = set(value) - {"kind", "schema_version", "layers"}
        if unknown:
            raise ValueError(f"unknown rotation-plan fields: {sorted(unknown)}")
        layers_raw = value.get("layers")
        if not isinstance(layers_raw, dict):
            raise ValueError("rotation-plan layers must be an object")
        layers: list[tuple[int, QSRTLayerRotationPlan]] = []
        for layer_text, plan_raw in layers_raw.items():
            if not isinstance(layer_text, str) or not layer_text.isdecimal():
                raise ValueError("rotation-plan layer keys must be decimal strings")
            layers.append(
                (int(layer_text), QSRTLayerRotationPlan.from_json(plan_raw))
            )
        return cls(tuple(sorted(layers)))


def load_qsrt_rotation_plan(path: Path) -> QSRTRotationPlan:
    """Load, validate, and canonicalize a rotation plan."""

    return QSRTRotationPlan.from_json(json.loads(path.read_text()))
