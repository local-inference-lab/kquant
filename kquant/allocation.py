"""allocation.json (de)serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kquant import constants as C
from kquant.rank.solver import Allocation

SCHEMA_VERSION = 1


def save_allocation(
    alloc: Allocation,
    path: str | Path,
    budget_info: dict,
    ranker_info: dict,
    solver_info: dict,
) -> Path:
    path = Path(path)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "model": C.MODEL_ID,
        "revision": C.REVISION,
        "formats": alloc.formats,
        "budget": budget_info,
        "ranker": ranker_info,
        "solver": solver_info,
        "achieved": {
            "total_bytes": alloc.total_bytes,
            "bpw": alloc.achieved_bpw,
            "solved_bytes": alloc.solved_bytes,
            "solved_bpw": alloc.solved_bpw,
            "total_damage": alloc.total_damage,
            "counts": alloc.counts(),
            "keep_per_layer": alloc.keep_per_layer,
        },
        "layers_solved": alloc.layers_solved,
        "assignment": alloc.assignment.tolist(),
    }
    path.write_text(json.dumps(doc, indent=1))
    return path


def load_allocation(path: str | Path) -> tuple[Allocation, dict]:
    doc = json.loads(Path(path).read_text())
    if doc["schema_version"] > SCHEMA_VERSION:
        raise ValueError("allocation schema newer than this kquant")
    alloc = Allocation(
        formats=doc["formats"],
        assignment=np.array(doc["assignment"], dtype=np.int8),
        layers_solved=doc["layers_solved"],
        total_bytes=doc["achieved"]["total_bytes"],
        achieved_bpw=doc["achieved"]["bpw"],
        solved_bytes=doc["achieved"].get("solved_bytes", 0),
        solved_bpw=doc["achieved"].get("solved_bpw", 0.0),
        total_damage=doc["achieved"]["total_damage"],
        keep_per_layer={int(k): v for k, v in doc["achieved"]["keep_per_layer"].items()},
    )
    return alloc, doc
