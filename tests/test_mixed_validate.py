from __future__ import annotations

import pytest

from kquant import constants as C
from kquant.mixed_exl3 import (
    EXPERT_RANK_MXFP4_BYTES,
    EXPERT_TRELLIS_BYTES,
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    SCALE_BYTES,
    TP_SIZE,
    ExpertFormatSpec,
    TP12LayerLayout,
)
from kquant.pack.mixed_materialize import LayerMaterializationSpec
from kquant.pack.mixed_validate import (
    _validate_layer_manifest_entry,
    expected_logical_source_bytes,
)


def _spec() -> LayerMaterializationSpec:
    kept = (7,)
    compressed = tuple(expert for expert in range(C.NUM_EXPERTS) if expert != 7)
    formats = tuple(
        ExpertFormatSpec.mxfp4()
        if expert == 7
        else ExpertFormatSpec.compressed(0)
        for expert in range(C.NUM_EXPERTS)
    )
    return LayerMaterializationSpec(
        layer=1,
        formats=formats,
        compressed=compressed,
        kept=kept,
        layout=TP12LayerLayout(len(compressed), len(kept)),
    )


def test_logical_source_byte_coverage_counts_both_tiers() -> None:
    spec = _spec()
    compressed_expert_bytes = (
        EXPERT_TRELLIS_BYTES
        + 3 * (LATENT_CHANNELS + INTERMEDIATE_CHANNELS) * SCALE_BYTES
    )
    expected = (
        len(spec.compressed) * compressed_expert_bytes
        + TP_SIZE * EXPERT_RANK_MXFP4_BYTES
    )

    assert expected_logical_source_bytes(spec) == expected


def test_layer_manifest_entry_requires_full_bit_exact_closure() -> None:
    spec = _spec()
    structural = {
        "file": "mixed-exl3-tp12-layer-00001.bin",
        "disk_bytes": spec.layout.disk_bytes,
        "compressed_experts": len(spec.compressed),
        "kept_experts": len(spec.kept),
    }
    entry = {
        **structural,
        "payload_closure": "bit_exact",
        "compressed_experts_verified": len(spec.compressed),
        "kept_experts_verified": len(spec.kept),
        "logical_source_bytes_verified": expected_logical_source_bytes(spec),
        "layout": spec.layout.to_manifest(),
    }

    _validate_layer_manifest_entry(entry, spec, structural)

    entry["payload_closure"] = "unchecked"
    with pytest.raises(ValueError, match="payload_closure"):
        _validate_layer_manifest_entry(entry, spec, structural)
