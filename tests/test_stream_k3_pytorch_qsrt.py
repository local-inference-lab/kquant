from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.stream_k3_pytorch as stream
from kquant import constants as C
from kquant.qsrt import (
    FORMAT_SECTION_BYTES,
    LAYER_HEADER_BYTES,
    SHARED_SCALE_SECTION_BYTES,
    ExpertFormatSpec,
    QSRTTrellisDescriptor,
    matrix_rate_axis,
)
from kquant.pack.qsrt_materialize import (
    QSRT_ARTIFACT_KIND,
    QSRT_ARTIFACT_SCHEMA_VERSION,
    QSRT_MANIFEST_FILENAME,
)
from kquant.pack.qsrt_atoms import layer_filename
from kquant.source_weights import PackedMXFP4Matrix
from kquant.x4t import X4T_DATA_OFFSET, x4t_layer_path


def _artifact_tree(root: Path) -> None:
    manifest = {
        "kind": QSRT_ARTIFACT_KIND,
        "schema_version": QSRT_ARTIFACT_SCHEMA_VERSION,
        "complete": True,
        "codec": "QSRT",
        "trellis_codebook": "sqg-normal-e4m3",
        "layers": {
            str(layer): {
                "qsrt_atoms": layer_filename(layer),
                "x4t": x4t_layer_path(Path("."), layer).name,
            }
            for layer in C.MOE_LAYERS
        },
    }
    (root / QSRT_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    for layer in C.MOE_LAYERS:
        (root / layer_filename(layer)).touch()
        x4t_layer_path(root, layer).touch()


def test_qsrt_artifact_requires_complete_exact_inventory(
    tmp_path: Path,
) -> None:
    _artifact_tree(tmp_path)

    artifact = stream.QSRTArtifact.load(tmp_path)

    assert artifact.root == tmp_path.resolve()
    assert artifact.layer_path(17) == (tmp_path / layer_filename(17)).resolve()
    assert artifact.x4t_layer_path(17) == x4t_layer_path(tmp_path, 17).resolve()
    (tmp_path / layer_filename(92)).unlink()
    with pytest.raises(ValueError, match="atom layer inventory does not close"):
        stream.QSRTArtifact.load(tmp_path)


def test_qsrt_compressed_decode_uses_matrix_mode_and_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (C.EXPERT_INTER, C.LATENT)
    descriptor = QSRTTrellisDescriptor(
        mode_id=2,
        rate_axis=matrix_rate_axis("w1"),
        k_tiles=expected[1] // 16,
        n_tiles=expected[0] // 16,
    )
    parts = {
        "trellis": torch.zeros(descriptor.payload_words, dtype=torch.int16),
        "suh": torch.ones(expected[1], dtype=torch.float16),
        "svh": torch.ones(expected[0], dtype=torch.float16),
    }

    class Reader:
        header = SimpleNamespace(layer=7)
        format_specs = tuple(ExpertFormatSpec.compressed(2, 1) for _ in range(9))

        @staticmethod
        def read_compressed_matrix(
            expert_id: int, matrix: str
        ) -> dict[str, torch.Tensor]:
            assert expert_id == 8
            assert matrix == "w1"
            return parts

    observed: dict[str, object] = {}

    def fake_decode(packed, suh, svh, *, codebook):
        observed["descriptor"] = packed.descriptor
        observed["suh"] = suh
        observed["svh"] = svh
        observed["codebook"] = codebook
        value = torch.zeros(
            (expected[1], expected[0]), dtype=torch.float16
        )
        value[1, 2] = 7
        return value

    monkeypatch.setattr(stream, "decode_qsrt_exl3_weight", fake_decode)

    dense, bytes_read = stream._decode_qsrt_trellis_matrix(
        Reader(),
        expert_id=8,
        matrix="w1",
        expected=expected,
        device=torch.device("cpu"),
    )

    actual_descriptor = observed["descriptor"]
    assert isinstance(actual_descriptor, QSRTTrellisDescriptor)
    assert actual_descriptor.mode_id == 2
    assert actual_descriptor.rate_axis == "n"
    assert observed["codebook"] == "qsrt-e4m3"
    assert (actual_descriptor.k_tiles, actual_descriptor.n_tiles) == (224, 192)
    assert dense.dtype == torch.bfloat16
    assert tuple(dense.shape) == expected
    assert dense[2, 1] == 7
    assert bytes_read == parts["trellis"].nbytes + parts["svh"].nbytes


def test_qsrt_stager_classifies_tiers_and_releases_to_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix_shapes = {"w1": (3, 2), "w2": (2, 3), "w3": (3, 2)}

    class Reader:
        header = SimpleNamespace(layer=1)
        format_specs = (
            ExpertFormatSpec.compressed(1),
            ExpertFormatSpec.x4t(),
        )

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class X4TReader:
        layer = 1

        @staticmethod
        def read(expert_id: int, matrix: str) -> PackedMXFP4Matrix:
            assert expert_id == 1
            shape = matrix_shapes[matrix]
            return PackedMXFP4Matrix(
                packed=torch.ones(shape, dtype=torch.uint8),
                scale=torch.ones((1,), dtype=torch.uint8),
            )

        @staticmethod
        def record_bytes(expert_id: int, matrix: str) -> int:
            assert expert_id == 1
            return torch.empty(matrix_shapes[matrix], dtype=torch.uint8).nbytes + 1

    class Compressor:
        @staticmethod
        def decompress(payload, unused_scheme):
            return {
                "weight": torch.full(
                    payload["weight_packed"].shape,
                    9,
                    dtype=torch.bfloat16,
                )
            }

    class Expert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = torch.nn.Linear(
                2, 3, bias=False, device="meta", dtype=torch.bfloat16
            )
            self.w2 = torch.nn.Linear(
                3, 2, bias=False, device="meta", dtype=torch.bfloat16
            )
            self.w3 = torch.nn.Linear(
                2, 3, bias=False, device="meta", dtype=torch.bfloat16
            )

    reader = Reader()
    experts = torch.nn.ModuleList((Expert(), Expert()))
    stager = stream.QSRTStagedExperts(
        reader=reader,
        x4t_reader=X4TReader(),
        layer_idx=1,
        scheme=object(),
        compressor=Compressor,
        device=torch.device("cpu"),
    )

    def fake_compressed(*, expert_id: int, matrix: str, expected: tuple[int, ...]):
        assert expert_id == 0
        return torch.full(expected, 4, dtype=torch.bfloat16), 5

    monkeypatch.setattr(stager, "_reconstruct_compressed", fake_compressed)
    stager._stage(experts, torch.tensor([[1, 0, 1]], dtype=torch.int64))

    assert stager.stats.expert_ids == [0, 1]
    assert stager.stats.exl3_expert_ids == [0]
    assert stager.stats.kept_expert_ids == [1]
    assert bool(torch.all(experts[0].w1.weight == 4))
    assert bool(torch.all(experts[1].w2.weight == 9))
    expected_metadata = (
        LAYER_HEADER_BYTES
        + FORMAT_SECTION_BYTES
        + SHARED_SCALE_SECTION_BYTES
        + X4T_DATA_OFFSET
    )
    expected_kept = sum(
        torch.empty(shape, dtype=torch.uint8).nbytes + 1
        for shape in matrix_shapes.values()
    )
    assert stager.stats.packed_bytes == expected_metadata + 3 * 5 + expected_kept
    with pytest.raises(RuntimeError, match="staged more than once"):
        stager._stage(experts, torch.tensor([0]))

    stager.close()
    assert reader.closed
    for expert in experts:
        for matrix in stream.EXPERT_MATRICES:
            assert getattr(expert, matrix).weight.device.type == "meta"
