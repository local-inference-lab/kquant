import pytest

from kquant.kernel_audit import audit_kernel_path


def test_stock_w4a16_audit_requires_ordinary_packed_kernel() -> None:
    log = "\n".join(
        (
            "Using B12xExperts",
            "B12X MoE force-A16 enabled: using quant_mode=w4a16.",
            "target=some.module.W4A16FusedMoeKernel",
            "attrs={'weight_layout': 'packed'}",
            "repeat implementation=w4a16",
        )
    )

    report = audit_kernel_path(log, "stock-w4a16")

    assert report["pass"] is True


def test_stock_w4a16_audit_rejects_hybrid_trellis() -> None:
    log = "\n".join(
        (
            "Using B12xExperts",
            "B12X MoE force-A16 enabled: using quant_mode=w4a16.",
            "target=some.module.W4A16FusedMoeKernel",
            "attrs={'weight_layout': 'packed'}",
            "repeat implementation=w4a16",
            "quantization=nvfp4_nf3_hybrid",
            "weight_layout=trellis3_t256",
        )
    )

    report = audit_kernel_path(log, "stock-w4a16")

    assert report["pass"] is False
    assert set(report["unexpected_evidence"]) == {
        "hybrid_quantization",
        "exl3_trellis_layout",
    }


def test_hybrid_audit_requires_exl3_trellis_evidence() -> None:
    log = "\n".join(
        (
            "quantization=nvfp4_nf3_hybrid",
            "allocated w13_exl3_trellis",
            "target=some.module.W4A16FusedMoeKernel",
            "weight_layout=trellis3_t256",
            "repeat implementation=w4a16",
        )
    )

    assert audit_kernel_path(log, "hybrid-exl3")["pass"] is True


def test_qsrt_tp12_audit_requires_materialized_slab_reader() -> None:
    log = "\n".join(
        (
            "quantization=nvfp4_nf3_hybrid",
            "Loaded qsrt_tp12 layer 1 rank 0: 800 compressed, 96 MXFP4",
            "target=some.module.W4A16FusedMoeKernel",
            "repeat implementation=w4a16",
        )
    )

    assert audit_kernel_path(log, "qsrt-tp12")["pass"] is True


def test_kernel_audit_rejects_unknown_path() -> None:
    with pytest.raises(ValueError, match="unknown kernel path"):
        audit_kernel_path("", "other")
