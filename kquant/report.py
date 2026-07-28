"""Assemble a markdown report from whatever outputs exist in the out dir."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kquant import constants as C
from kquant.dynstats import load_bundle
from kquant.rank.budget import bytes_to_bpw


def _fmt_gib(b: float) -> str:
    return f"{b / C.GIB:.2f} GiB"


def build_report(out_dir: str | Path) -> str:
    out = Path(out_dir)
    lines = [f"# kquant report — {C.MODEL_ID} @ {C.REVISION[:8]}", ""]

    inv_p = out / "inventory.json"
    if inv_p.exists():
        inv = json.loads(inv_p.read_text())
        lines += [
            "## Inventory",
            f"- shards present: {inv['shards_present']}/96, "
            f"complete MoE layers: {inv['num_complete_moe_layers']}/92",
            f"- latent projections dtype: **{inv['latent_proj_dtype']}**",
            "",
            "| category | params | GiB |",
            "|---|---:|---:|",
        ]
        for cat, r in inv["categories"].items():
            lines.append(f"| {cat} | {r['params']:,} | {r['bytes'] / C.GIB:.2f} |")
        lines.append("")

    st_p = out / "static.kqstats"
    if st_p.exists():
        b = load_bundle(st_p)
        ent = np.asarray(b["entropy"])
        zf = np.asarray(b["zero_frac"])
        sat = np.asarray(b["sat_frac"])
        mask = np.asarray(b["n_elem"]) > 0
        lines += ["## Static stats (measured layers)", ""]
        for i, m in enumerate(C.EXPERT_MATRICES):
            mm = mask[:, :, i]
            if not mm.any():
                continue
            lines.append(
                f"- **{m}**: code entropy mean {ent[:, :, i][mm].mean():.3f} b "
                f"(p5 {np.percentile(ent[:, :, i][mm], 5):.3f} / "
                f"p95 {np.percentile(ent[:, :, i][mm], 95):.3f}), "
                f"zero frac {zf[:, :, i][mm].mean():.3%}, "
                f"saturation frac {sat[:, :, i][mm].mean():.3%}"
            )
        ent_all = ent[mask]
        lines += [
            "",
            f"- **effective lossless bound**: mean code entropy "
            f"{ent_all.mean():.3f} b + 0.25 b scale ≈ "
            f"**{ent_all.mean() + 0.25:.2f} bpw** vs 4.25 stored",
            "",
        ]

    rf_p = out / "refit.json"
    if rf_p.exists():
        rf = json.loads(rf_p.read_text())["global"]
        lines += ["## Codebook refit (global)", ""]
        for base in ("nf3", "nf2"):
            e = rf[base]
            gain = 1 - e["refit_mse"] / max(e["stock_mse"], 1e-12)
            lines.append(
                f"- **{base}**: stock MSE {e['stock_mse']:.5f} → refit "
                f"{e['refit_mse']:.5f} ({gain:.1%} lower); levels "
                f"{[round(x, 4) for x in e['refit_levels']]}"
            )
        lines.append("")

    di_p = out / "distortion.kqstats"
    if di_p.exists():
        b = load_bundle(di_p)
        lines += ["## Static distortion (sum over matrices, measured layers)", ""]
        m2 = None
        if st_p.exists():
            m2 = np.asarray(load_bundle(st_p)["m2_sum"]).sum(axis=2)
        for key in sorted(b.keys()):
            if not key.startswith("distortion.") or key.startswith("distortion_fold"):
                continue
            d = np.asarray(b[key]).sum(axis=2)
            mask = d > 0
            if not mask.any():
                continue
            line = f"- **{key.removeprefix('distortion.')}**"
            if m2 is not None:
                rel = d[mask] / np.maximum(m2[mask], 1e-12)
                line += (
                    f": relative distortion mean {rel.mean():.4f}, "
                    f"p50 {np.percentile(rel, 50):.4f}, "
                    f"p95 {np.percentile(rel, 95):.4f}"
                )
            lines.append(line)
        lines.append("")

    al_p = out / "allocation.json"
    if al_p.exists():
        doc = json.loads(al_p.read_text())
        ach = doc["achieved"]
        lines += [
            "## Allocation",
            f"- target: {doc['budget'].get('target_bpw', '?')} bpw → ranked layers "
            f"**{ach.get('solved_bpw', 0):.3f} bpw**; full-model extrapolation "
            f"{ach['bpw']:.3f} bpw ({_fmt_gib(ach['total_bytes'])})",
            f"- tier counts: {ach['counts']}",
            f"- ranker: {doc['ranker']} | solver: {doc['solver']}",
            f"- layers ranked: {len(doc['layers_solved'])}/92",
            "",
        ]
        kpl = ach["keep_per_layer"]
        solved = {int(k): v for k, v in kpl.items()}
        vals = [v for k, v in sorted(solved.items())]
        if any(vals):
            lines.append(
                f"- keeps/layer: min {min(vals)}, median "
                f"{sorted(vals)[len(vals) // 2]}, max {max(vals)}"
            )
            lines.append("")

    vf_p = out / "verify.json"
    if vf_p.exists():
        v = json.loads(vf_p.read_text())
        lines += [
            "## Verify",
            f"- ok: **{v['ok']}** ({v['samples']} samples; keep bitwise "
            f"{v['keep_bitwise_ok']}/{v['keep_bitwise_ok'] + v['keep_bitwise_bad']}, "
            f"demoted deterministic {v['demoted_deterministic_ok']}"
            f"/{v['demoted_deterministic_ok'] + v['demoted_deterministic_bad']})",
            f"- artifact bytes: {_fmt_gib(v['artifact_file_bytes'])} "
            f"(≈ {bytes_to_bpw(v['artifact_tensor_bytes']):.3f} bpw if all layers)",
            "",
        ]
    return "\n".join(lines)
