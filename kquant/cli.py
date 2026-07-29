"""kquant command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _device(arg: str | None) -> str:
    if arg:
        return arg
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_layers(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _resolve_cache(args):
    from kquant.io.hf_cache import resolve

    return resolve(args.cache_dir, args.revision)


def _formats_registry(out_dir: Path, names: list[str]):
    from kquant.quantize import BUILTIN_FORMATS
    from kquant.stats.refit import load_refit_formats

    reg = dict(BUILTIN_FORMATS)
    refit_p = out_dir / "refit.json"
    if refit_p.exists():
        reg.update(load_refit_formats(refit_p))
    missing = [n for n in names if n not in reg and n != "keep_mxfp4"]
    if missing:
        raise SystemExit(f"unknown formats {missing}; have {sorted(reg)}")
    return {n: reg[n] for n in names if n != "keep_mxfp4"}


def cmd_inventory(args):
    from kquant.inventory import build_inventory, format_summary, write_inventory

    cache = _resolve_cache(args)
    inv = build_inventory(cache)
    p = write_inventory(inv, Path(args.out))
    print(format_summary(inv))
    print(f"\nwrote {p}")


def cmd_stats(args):
    from kquant.dynstats import save_bundle
    from kquant.stats.static import run_static_stats

    cache = _resolve_cache(args)
    arrays, meta = run_static_stats(
        cache, device=_device(args.device), batch_size=args.batch_size,
        layers=_parse_layers(args.layers),
    )
    p = save_bundle(Path(args.out) / "static", arrays, "kquant-static", meta)
    print(f"wrote {p} ({meta['batches']} batches, {meta['elapsed_s']}s)")


def cmd_refit(args):
    from kquant.dynstats import load_bundle
    from kquant.stats.refit import refit_codebooks

    b = load_bundle(Path(args.out) / "static.kqstats")
    result = refit_codebooks(np.asarray(b["joint_hist"]))
    p = Path(args.out) / "refit.json"
    p.write_text(json.dumps(result, indent=1))
    g = result["global"]
    for base in ("nf3", "nf2"):
        e = g[base]
        print(
            f"{base}: stock {e['stock_mse']:.6f} -> refit {e['refit_mse']:.6f}  "
            f"levels {[round(x, 4) for x in e['refit_levels']]}"
        )
    print(f"wrote {p}")


def cmd_distortion(args):
    from kquant.dynstats import save_bundle
    from kquant.stats.distortion import run_distortion

    cache = _resolve_cache(args)
    fmts = _formats_registry(Path(args.out), args.formats.split(","))
    arrays, meta = run_distortion(
        cache, formats=fmts, device=_device(args.device),
        batch_size=args.batch_size, layers=_parse_layers(args.layers),
        fold_u=not args.no_fold,
    )
    p = save_bundle(Path(args.out) / "distortion", arrays, "kquant-distortion", meta)
    print(f"wrote {p} ({meta['batches']} batches, {meta['elapsed_s']}s)")


def cmd_rank(args):
    from kquant import constants as C
    from kquant.allocation import save_allocation
    from kquant.dynstats import ComposedStats, load_bundle
    from kquant.rank.budget import bpw_to_bytes, bytes_to_bpw, compute_budget
    from kquant.rank.l0 import damage_table
    from kquant.rank.solver import solve

    out = Path(args.out)
    bundles = [load_bundle(out / "static.kqstats"), load_bundle(out / "distortion.kqstats")]
    for extra in args.stats or []:
        bundles.append(load_bundle(extra))
    stats = ComposedStats(bundles)

    if args.budget == "auto":
        inv = json.loads((out / "inventory.json").read_text())
        budget = compute_budget(inv)
        budget_bytes = budget.expert_budget_bytes
        print(budget.summary())
        budget_info = {"source": "ledger", "target_bpw": budget.expert_bpw,
                       "bytes": budget_bytes}
    else:
        target = float(args.target_bpw)
        budget_bytes = bpw_to_bytes(target)
        budget_info = {"source": "target_bpw", "target_bpw": target,
                       "bytes": budget_bytes}

    demoted = args.formats.split(",")
    formats, damage, traffic = damage_table(
        stats, demoted, traffic_mode=args.traffic,
        traffic_alpha=args.traffic_alpha, w2_mode=args.w2,
    )
    solved_mask = np.zeros(C.NUM_MOE_LAYERS, dtype=bool)
    d0 = damage[:, :, 1]
    solved_mask[:] = np.isfinite(d0).all(axis=1) & (d0.sum(axis=1) > 0)
    alloc = solve(
        formats, damage, budget_bytes, solved_mask,
        min_keep_per_layer=args.min_keep,
    )
    p = save_allocation(
        alloc, out / "allocation.json", budget_info,
        {"ranker": "l0", "traffic": args.traffic, "alpha": args.traffic_alpha,
         "w2": args.w2, "demoted_formats": demoted},
        {"min_keep_per_layer": args.min_keep},
        revision=args.revision,
    )
    print(
        f"allocation: {alloc.counts()} | ranked layers: {len(alloc.layers_solved)} "
        f"at {alloc.solved_bpw:.3f} bpw | full-model extrapolation "
        f"{alloc.achieved_bpw:.3f} bpw ({alloc.total_bytes / C.GIB:.1f} GiB)"
        f"\nwrote {p}"
    )


def cmd_pack(args):
    from kquant.allocation import load_allocation
    from kquant.pack.writer import pack_artifact

    cache = _resolve_cache(args)
    out = Path(args.out)
    alloc, doc = load_allocation(out / "allocation.json")
    demoted = [f for f in alloc.formats if f != "keep_mxfp4"]
    fmts = _formats_registry(out, demoted)
    config = pack_artifact(
        cache, alloc, fmts, args.dest or (out / "artifact"),
        device=_device(args.device), layers=_parse_layers(args.layers),
        batch_size=args.batch_size,
    )
    print(
        f"packed layers {config['layers_packed'][0]}..{config['layers_packed'][-1]}"
        f" ({len(config['layers_packed'])}) | tiers {config['tier_expert_counts']}"
        f" | {config['total_bytes'] / 2**30:.1f} GiB in {config['elapsed_s']}s"
    )


def cmd_verify(args):
    from kquant.allocation import load_allocation
    from kquant.verify import verify_artifact

    cache = _resolve_cache(args)
    out = Path(args.out)
    alloc, _ = load_allocation(out / "allocation.json")
    demoted = [f for f in alloc.formats if f != "keep_mxfp4"]
    fmts = _formats_registry(out, demoted)
    report = verify_artifact(
        cache, args.dest or (out / "artifact"), fmts,
        alloc.assignment, alloc.formats, samples=args.samples,
    )
    (out / "verify.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if not report["ok"]:
        sys.exit(1)


def cmd_report(args):
    from kquant.report import build_report

    text = build_report(args.out)
    p = Path(args.out) / "report.md"
    p.write_text(text)
    print(text)
    print(f"\nwrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kquant")
    ap.add_argument("--out", default="out", help="output directory (default ./out)")
    ap.add_argument("--cache-dir", default=None,
                    help="HF cache repo dir (default: resolve from HF_HUB_CACHE)")
    from kquant import constants as C

    ap.add_argument("--revision", default=C.REVISION)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inventory")
    p.set_defaults(fn=cmd_inventory)

    p = sub.add_parser("stats")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--layers", default=None, help="e.g. 1-8 or 1,2,5")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("refit")
    p.set_defaults(fn=cmd_refit)

    p = sub.add_parser("distortion")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--layers", default=None)
    p.add_argument("--formats", default="nf3,nf2")
    p.add_argument("--no-fold", action="store_true")
    p.set_defaults(fn=cmd_distortion)

    p = sub.add_parser("rank")
    p.add_argument("--formats", default="nf3",
                   help="demoted formats, cheapest-last not required")
    p.add_argument("--target-bpw", default=str(C.DEFAULT_TARGET_BPW))
    p.add_argument("--budget", default="target",
                   help="'target' (use --target-bpw) or 'auto' (ledger)")
    p.add_argument("--traffic", default="bias",
                   choices=["bias", "uniform", "measured"])
    p.add_argument("--traffic-alpha", type=float, default=1.0)
    p.add_argument("--w2", default="raw", choices=["raw", "folded"])
    p.add_argument("--min-keep", type=int, default=0)
    p.add_argument("--stats", nargs="*", help="extra .kqstats bundles (dynamic)")
    p.set_defaults(fn=cmd_rank)

    p = sub.add_parser("pack")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--layers", default=None)
    p.add_argument("--dest", default=None)
    p.set_defaults(fn=cmd_pack)

    p = sub.add_parser("verify")
    p.add_argument("--dest", default=None)
    p.add_argument("--samples", type=int, default=64)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
