"""Layer-closure test for EXL3 artifacts: full top-k SiTU MoE forward through
the b12x trellis kernel (TP12-sliced, partials summed) vs an fp32 reference
built from the dequantized MXFP4 source weights.

This is the test that sees cross-expert error correlation, which per-matrix
proxy errors cannot. Used to A/B per-expert vs shared su sign vectors.

Usage:
  python scripts/closure_exl3_layer.py --artifact DIR --layer 46 [--tokens 256]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from kquant.io.hf_cache import resolve
from kquant.io.mxfp4 import dequant
from kquant.io.stream import load_tensor

HIDDEN = 3584
INTER = 3072
TP = 12
TOPK = 8


def situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    g = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    u = 25.0 * torch.tanh(up / 25.0)
    return g * u


def reference_forward(cache, layer, experts, x, topk_ids, topk_w):
    """fp32 SiTU MoE over dequantized MXFP4 source weights."""
    y = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    ws = {}
    # fp16 storage (fp32 doesn't fit alongside a running pack worker); GPU
    # matmuls accumulate fp32, and both A/B arms share the same reference.
    for e in experts:
        base = (
            f"language_model.model.layers.{layer}.block_sparse_moe."
            f"experts.{e}"
        )
        ws[e] = [
            dequant(
                load_tensor(cache, f"{base}.{m}.weight_packed"),
                load_tensor(cache, f"{base}.{m}.weight_scale"),
            ).to(x.device, torch.float16)
            for m in ("w1", "w3", "w2")
        ]
    for t in range(x.shape[0]):
        xt = x[t].to(torch.float16)
        for k in range(TOPK):
            w1, w3, w2 = ws[experts[int(topk_ids[t, k])]]
            h = situ((xt @ w1.T).float(), (xt @ w3.T).float())
            y[t] += topk_w[t, k].float() * (h.half() @ w2.T).float()
    return y


def trellis_forward(art, layer, experts, x, topk_ids, topk_w, mcg,
                    broadcast=False):
    """TP12-sliced trellis kernel forward; per-rank partials summed."""
    from sparkinfer.moe import trellis_moe

    dev = x.device
    E = len(experts)
    base = (
        f"language_model.model.layers.{layer}.block_sparse_moe.experts"
    )
    with safe_open(
        str(Path(art) / f"exl3-layer-{layer:05d}.safetensors"), framework="pt"
    ) as sf:
        def g(e, m, part):
            return sf.get_tensor(f"{base}.{e}.{m}.exl3_{part}")

        w13_tr = torch.stack(
            [
                torch.stack([g(e, "w1", "trellis"), g(e, "w3", "trellis")])
                for e in experts
            ]
        ).to(dev)  # [E, 2, H/16, I/16, 48]
        w13_suh = torch.stack(
            [torch.stack([g(e, "w1", "suh"), g(e, "w3", "suh")]) for e in experts]
        ).to(dev)  # [E, 2, H]
        w13_svh = torch.stack(
            [torch.stack([g(e, "w1", "svh"), g(e, "w3", "svh")]) for e in experts]
        ).to(dev)  # [E, 2, I]
        w2_tr = torch.stack([g(e, "w2", "trellis") for e in experts]).to(dev)
        w2_suh = torch.stack([g(e, "w2", "suh") for e in experts]).to(dev)
        w2_svh = torch.stack([g(e, "w2", "svh") for e in experts]).to(dev)

    if broadcast:
        # shared-su artifact: collapse identical per-expert H-side rows to
        # [1, H] broadcast tables, exercising the zero-expert-stride kernels
        assert torch.equal(w13_suh[0], w13_suh[1]) and torch.equal(
            w2_svh[0], w2_svh[1]
        ), "--broadcast requires a shared-su artifact"
        w13_suh = w13_suh[:1]
        w2_svh = w2_svh[:1]

    it, m = INTER // TP // 16, x.shape[0]
    y = torch.zeros(m, HIDDEN, dtype=x.dtype, device=dev)
    for r in range(TP):
        sl_t, sl = slice(r * it, (r + 1) * it), slice(
            r * INTER // TP, (r + 1) * INTER // TP
        )
        inter_rot = torch.cat(
            [w13_svh[:, 0, sl], w2_suh[:, sl], w13_svh[:, 1, sl]], dim=1
        ).contiguous()
        weights = trellis_moe.prepare_weights(
            w13_tr[:, :, :, sl_t].permute(1, 0, 2, 3, 4).contiguous(),
            w2_tr[:, sl_t].contiguous(),
            gate_suh=w13_suh[:, 0].contiguous(),
            up_suh=w13_suh[:, 1].contiguous(),
            intermediate_rotations=inter_rot,
            down_svh=w2_svh.contiguous(),
            codebook="mcg",
            mcg=mcg,
        )
        caps = trellis_moe.Caps(
            max_tokens=m,
            num_topk=TOPK,
            num_experts=E,
            hidden_size=HIDDEN,
            intermediate_size=INTER // TP,
            device=dev.index or 0,
            activation="situ",
            trellis_bits=3,
            route_num_experts=E,
        )
        plan = trellis_moe.plan(caps)
        scratch = torch.empty(plan.scratch_nbytes, dtype=torch.uint8, device=dev)
        binding = trellis_moe.bind(
            plan,
            scratch=scratch,
            a=x.contiguous(),
            weights=weights,
            topk_weights=topk_w,
            topk_ids=topk_ids,
        )
        y += trellis_moe.run(binding=binding)[:m]
        del weights, scratch
        torch.cuda.empty_cache()
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--layer", type=int, default=46)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--broadcast", action="store_true")
    args = ap.parse_args()

    cache = resolve()
    alloc = json.loads(
        (Path(args.artifact) / "allocation-exl3.json").read_text()
    )["layers"]
    experts = alloc[str(args.layer)]["exl3"]
    manifest = json.loads(
        (Path("/models/Kimi-K3-EXL3-3p19") / "kquant_exl3_manifest.json").read_text()
    )

    dev = torch.device("cuda", 0)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    x = (
        torch.randn(args.tokens, HIDDEN, generator=gen).to(dev, torch.bfloat16)
    )
    perm = torch.stack(
        [torch.randperm(len(experts), generator=gen)[:TOPK] for _ in range(args.tokens)]
    )
    topk_ids = perm.to(dev, torch.int32)
    logits = torch.randn(args.tokens, TOPK, generator=gen)
    topk_w = torch.softmax(logits, dim=-1).to(dev, torch.float32)

    y_q = trellis_forward(
        args.artifact, args.layer, experts, x, topk_ids, topk_w,
        manifest["mcg_mult"], broadcast=args.broadcast,
    )
    y_ref = reference_forward(cache, args.layer, experts, x, topk_ids, topk_w)

    rel = (y_q.float() - y_ref).norm() / y_ref.norm()
    print(f"artifact={args.artifact} layer={args.layer} tokens={args.tokens}")
    print(f"closure rel error: {rel.item():.4f}")


if __name__ == "__main__":
    main()
