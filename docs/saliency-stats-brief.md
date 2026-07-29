# Technical brief: K3 expert saliency statistics (L1/L2 dynstats)

**For:** Martin — **From:** Luke / kquant — **Re:** measured expert statistics
collection on the TP16 K3 deployment, feeding the Phase-B keep+EXL3 artifact.

## Why

Every allocation decision in the current kquant artifacts (uniform NF3-refit
and the shelved keep/NF3/NF2 hybrid) is driven by an **L0 traffic proxy**
derived from the router's `e_score_correction_bias` — a checkpoint-static
guess. Our headline claims (top 5% of experts carry ~55% of routing mass;
hybrid beats uniform NF3 by 2.9x traffic-weighted distortion) are computed
against that same proxy, so they are circular. Measured statistics break the
circularity and directly select the MXFP4 keep-set for the Phase-B
keep+EXL3-3.0 artifact (~15% keeps at 3.19 bpw, TP12-friendly). Getting these
right is the highest-leverage quality input to the next quant.

## L1 — routing statistics (do this first; always-on, negligible cost)

Per (MoE layer, expert), three accumulators over served traffic:

| key | shape | meaning |
|---|---|---|
| `tokens_routed` | [92, 896] | count of tokens that selected this expert (post-bias top-16 selection) |
| `gate_sum` | [92, 896] | sum of the **applied** routing weights (the renormalized combine weights actually multiplied into expert outputs — not the selection scores, not the bias) |
| `gate_sq_sum` | [92, 896] | sum of squared applied weights |

- **Layer indexing (off-by-one hazard):** row `i` = decoder layer `i+1`
  (MoE layers are decoder layers 1..92; layer 0 is dense). This matches every
  other kquant bundle.
- Accumulate in fp64 on rank 0 only (`topk_ids`/`topk_weights` are replicated
  across TP; per layer it's one `bincount(topk_ids, weights=...)`).
- Collection point in your branch: the FusedMoE apply path right after topk —
  suggest an env-gated hook (`VLLM_KQUANT_STATS_PATH=/models/dynstats/...`)
  that dumps on SIGUSR1 and at graceful shutdown.

**Format:** one safetensors file + `manifest.json` sidecar ("kqstats bundle"):

```
manifest.json: { "schema_version": 1, "producer": "martin-vllm-tp16",
  "model": "moonshotai/Kimi-K3", "revision": "9f62e4e9...",
  "tokens_total": <int>, "corpus": "<free-form mix description>",
  "collected": "<ISO date range>" }
data.safetensors: tokens_routed, gate_sum, gate_sq_sum  (fp64 or fp32)
```

kquant's `--traffic measured` mode consumes `gate_sq_sum` directly
(per-layer mean-normalized); the other two keys are for diagnostics and
concentration curves. Loader validates shape `[92, 896]` only — dtype is free.

**Volume target:** ≥10M routed tokens of *mixed* workload (chat + code +
long-context if possible). At top-16-of-896, that's ~180k expected hits per
expert cell — the tail distribution is what we're actually after, so more is
better and corpus diversity matters more than raw count. Please record the mix
honestly in the manifest; if traffic is homogeneous (e.g. all benchmarks),
say so — we'll discount the tail accordingly.

## L2 — activation second moments (phase two; sampled)

Per (MoE layer, expert), per-channel mean-square of expert inputs:

| key | shape | meaning |
|---|---|---|
| `act_m2_in` | [92, 896, 3584] | E[x²] per channel of the expert input (latent space) — w1/w3 Hessian-diagonal proxy |
| `act_m2_mid` | [92, 896, 3072] | E[h²] per channel of the w2 input (post-SiTU intermediate) |
| `act_count` | [92, 896] | sample counts backing the above |

Sampled (1-in-16 requests is plenty), fp32 accumulators, ~2.3 GB total.
These power error-weighted quantization in Phase B (activation-aware EXL3
grouping/scale decisions and a re-check of the refit codebook under true error
weighting) plus a saliency score = traffic x sensitivity that may reorder the
keep-set versus traffic alone. If the scatter-by-expert accumulation is
awkward in the fused path, skip L2 for now — L1 alone unblocks the keep-set.

## What happens downstream

1. You drop `dynstats-l1-<date>.kqstats` somewhere shared.
2. We run `kquant rank --traffic measured --formats ... --target-bpw 3.19`
   → measured-traffic allocation + a proxy-vs-measured concentration report
   (how wrong was the bias proxy — publishable curiosity on its own).
3. Keep-set goes into the Phase-B keep+EXL3-3.0 pack; eval bake-off vs the
   uniform NF3-refit control you're serving now.

One more thing worth aligning: you're currently serving the artifact with
b12x's **default** NF3 codebook — the artifact was quantized with **refit
levels** (−18% MSE) carried in its `kquant_config.json` (`nf3_levels`). Our
branch has the override plumbing (module-global + env stamp for the compile
cache); grab it or we'll send the patch — your measured stats are unaffected
either way, but serving quality is ~18% of weight-MSE better with it.
