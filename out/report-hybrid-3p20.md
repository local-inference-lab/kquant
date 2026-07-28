# kquant report — moonshotai/Kimi-K3 @ 9f62e4e9

## Inventory
- shards present: 73/96, complete MoE layers: 72/92
- latent projections dtype: **BF16**

| category | params | GiB |
|---|---:|---:|
| attention | 36,190,795,008 | 67.43 |
| attn_residual | 2,680,832 | 0.00 |
| dense_mlp | 726,663,168 | 1.35 |
| embeddings | 1,174,405,120 | 2.19 |
| latent_proj | 4,727,310,336 | 8.81 |
| lm_head | 1,174,405,120 | 2.19 |
| norms_other | 1,340,416 | 0.00 |
| routed_experts | 2,722,740,830,208 | 1347.12 |
| router | 590,955,008 | 1.10 |
| shared_experts | 12,155,092,992 | 22.64 |

## Static stats (measured layers)

- **w1**: code entropy mean 3.759 b (p5 3.753 / p95 3.766), zero frac 11.515%, saturation frac 1.090%
- **w3**: code entropy mean 3.759 b (p5 3.753 / p95 3.766), zero frac 11.519%, saturation frac 1.091%
- **w2**: code entropy mean 3.758 b (p5 3.755 / p95 3.766), zero frac 12.967%, saturation frac 1.146%

- **effective lossless bound**: mean code entropy 3.759 b + 0.25 b scale ≈ **4.01 bpw** vs 4.25 stored

## Codebook refit (global)

- **nf3**: stock MSE 0.00668 → refit 0.00546 (18.3% lower); levels [-1.0, -0.5863, -0.3085, -0.102, 0.102, 0.3085, 0.5863, 1.0]
- **nf2**: stock MSE 0.04134 → refit 0.03566 (13.7% lower); levels [-1.0, -0.3098, 0.3098, 1.0]

## Static distortion (sum over matrices, measured layers)

- **nf2**: relative distortion mean 0.2349, p50 0.2322, p95 0.2508
- **nf2_refit**: relative distortion mean 0.1969, p50 0.1950, p95 0.2088
- **nf3**: relative distortion mean 0.0389, p50 0.0386, p95 0.0405
- **nf3_refit**: relative distortion mean 0.0309, p50 0.0308, p95 0.0319

## Allocation
- target: 3.2 bpw → achieved **2.271 bpw** (719.72 GiB)
- tier counts: {'keep_mxfp4': 325, 'nf3_refit': 1052, 'nf2_refit': 81055}
- ranker: {'ranker': 'l0', 'traffic': 'bias', 'alpha': 1.0, 'w2': 'folded', 'demoted_formats': ['nf3_refit', 'nf2_refit']} | solver: {'min_keep_per_layer': 0}
- layers ranked: 2/92

- keeps/layer: min 0, median 0, max 205

## Verify
- ok: **True** (64 samples; keep bitwise 13/13, demoted deterministic 51/51)
- artifact bytes: 23.65 GiB (≈ 0.075 bpw if all layers)
