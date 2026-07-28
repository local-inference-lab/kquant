# kquant report — moonshotai/Kimi-K3 @ 9f62e4e9

## Inventory
- shards present: 96/96, complete MoE layers: 92/92
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
| vision | 447,358,976 | 0.83 |

## Static stats (measured layers)

- **w1**: code entropy mean 3.754 b (p5 3.749 / p95 3.764), zero frac 11.506%, saturation frac 1.021%
- **w3**: code entropy mean 3.753 b (p5 3.749 / p95 3.761), zero frac 11.499%, saturation frac 1.005%
- **w2**: code entropy mean 3.754 b (p5 3.749 / p95 3.761), zero frac 11.738%, saturation frac 1.045%

- **effective lossless bound**: mean code entropy 3.754 b + 0.25 b scale ≈ **4.00 bpw** vs 4.25 stored

## Codebook refit (global)

- **nf3**: stock MSE 0.00673 → refit 0.00552 (17.9% lower); levels [-1.0, -0.5872, -0.3084, -0.1021, 0.1021, 0.3084, 0.5872, 1.0]
- **nf2**: stock MSE 0.04077 → refit 0.03538 (13.2% lower); levels [-1.0, -0.3114, 0.3114, 1.0]

## Static distortion (sum over matrices, measured layers)

- **nf2**: relative distortion mean 0.2297, p50 0.2293, p95 0.2338
- **nf2_refit**: relative distortion mean 0.1938, p50 0.1936, p95 0.1974
- **nf3**: relative distortion mean 0.0387, p50 0.0386, p95 0.0391
- **nf3_refit**: relative distortion mean 0.0309, p50 0.0309, p95 0.0313

## Allocation
- target: 3.25 bpw → ranked layers **3.250 bpw**; full-model extrapolation 3.250 bpw (1030.15 GiB)
- tier counts: {'keep_mxfp4': 0, 'nf3_refit': 82432}
- ranker: {'ranker': 'l0', 'traffic': 'bias', 'alpha': 1.0, 'w2': 'folded', 'demoted_formats': ['nf3_refit']} | solver: {'min_keep_per_layer': 0}
- layers ranked: 92/92

## Verify
- ok: **True** (128 samples; keep bitwise 0/0, demoted deterministic 128/128)
- artifact bytes: 1030.22 GiB (≈ 3.250 bpw if all layers)
