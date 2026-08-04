# Codec transfers for K3 expert-weight quantization

Status: exploratory, 2026-07-31. This note records the cross-discipline
literature review and the interim-checkpoint experiments behind the current
codec proposal. It is not a model-quality result or a production format
specification.

## Bottom line

The most promising successor to uniform Hadamard-rotated EXL3 is not another
single universal vector quantizer. It is an **expert-static rate-transfer
trellis codec** around EXL3:

1. retain the dense captured-Hessian EXL encoder as the base quantizer;
2. rank coupled expert middle channels by routed activation importance;
3. search the `R0..R12` ladder, transferring `r` complete records from K3 to
   K2 and the same number from K3 to K4 at exactly 3 bits per weight;
4. let the full dense Hessian couple all contexts during LDLQ, with the
   high-importance context processed first;
5. bake the context order into the expert's physical intermediate-neuron
   permutation, so no runtime gather or per-channel context map is needed;
6. after encoding, move complete 128-channel records into low/high-rate pairs,
   giving every TP12 rank two records averaging 3 bits per weight; and
7. test shared `r` against separate `(r13, r2)` schedules under the same
   physical permutation; and
8. select the expert schedule by held-out routed-output distortion plus the
   exact payload cost, rather than raw weight MSE.

On the very small interim-only pilot, choosing between uniform `3,3,3,3` and
mixed `2,3,3,4` modes using the capture's training split reduced held-out
`w2` output NMSE by a 22.4% geometric mean relative to ordinary captured-H
EXL3, with 7 of 8 experts improving. The scale-inclusive simulated rate was
3.00968 bits/weight before unimplemented stream alignment and offset-table
costs. These numbers are hypothesis-forming only: most experts have two
training rows and one validation row.

## Hard data boundary

No official Kimi-K3 checkpoint was resolved or loaded for this work.

- Weight source: `/models/Kimi-K3-EXL3-3p09-serve`, resolving to
  `/data/models/Kimi-K3-EXL3-3p09-serve`.
- Float source weights: only the original MXFP4 weights retained inside that
  interim hybrid artifact.
- Activation and routing source:
  `/data/kquant/k3-route-aware-pilot-v2-exl3-r4.kqcapture`, captured from the
  interim EXL3 variant.
- Hessian subset:
  `out/k3-route-aware-pilot-v2-exl3-r4-subset.kqhess`, built from that capture.
- Exact EXL encoding imports only the encoder implementation under
  `/home/luke/projects/exllamav3`; it does not instantiate a serving model.

The eight held-out `w2` cases are layer/expert 1/15, 1/61, 24/576, 24/6,
24/835, 40/128, 40/495, and 40/813. This is a deliberately diverse pilot,
not a representative sample of all 82,432 routed-expert assignments. It is
also biased toward the 7,007 experts retained by the current allocation.

## Geometry and the useful coding atom

K3 expert matrices are:

- `w1` and `w3`: `[3072, 3584]` in PyTorch `[out, in]` order;
- `w2`: `[3584, 3072]`; and
- the coupled intermediate-neuron axis has length 3072.

At TP12 that axis is 256 channels per rank. EXL's trellis
payload is tiled at 16 input rows, but 128 channels are the useful **format
motion atom** because the transform/scale representation is 128-aligned and
TP12 consumes exactly two such records per rank. This is not a claim that all
SM120 MMA atoms are 16x16, nor that the codec should dictate a single MMA
shape. A decoder may subdivide a 128-channel record into the MMA atoms that
fit its operand types and layout.

For `R_r`, there are `r` K2/K4 pairs and `12-r` K3/K3 pairs. Both pair types
average 3 bits. Equal rate does not establish equal runtime: K2 and K4 decoding may have
nonlinear costs, so the kernel still needs a measured balance test.

## What the other fields actually teach

The useful transfer from post-DXT texture and modern video codecs is their
control structure, not their color-specific endpoint math.

| Field / codec | Transferable idea | Interim result or implication |
|---|---|---|
| BC7 | A fixed-size block chooses among a small set of modes, partitions, endpoint layouts, and index widths. | Strong match for an expert mode ID selecting a known mixed-K layout. Direct endpoint-line coding did not fit weight tiles well. |
| ASTC | Per-block partition and endpoint modes divide a fixed budget; BISE efficiently represents non-power-of-two alphabets; weight grids trade local detail for side data. | Supports explicit rate allocation and compact mode syntax. The experiment says to allocate bits among activation contexts, not spatial image regions. BISE is secondary unless non-power-of-two trellis alphabets prove useful. |
| UASTC | A transcodable mode-rich intermediate representation can prioritize decode targets over legacy bitstream compatibility. | Supports changing the EXL format and choosing records around the actual TP/MMA consumer. “Zero format change” is not assumed. |
| VVC dependent quantization | Trellis state and context-aware RDO can beat independent scalar decisions; practical encoders prune and vectorize the search. | EXL's trellis remains a strong base. The next gain is better distortion/context modeling and mode pruning, not abandoning trellis search. |
| AV2 transform/entropy coding | TCQ is combined with RD-selected partitions, data-driven/mode-dependent transforms, adaptive coefficient contexts, skip coding, quantization matrices, and compact signaling. | Closest architectural precedent. The empirical K3 result is the same: calibrated trellis plus modes, ordering, rate allocation, and sparse skip—not one replacement quantizer. |
| Daala/Opus PVQ | Separate gain from shape, conserve energy, and predict in a transformed domain. | Attractive in theory, but tested PVQ/gain-shape variants were not competitive on these weights. Channel gain side information alone did not close the gap. |
| ELIC | Energy compaction justifies uneven channel groups and progressive/contextual coding. | Directly motivated activation-ranked contexts. A layer-shared 4-D palette with mild uneven rates beat its uniform palette in 7/8 held-out experts, although captured-H EXL remained much better overall. |
| SoundStream and residual VQ | Stacked codebooks and structured stage dropout provide variable rate and scalable decoding. | Shared RVQ was testable and compact, but did not beat calibrated EXL. Its useful residue is the idea of a tiny discrete mode ladder, not an RVQ replacement. |
| Neural texture compression | Jointly compress correlated assets and use a small asset-specific decoder while preserving random access. | Suggests layer- or expert-family side models. A per-expert neural decoder is probably too expensive for MoE GEMM; a tiny layer-shared table or transform is still plausible. |
| ELIC/JPEG AI/modern neural codecs | Optimize latent allocation and the task distortion jointly; support variable rate or progressive layers. | Reinforces capture-driven distortion and candidate-mode RDO. A runtime neural entropy decoder is a poor match for direct weight streaming unless fused cost is demonstrated. |
| DeepCABAC / MPEG NNC | Quantize weights against task distortion, then entropy-code the nonuniform symbols with learned contexts. | Entropy coding could shrink files on disk, but irregular serial decode is unlikely to help a bandwidth-bound fused GEMM. Keep it as an offline/storage layer unless fixed-latency random access is retained. |
| Per-instance overfitted codecs such as COOL-CHIC | Spend small decoder parameters to specialize a generic representation to one source. | Motivates per-layer codebooks or a low-rank correction, but the tested low-rank and sparse enhancement layers did not justify their side cost yet. |

Primary references:

- [Microsoft BC7 format](https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format)
- [Arm ASTC format overview](https://chromium.googlesource.com/external/github.com/ARM-software/astc-encoder/%2B/HEAD/Docs/FormatOverview.md)
- [UASTC texture specification](https://github.com/BinomialLLC/basis_universal/wiki/UASTC-Texture-Specification/b624c07ad3c659e7b0f0badcb36e9a6b8820a99d)
- [Transform and Entropy Coding in AV2](https://arxiv.org/html/2601.02712v2)
- [Fast Dependent Quantization Using Trellis Pruning, Forward Context Adaptation and Vectorization](https://publica.fraunhofer.de/entities/publication/d5ef917c-1476-416d-95de-091f6935b6a0)
- [Perceptual Vector Quantization for Video Coding](https://arxiv.org/abs/1602.05209)
- [ELIC](https://arxiv.org/abs/2203.10886)
- [SoundStream](https://arxiv.org/abs/2107.03312)
- [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/publication/2023-08_random-access-neural-compression-material-textures)
- [JPEG AI documentation and standard publications](https://jpeg.org/jpegai/documentation.html)
- [DeepCABAC](https://publica.fraunhofer.de/entities/publication/5b0bcbc9-ca77-4190-bb61-c44f457949f3)
- [COOL-CHIC](https://arxiv.org/abs/2307.12706)

## Measured interim results

### Corrected EXL3 anchors

The first version of the exact EXL comparison accidentally scored against an
input tensor that the exllamav3 encoder mutates in place. Those near-1.0 NMSE
readings are invalid and were discarded. Correct identity-H raw weight NMSE
is about 0.015 for these cases. The routed-output anchors below were rerun
with an immutable source and separate training/validation activation splits.

| Layer / expert | Identity-H output NMSE | Captured-H output NMSE | Captured / identity |
|---|---:|---:|---:|
| 1 / 15 | 0.015514 | 0.014613 | 0.942 |
| 1 / 61 | 0.015613 | 0.015925 | 1.020 |
| 24 / 576 | 0.012259 | 0.000441 | 0.036 |
| 24 / 6 | 0.014808 | 0.003383 | 0.228 |
| 24 / 835 | 0.013720 | 0.000157 | 0.011 |
| 40 / 128 | 0.015738 | 0.001916 | 0.122 |
| 40 / 495 | 0.015404 | 0.002308 | 0.150 |
| 40 / 813 | 0.014959 | 0.002076 | 0.139 |

Captured-H EXL won 7/8 and its geometric-mean error ratio was 0.148. The
extreme gains are also a warning: the four-token Hessians are low-rank and
can specialize to the pilot. This establishes that real capture matters; it
does not estimate the gain of a representative production capture.

### Activation-context palettes

A layer-shared 4-D/4096-entry palette was trained separately for four
equal-population activation-energy contexts. Uniform indices use
`12,12,12,12` bits per four weights; the mild mode uses `10,11,13,14`. Both
average exactly 3 bits/weight before small codebook and map overheads.

| Layer / expert | Uniform palette | Mild uneven palette | Uneven / uniform |
|---|---:|---:|---:|
| 1 / 15 | 0.016695 | 0.012653 | 0.758 |
| 1 / 61 | 0.018799 | 0.019613 | 1.043 |
| 24 / 576 | 0.027111 | 0.018593 | 0.686 |
| 24 / 6 | 0.015086 | 0.007818 | 0.518 |
| 24 / 835 | 0.044737 | 0.034045 | 0.761 |
| 40 / 128 | 0.018303 | 0.009046 | 0.494 |
| 40 / 495 | 0.018384 | 0.010160 | 0.553 |
| 40 / 813 | 0.017832 | 0.008947 | 0.502 |

The uneven palette won 7/8 with a 0.643 geometric-mean ratio to its uniform
version. It roughly matched identity-H EXL (0.908 geometric-mean ratio, 5/8
wins), but was 6.14 times worse than captured-H EXL. The lesson is strong
evidence for **uneven rate**, not for replacing EXL with a shared palette.

### Full-Hessian mixed-rate EXL

Partitioning contexts into separate EXL calls discarded cross-context
Hessian terms and was rejected. The refined encoder preserves the full dense
Hessian and changes `K` per 16 input rows. Contexts are ordered from low to
high activation energy; LDLQ runs backward, so the important group is fixed
first and its error feeds later decisions.

For four contexts, selecting between uniform and mild modes by training-split
routed-output error produced:

| Layer / expert | Energy skew high/low | Captured-H anchor | Selected mode | Validation NMSE | Ratio |
|---|---:|---:|---|---:|---:|
| 1 / 15 | 91 | 0.014613 | `3,3,3,3` | 0.015059 | 1.031 |
| 1 / 61 | 80 | 0.015925 | `3,3,3,3` | 0.015191 | 0.954 |
| 24 / 576 | 1,062 | 0.000441 | `2,3,3,4` | 0.000230 | 0.522 |
| 24 / 6 | 138 | 0.003383 | `2,3,3,4` | 0.002484 | 0.734 |
| 24 / 835 | 28,094 | 0.000157 | `2,3,3,4` | 0.000064 | 0.405 |
| 40 / 128 | 37 | 0.001916 | `3,3,3,3` | 0.001905 | 0.994 |
| 40 / 495 | 30 | 0.002308 | `3,3,3,3` | 0.002146 | 0.930 |
| 40 / 813 | 36 | 0.002076 | `3,3,3,3` | 0.001935 | 0.932 |

The geometric-mean ratio was 0.776, the median was 0.931, and 7/8 improved.
A crude energy-skew threshold of 100 selected the same modes on this pilot,
whereas the encoder's Hessian proxy selected uniform mode everywhere. That
proxy is therefore not yet an adequate expert-mode RDO objective.

An eight-context targeted allocation, `2,3,3,3,3,3,3,4`, had a 0.847
geometric-mean ratio when applied universally, but improved only 4/8 cases.
Training-split selection among the uniform, targeted, and quartile modes gave
a 0.782 ratio and 6/8 wins. Combining all four- and eight-context candidates
only improved the geometric mean to about 0.757, too little to justify the
extra mode complexity before a larger capture.

The mixed mode's current accounting is:

- trellis indices: exactly 3.0 bits/weight;
- one `suh` and one `svh` scale vector plus a two-bit expert mode:
  3.0096757 bits/weight total for `w2`;
- explicit four-context map instead of a baked permutation:
  3.0098150 bits/weight; and
- not yet counted: variable-stream offsets, alignment, padding, and any
  kernel-facing descriptor table.

### Exact-zero channels

The entire retained MXFP4 tier was scanned without reading the official
checkpoint. Of 7,007 retained layer/expert assignments, 640 (9.13%) had at
least one exact all-zero `w2` input column. The mean was 2.194 zero columns
per expert, or 0.0714% of the 3072-channel middle axis; the median and 90th
percentile were both zero. A few layer 1/24 outliers were much larger, up to
1,071 columns.

Exact zero restore can materially improve raw weight NMSE in those outliers,
but it is not a universal codec. A useful format mode would move exact-zero
neurons to a tail and omit only whole 16- or 128-channel aligned records.
Because the retained tier is allocation-biased, these frequencies must not be
projected onto all demoted experts.

## Proposed format: mode-adaptive context EXL

The current working format proposal is intentionally allowed to differ from
EXL3.

### Encoder

1. Capture representative routed samples from the interim teacher and split
   documents/prompts into disjoint train and validation sets.
2. For each expert, estimate routed traffic, `w1/w3` input Hessian, `w2` input
   Hessian, and per-middle-channel activation energy. Stabilize estimates with
   shrinkage and minimum-sample rules.
3. Rank the 3072 middle channels in four-channel groups. Make the eventual
   partitions 128-aligned.
4. Encode the `R0..R12` ladder, then compare a shared `r` with separate
   `(r13, r2)` schedules. Treat old U4/T8/M4 as aliases for R0/R3/R6.
5. Preserve the full Hessian across all contexts. Encode in importance order;
   do not make independent Hessian blocks merely to simplify packing.
6. Score candidates on held-out routed latent/output distortion, including
   gate weights and traffic, plus `lambda * exact_bits`. Use a conservative
   fallback to `R0` when sample support is weak or train/validation disagree.
7. Apply a common physical intermediate-neuron permutation to `w1` and `w3`
   output rows and `w2` input columns. This is function-preserving and removes
   the runtime context map.
8. Reorder already-encoded complete 128-channel records into `K2/K4` and
   `K3/K3` pairs for physical TP placement. This decouples the LDLQ dependency
   order from the final storage/rank order. The proposal still needs exact
   pack/decode closure; it is not yet demonstrated by a production payload.

### Bitstream / checkpoint metadata

At minimum, a mixed expert needs:

- provisional `r13` and `r2` values or a later frozen mode ID;
- fixed mode-specific K schedules at 128-channel granularity;
- offsets or independently addressable payloads for variable-length K2/K3/K4
  trellis records;
- the usual `suh`, `svh`, seed/codebook identifier, and scaling contract; and
- the baked middle-neuron permutation reflected consistently in all three
  matrices.

The mode table should be global and tiny. A per-channel K map is unnecessary
if modes prescribe the schedule. Record starts should be aligned for the
actual loader/TMA path, even if this costs more than the ideal rate above.

### Decoder and kernel

The kernel contract should accept a K descriptor per 128-channel slab, while
the inner implementation chooses legal SM120 MMA shapes for the operand type.
The first microbenchmark should compare:

- uniform K3/K3 rank pairs;
- mixed K2/K4 pairs;
- mixed K3/K3 pairs in a mode-bearing format; and
- divergent expert mixtures in the real routed-token distribution.

Measure bytes read, effective bandwidth, decode instructions, register use,
occupancy, graph replay, and end-to-end MoE latency. A rate win that causes
rank imbalance or prevents efficient bulk decoding is not a useful codec win.

## Rejected or deferred directions

- **Endpoint/line and low-rank 16x16 block modes:** weight tiles were not
  sufficiently image-like, and side data erased the nominal rate advantage.
- **Shared RVQ/PVQ as the main representation:** competitive among alternate
  codebook experiments but below captured-H EXL on routed output.
- **Diagonal or block whitening followed by VQ:** naive KLT/Cholesky transfer
  did not reproduce LDLQ's error feedback and regressed.
- **Inter-expert or spatial prediction:** nearest/permuted expert residuals
  were too large; expert identity is not video motion.
- **Sparse enhancement layer:** reduced selected residuals but not enough for
  its position/value side stream.
- **Independent context EXL:** loses dense cross-context Hessian feedback.
- **Aggressive fixed mixed mode:** `2,2,4,4` helped only the most skewed cases
  and regressed the pilot aggregate.
- **Runtime entropy coding:** potentially useful for archival size, but not a
  candidate for the fused random-access GEMM path without a bounded decoder.

## Next decisive experiments

1. **Representative interim capture.** Run the documented large capture from
   the interim EXL3 teacher, with prompt/document-separated validation. This
   is the prerequisite for any quality claim or all-expert allocation.
2. **Pilot-to-layer scale-up.** Encode every retained expert in layers 1, 24,
   and 40 over `R0..R12`, then test whether traffic, sample support, energy skew,
   or a regularized output objective predicts the held-out winner.
3. **Coupled three-matrix modes.** Compare shared `r` with separate
   `(r13, r2)` under one common middle-neuron permutation.
4. **Real mixed packer and decoder closure.** Emit variable-K trellis records,
   repack them after importance-ordered LDLQ, decode them, and prove equality
   to the simulated reconstruction before touching a checkpoint pack.
5. **Permutation closure.** Demonstrate numerically that the common
   `w1/w3`-row and `w2`-column permutation preserves the unquantized function,
   then close the quantized CPU and CUDA paths at TP12.
6. **SM120 kernel microbenchmark.** Establish whether paired K2/K4 records
   maintain bandwidth and occupancy relative to K3/K3. Include real routing
   distributions and graph capture.
7. **Zero/skip mode.** Measure whole-16 and whole-128 exact-zero or safely
   prunable groups after permutation, accounting for permutation and mode
   overhead. Do not extrapolate from the retained-tier scan.
8. **End-to-end RDO.** Compare quality per byte and quality per millisecond,
   not just weight NMSE. Candidate selection should include exact stored size,
   kernel latency, and routed validation distortion.

The immediate gate is item 1. The pilot already rejects enough attractive
alternatives to avoid building their kernels, while giving a concrete mixed-K
format worth validating on representative interim-teacher data.

## Reproduction map

The main research scripts are:

- `scripts/experiment_codec_transfers.py`: texture block modes, low rank,
  prediction, exact EXL anchors, and activation scoring;
- `scripts/experiment_shared_rvq.py`: layer-shared RVQ and PVQ;
- `scripts/experiment_gain_vq.py`: channel-gain factorization;
- `scripts/experiment_sparse_enhancement_vq.py`: sparse residual side layer;
- `scripts/experiment_hessian_vq.py`: Hessian-derived transform VQ;
- `scripts/experiment_context_palettes.py`: uneven context palette rates;
- `scripts/experiment_context_exl3.py`: partitioned and full-H mixed-K EXL;
  and
- `scripts/scan_interim_zero_columns.py`: retained-tier exact-zero scan.

Machine-readable results are under `out/codec-transfer-*`,
`out/context-palettes-*`, `out/context-exl3-*`, and
`out/interim-retained-zero-columns.json`. These are generated research data
and must not be committed as checkpoint or correctness payloads.
