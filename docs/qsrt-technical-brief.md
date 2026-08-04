# Kimi-K3-QSRT v1 technical brief

Status: TP12 format/kernel implementation and calibration redesign, 2026-08-04.

`QSRT` means **Quantile-Stratified Rate-shifted Trellis codec**. QSRT is an
expert-static, fixed-payload mixed-rate trellis codec for gated
mixture-of-experts weights. Version 1 deliberately instantiates the general
construction for Kimi-K3 at TP12. It combines three independently useful ideas:

1. an L16 stratified-quantile graph whose transitions reconstruct finite E4M3
   values (`SQG-E4M3`);
2. equal-byte K2/K4 exchanges around K3 (`R0`, `R1`, and `R2`), selected
   separately for fused `w1`/`w3` and for `w2`; and
3. `X4T`, an exact high-quality endpoint that preserves the official MXFP4
   nibble plane and losslessly compresses its UE8M0 scale plane.

The intended artifact name is `Kimi-K3-QSRT`. The first usable checkpoint will
contain QSRT experts selected from a fresh all-expert candidate pool and X4T
experts chosen by an exact-byte global allocator. There is no raw-MXFP4 keep
tier in the final v1 storage contract.

## Frozen v1 scope

The initial production experiment is intentionally narrow:

```text
tensor-parallel geometry       TP12 only
trellis window                 L16
reconstruction family         SQG normal -> finite E4M3, RNE
lossy rate candidates          R0, R1, R2
gate/up decision               one shared r13
down-projection decision       independent r2
high-quality endpoint          exact X4T
source weights                 official Kimi-K3 MXFP4 checkpoint
calibration teacher            resident interim EXL3 checkpoint
encoder objective              expert-stratified dense-H BlockLDLQ + routed replay
```

TP4, TP16, SQG tail/zero companders, K5, wider rate ladders, learned
per-layer tables, and entropy-coded hot streams are deferred.  They are not
allowed to expand the first checkpoint's search or runtime surface.

## Encoder ownership

QSRT's offline implementation is owned by kquant. The mixed-rate dense-H
BlockLDLQ backend is `kquant/exl3_encoder_backend.py`; SQG label generation,
packed traceback, tail-biting Viterbi, and its CUDA sources live under
`kquant/sqg_e4m3.py`, `kquant/sqg_quantizer.py`, and `kquant/csrc`.

The ExLlamaV3 checkout is an unmodified upstream dependency. It supplies only
the established EXL packing, Hadamard, and tensor utilities used by the
encoder. No QSRT format, rate-selection, SQG, LDLQ, or CUDA change may be
carried as a local ExLlamaV3 patch. The exact upstream-derived source retained
in kquant is covered by `THIRD_PARTY_NOTICES.md`.

## SQG-E4M3 reconstruction

QSRT's reconstruction mechanism is the **Stratified Quantile Graph (SQG)**.
SQG assigns the $2^L$ directed edges of an $L$-bit de Bruijn trellis
bijectively to equal-probability microcells of a reference distribution. At
rate $K$, each state retains $L-K$ history bits and has $2^K$ outgoing
branches. A history-dependent branch permutation selects one branch from each
of $2^K$ coarse quantile strata, while a bijective state permutation selects
the within-stratum phase. Consequently, every state exposes exactly one
reconstruction candidate from every stratum, and every global probability
rank occurs exactly once across the directed edge set.

The graph and scalar reconstruction law are separate design objects. For a
reference distribution with quantile function $F^{-1}$, microcell $r$ spans

$$
I_r = \left[\frac{r}{2^L},\frac{r+1}{2^L}\right),
$$

and its canonical representative is the conditional mean

$$
c_r = \mathbb{E}\!\left[X\mid F(X)\in I_r\right].
$$

This value is MSE-optimal within that microcell. It is then projected with
round-to-nearest-even to finite E4M3. Numerically identical E4M3 labels may
remain on different directed edges and lead to different successors; scalar
label collisions therefore do not collapse the richer trellis geometry.

The rank-to-E4M3 law may be implemented with an E4M3-aware piecewise Chebyshev
approximation. Instead of minimizing real-valued approximation error, its
coefficients are constrained so finite-arithmetic evaluation lands inside the
rounding interval of the desired E4M3 label at every discrete rank. Exhaustive
validation over all $2^L$ ranks then proves label identity. This provides a
compact arithmetic decoder without a 65,536-entry table.

For Kimi-K3, $L=16$. The current R44 labeler and the SQG-Cheb labeler are
explicitly different implementation candidates for the same frozen SQG graph;
they must not be conflated with changes to the transition topology. Bit-exact
rank mixing and label generation live in `kquant/sqg_e4m3.py`, and encoder
integration lives in `kquant/sqg_quantizer.py`.

## Fixed-payload rate shifting

The common 3,072-neuron intermediate axis is divided into 24 records of 128
neurons.  Each record retains the existing 16x16 coding tiles.  A single
function-preserving permutation is applied as

```text
W1' = P W1
W3' = P W3
W2' = W2 P^T.
```

Because SiTU is coordinatewise, this changes neither the expert function nor
the coordinate presented at the nonlinear boundary.  It makes importance
regions contiguous and makes the record rate derivable from a small mode ID,
without a per-channel rate map or runtime shuffle.

For a matrix family, mode `Rr` assigns

```text
first r records       K2
middle 24 - 2r        K3
last r records        K4.
```

Thus every mode averages exactly three path bits per weight:

```text
R0 =  0 K2 + 24 K3 + 0 K4
R1 =  1 K2 + 22 K3 + 1 K4
R2 =  2 K2 + 20 K3 + 2 K4
```

QSRT redistributes rate over paired 128-channel records without changing
payload size. A `P24` container assigns K2 to a low-priority donor record and
K4 to a high-priority recipient, while a `P33` container assigns K3 to both.
Each consumes six trellis bits per coefficient pair and occupies the same
physical size. Pair ownership is rotated by layer and expert so any P24/P33
execution imbalance is distributed across the 12 ranks.

The v1 mode is `(r13, r2)`. `w1` and `w3` share `r13` for fused execution;
`w2` selects `r2` independently. The common physical neuron permutation does
not require the three matrices to share a rate schedule.

## Dense-H encoding and statistical selection

Cheap importance scores only propose the permutation and donor/recipient
records. They do not authorize a rate shift. Down-projection candidates are
evaluated through complete dense-$H$ BlockLDLQ re-encodes so cross-record
covariance feedback is retained. For `w1`/`w3`, the common input covariance is
retained while the selected output-row records receive their assigned rates.

The encoder then reconstructs the full expert and scores applied-gate-square
weighted routed output error on document-disjoint samples.  A nonzero mode is
accepted only when its paired document-bootstrap lower confidence bound clears
the frozen improvement margin over matched SQG `R0`; uncertain experts fall
back to `(R0,R0)`.

The initial search evaluates only the 3x3 Cartesian grid

```text
(r13, r2) in {0,1,2} x {0,1,2}.
```

This keeps the all-expert encode operationally viable while retaining the
independent `w2` decisions that earlier studies showed were important. Modes
are expert-static: serving reads a compact format code and never performs
runtime rate selection.

## X4T exact endpoint

Official MXFP4 uses four E2M1 bits per weight plus one UE8M0 scale byte per 32
weights, or 4.25 bpw.  X4T changes no represented value:

- the packed E2M1 nibble plane is copied byte-for-byte, including both zero
  codes;
- each scale row chooses the adjacent UE8M0 pair that covers the most entries;
- every 16-row slab stores 16 base bytes followed by fixed-stride selector
  bits, with values outside the chosen pair carried by a sorted uint32
  exception stream; and
- decoding recovers both official tensors exactly before ordinary TP12
  sharding and W4A16 preparation.

The selector stream is directly indexable and needs no tile offset table,
prefix sum, or exception search.  X4T is therefore the v1 high-quality
endpoint; uniform K4 remains lossy and is not treated as a substitute for the
official weights.  The all-expert X4T index records exact aligned bytes for
each expert rather than relying on a nominal bpw estimate.

The frozen sidecar representation uses:

```text
one sparse sidecar per MoE layer
4 KiB canonical header
64 KiB fixed expert/matrix directory
24-byte directory entry per (expert, matrix)
full-matrix records in expert-major w1, w3, w2 order
64-byte record header
raw nibble bytes + compact scale payload
4 KiB record alignment
CRC32 for the directory and every record
strict zero padding and canonical re-encode validation
```

Full-matrix coding is intentional: compressing after TP12 sharding would throw
away scale-plane context, especially for the eight-column local `w2` scale
slices.  The container is decoded into TP12 operand slices only after the
full-matrix record has been recovered.

The reference record/container implementation is `kquant/x4t.py`; the scale
codec is `kquant/mxfp4_scale_codec.py`.

### X4T runtime refinement

The compressed X4T scale planes can remain persistent in device memory rather
than being expanded for every expert at model initialization.  Immediately
before the ordinary W4A16 call, one graph-safe launch expands only the routed
experts into a caller-owned packed-scale scratch buffer.  That scratch is
reused across layers on the same stream; there is no per-call allocation, CPU
parsing, prefix scan, exception search, or disk access.

The TP12 implementation exactly reproduces the active packed W4A16 scale
bytes, folds the fused-`w1`/`w3` row rotation and BF16 E8M0 clamp into the same
launch, and survives scratch poisoning followed by CUDA graph replay.  On an
RTX PRO 6000 Blackwell Max-Q, a balanced 1,000-replay synthetic Kimi-K3 M=1
study measured the following complete routed-MoE costs:

| Active X4T experts | Dense W4A16 | X4T + W4A16 | Added latency |
| ---: | ---: | ---: | ---: |
| 1 | 22.08 us | 24.16 us | 2.08 us |
| 2 | 22.11 us | 26.11 us | 4.00 us |
| 4 | 22.11 us | 26.21 us | 4.10 us |
| 8 | 24.16 us | 28.26 us | 4.10 us |
| 16 | 30.30 us | 32.35 us | 2.05 us |

M=2 and M=4 sweeps also closed exact scale-byte reconstruction; their added
latency ranged from 1.25 to 8.19 us depending on routed density.  Output
differences once four or more experts contribute match the dense kernel's own
repeatability envelope and come from nondeterministic atomic accumulation,
not scale decode.  The benchmark is
`b12x/benchmarks/benchmark_x4t_w4a16_moe_tp12.py`.

The runtime result clears the latency plausibility gate.  The remaining X4T
work for this checkpoint is to build its own all-expert exact-byte index and
rerun the routed benchmark with checkpoint-derived selections.

## Global allocation

Rate shifting and high-tier selection solve different problems.

1. For each expert, the candidate pool freezes the statistically selected
   `(r13,r2)` at the same three-bit trellis payload.
2. X4T then competes against that selected lossy candidate.  Promoting an
   expert removes its measured routed damage and incurs that expert's exact X4T
   record bytes rather than a fixed nominal four-bit cost.

The v1 byte cap is the logical TP12 expert-container size derived from the
validated 3p09 allocation:

```text
target container bytes = 1,058,586,247,168
```

This is the schema-level payload/alignment budget used by both allocation
formats, rather than `du` output that also includes serializer headers.  The
global allocator minimizes

```text
sum_e D_e(choice_e) + lambda * sum_e bytes_e(choice_e)
```

and sweeps `lambda` to meet the checkpoint budget.  Since X4T sizes vary by
expert, the old fixed-count top-damage rule is not the final QSRT allocator.
Candidate generation and X4T cost indexing are reusable, so changing the target
budget does not require another trellis encode.

An additive solution supplies the initial frontier.  A later layerwise routed-
mixture replay may refine borderline choices to account for error cancellation
or reinforcement among co-routed experts.

## Evidence and current quality blocker

The initial production-path SQG study used 24 official-source experts across
layers 1, 24, and 40. At fixed K2/K3/K4 endpoints, SQG normal beat both
MUL1-E4M3 and FP16 MCG for all 24 experts and all 216 matrix/rate comparisons.
Those results established SQG as a serious candidate, but they are now
hypothesis-forming rather than a production quality gate: their `w2` metric
used a layer-global post-SiTU covariance whose coordinate indices are not
shared across independently permuted experts.

The matched R0/R1/R2 gate on the same panel found:

```text
SQG selected nonzero r13       6 / 24 experts
SQG selected nonzero r2        2 / 24 experts
SQG proposed nonzero r13       8 / 24 experts
SQG proposed nonzero r2        5 / 24 experts
aggregate SQG R0 vs MUL1 R0    2.2443% lower confirmation SSE
aggregate SQG selected vs
  MUL1 selected                2.1071% lower confirmation SSE
```

The small 21-document confirmation fold was never a final model-quality claim.
It did establish that SQG survives the real Hadamard, LDLQ, Viterbi,
official-weight, and routed-replay path, and that independent `w2` selection
can remain active. It did not establish that the captured Hessian geometry was
representative enough for a checkpoint. The resulting R44/X4T artifact failed
the expected quality trajectory, and its generation path has been stopped.

The replacement gate begins with a source-controlled one-million-token
training capture. `H13` remains layer-global because its latent input basis is
shared. `H2` is rebuilt from expert-stratified routed post-SiTU rows, shrunk
toward identity according to support, and falls back to identity for
unsupported experts. Mode-selection and final-validation corpora remain
document-disjoint. No old R44 candidate pool is eligible for the next
checkpoint merely because it is complete.

The exact SQG decoder has also crossed the representative TP12 runtime gate.
The production B12X W4A16 path passed dense K2/K3/K4 GEMMs, both P24 and P33
pair orientations, and a dynamic fused SiTU MoE containing both pair types,
including CUDA graph replay.  This establishes numerical/runtime closure of
the codec primitive; it is not yet an end-to-end checkpoint latency result.

The native compute path is split deliberately.  The general W4A16 path keeps
BF16/FP16 activations and widens the exact E4M3 reconstruction before MMA; it
is the correctness fallback.  The W4A8 path converts each Hadamard-domain
activation block to E4M3 plus UE8M0 and feeds SQG's E4M3 weights directly to
SM120 block-scaled MMA.

The production W4A8 decoder uses exact, process-global execution tables rather
than checkpoint metadata.  For one token (16 routed rows), a 106 KiB
asymmetric table combines direct K3 labels with packed K2/K4 phase/syndrome
states.  At higher route concurrency, a 58 KiB table stores the packed state
for all K2/K3/K4 histories plus the 2 KiB rank-to-E4M3 staircase.  The host
selects between them from the already-known route shape, without inspecting
routing decisions or synchronizing the GPU.  Both reproduce all SQG labels
bit-for-bit, are shared by every layer and expert, use no checkpoint bytes,
and avoid the per-CTA 20--32 KiB shared-memory codebooks considered earlier.

On synthetic TP12 routed mixtures with 16 experts and CUDA graph replay, the
complete W4A8 MoE path measured 167--320 us for 1, 2, and 4 tokens across
P33, P24, sparse, and mixed pair-mode fixtures.  Matched W4A16 measured
600--759 us, so W4A8 took 27.9--44.7% of the time (2.24--3.58x faster).
Relative to the same SQG weights on W4A16, activation quantization produced
0.199--0.222% output NMSE with cosine similarity 0.998888--0.999007.  The
dynamic separate-`r13`/`r2` route test and CUDA graph replay pass.  These
measurements close the synthetic kernel gate; checkpoint-derived route
fixtures and end-to-end model latency remain pending.

## SQG-Cheb refinement branch

An investigator-proposed refinement replaces the clipped `R44` normal
compander with a dyadically range-reduced Chebyshev evaluator synthesized
against the rounding intervals of the final finite-E4M3 labels.  This is a
sound implementation technique: the polynomial only has to land inside the
Voronoi interval that rounds to the intended E4M3 byte, rather than minimize
real-valued inverse-CDF error.  Exhaustive evaluation of the supplied Q24/Q31
implementation reproduced every intended L16 normal label.

The coding experiment freezes a stricter contract than the investigator's
full proposal:

```text
rates                         K2, K3, K4 only
history and syndrome mixers   original SQG constants
branch permutation            original SQG rule
phase rule                    original baseline phase for every rate
rank reconstruction law       one shared normal staircase for every rate
rate-specific phase edits     prohibited
K5                            out of scope
```

The supplied K3 and K4 normal banks produce the same rank-indexed 65,536-byte
E4M3 staircase.  Applying that one staircase after the existing rank
permutation therefore defines K2 without inventing K2-specific graph or phase
semantics.  Relative to clipped R44, 597 of 65,536 rank labels change; the
important difference is restored tail resolution.  The Chebyshev polynomial
is a compact way to compute that staircase, while the numerical gain comes
from the changed full-tail reconstruction law rather than from polynomial
evaluation by itself.

A controlled production-path study compared the two staircases on the same
24 official-source experts across layers 1, 24, and 40.  It retained the same
Hadamard transforms, expert-specific scales, captured dense Hessians,
BlockLDLQ traversal, Viterbi graph, and document-disjoint routed replay:

| Rate | Dense-H SSE change | Validation routed SSE change | Validation wins |
| --- | ---: | ---: | ---: |
| K2 | +0.013% | -0.197% | 9 / 24 |
| K3 | -0.193% | -0.276% | 17 / 24 |
| K4 | -0.685% | -0.811% | 22 / 24 |

The K4 result is robust enough to remain the leading refinement.  K3 is
promising.  K2 is not a safe universal replacement: its traffic-weighted
validation aggregate improves, but its median expert regresses by 0.147% and
only 9 of 24 experts improve.  The K2/K3/K4 donor-to-recipient curvature is
essentially unchanged, so the normal Chebyshev law does not by itself create
a stronger rate-transfer frontier.

Accordingly, the sealed all-expert v1 pool remains immutable and continues
using `sqg-l16-normal-r44-v1`.  A later candidate pool may test clipped-R44 at
K2 with SQG-Cheb at K3/K4, or a small shared source-shape law family, but only
after full rate-shift counterfactual validation.  The provisional
normal/mild/spike/zero bank is not frozen: it was tuned on synthetic sources,
has no supplied K2 design, and adds a mode-selection problem vulnerable to
winner's curse.  Any such family must use the same graph and phase rule at all
rates.

## Execution checklist

- [x] Implement and unit-test L16 SQG-normal E4M3 labels for K2/K3/K4.
- [x] Integrate SQG into dense-H rate-shifted encoding and stored-state decode.
- [x] Validate SQG endpoints and the separate `(r13,r2)` R0/R1/R2 gate.
- [x] Start the resumable all-82,432-expert SQG candidate pool on 12 GPUs.
- [x] Complete and seal the all-82,432-expert R44 SQG candidate pool.
- [x] Freeze and unit-test exact X4T matrix records and sparse layer sidecars.
- [x] Add exact X4T load-time reconstruction and TP12 W4A16 preparation.
- [x] Implement exact fixed-stride X4T records and the one-launch, graph-safe
      routed TP12 W4A16 scale predecoder.
- [x] Benchmark X4T inside the complete routed W4A16 path across M=1/2/4 and
      1/2/4/8/16 active-expert densities.
- [ ] Build and seal the all-expert X4T byte-cost index.
- [x] Close representative SQG K2/K3/K4, P24/P33, fused-SiTU, and graph-replay
      execution through the production B12X API.
- [x] Close native SQG W4A8 dense/routed execution and measure its full-path
      activation-quantization error and latency against matched W4A16.
- [x] Seal the 223-document, 128K-token document-disjoint validation capture.
- [x] Validate the fixed-graph SQG-Cheb normal staircase at K2/K3/K4 on the
      24-expert production panel; retain it as a refinement branch rather than
      mutating the sealed all-expert pool.
- [x] Complete the synthetic unified K2/K3/K4 reconstruction-law study defined in
      [`sqg-unified-k234-investigator-brief.md`](sqg-unified-k234-investigator-brief.md),
      with K2 included during fitting and selection rather than extrapolated
      from K3/K4.
- [ ] Validate the shared cubic reconstruction law on the frozen production
      K2/K3/K4 path after this R44/X4T checkpoint closes.
- [ ] Score selected candidates on the untouched validation capture and run
      the matched-R0 rate-shift policy audit.
- [ ] Freeze the global QSRT allocation at the target checkpoint budget.
- [ ] Materialize compressed slabs plus selected X4T sidecars into a fresh
      artifact.
- [ ] Close the materialized artifact's structural validation, malformed-input
      rejection, exact state decode, exact X4T source reconstruction, and exact
      byte accounting.
- [ ] Re-run the TP12 kernel/performance gate on checkpoint-derived routed
      mixtures; the synthetic P33/P24/sparse/mixed gate has passed.
- [ ] Package a fresh serve directory; never mutate the validated 3p09 model.
- [ ] Run streamed official-vs-packaged traces, live TP12 routing/logit checks,
      and the expanded end-to-end quality suite.

The later evaluation suite should incorporate the 32x2048 KLD reference
dataset and the Kimi-K3 evaluation tools identified for final end-to-end
testing.  The current calibration/confirmation corpus must also be expanded
substantially before production quality claims are made.
