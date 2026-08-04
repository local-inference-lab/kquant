# Kimi-K3-SQRT-C v1 technical brief

Status: implementation and all-expert candidate generation, 2026-08-03.

`SQRT-C` means **Stratified Quantile Rate-Shifted Trellis Codec**.  Version 1
is deliberately a Kimi-K3 TP12 format.  It combines three independently useful
ideas:

1. an L16 stratified-quantile graph whose transitions reconstruct finite E4M3
   values (`SQG-E4M3`);
2. equal-byte K2/K4 exchanges around K3 (`R0`, `R1`, and `R2`), selected
   separately for fused `w1`/`w3` and for `w2`; and
3. `X4`, an exact high-quality endpoint that preserves the official MXFP4
   nibble plane and losslessly compresses its UE8M0 scale plane.

The intended artifact name is `Kimi-K3-SQRT-C`.  The first usable checkpoint
will contain SQRT-C experts selected from the all-expert candidate pool and X4
experts chosen by an exact-byte global allocator.  There is no raw-MXFP4 keep
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
high-quality endpoint          exact X4
source weights                 official Kimi-K3 MXFP4 checkpoint
calibration teacher            resident interim EXL3 checkpoint
encoder objective              captured dense-H BlockLDLQ + routed replay
```

TP4, TP16, SQG tail/zero companders, K5, wider rate ladders, learned
per-layer tables, and entropy-coded hot streams are deferred.  They are not
allowed to expand the first checkpoint's search or runtime surface.

## SQG-E4M3 reconstruction

For rate `K`, an L16 trellis edge consists of the retained `16-K` history bits
and the `K` new branch bits.  SQG separates the two jobs that an edge label
must perform:

- the new bits select one of `2^K` coarse quantile strata immediately;
- a bijective history mixer chooses a fine phase within the selected stratum;
- a separately mixed syndrome permutes branch-to-stratum assignments, changing
  successor geometry without degrading the current state's scalar menu.

Before E4M3 projection, the mapping from all 65,536 codewords to quantile rank
is a permutation.  For every state, all outgoing branches are distinct before
projection and cover every coarse stratum.  The normal compander then maps the
rank through the frozen rational inverse-normal approximation and rounds the
result to finite E4M3.

The bit-exact constants and implementation live in
`kquant/sqg_e4m3.py`; the encoder integration lives in
`kquant/sqg_quantizer.py`.  The format identifies this construction as
`sqg-l16-normal-r44-v1`.  No learned table or per-layer codebook is required.

This is not merely a new scalar palette.  The stratification improves the
instantaneous outgoing menu, while the phase and syndrome preserve useful
multi-step Viterbi routing.  The distinction explains why constructions with
excellent one-step quantization can still perform poorly as trellis codes.

## Rate shifting

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

Two 128-neuron records form one fixed-size pair.  `P24` contains K2 plus K4;
`P33` contains K3 plus K3.  Both carry six path bits per pair coefficient, so
the physical payload remains constant-stride.  Pair ownership is rotated by
layer and expert so any P24/P33 execution imbalance is distributed across the
12 ranks.

The v1 mode is `(r13, r2)`.  `w1` and `w3` share `r13` to retain a regular
fused gate/up path; `w2` selects `r2` independently.  The common physical
neuron permutation does not require the matrices to share their rate schedule.

## Dense-H encoding and statistical selection

Cheap importance scores only propose the permutation and donor/recipient
records.  They do not authorize a rate shift.  Every candidate is encoded as
a complete counterfactual using the selected K at each record while preserving
the dense covariance and BlockLDLQ error-feedback traversal.

The encoder then reconstructs the full expert and scores applied-gate-square
weighted routed output error on document-disjoint samples.  A nonzero mode is
accepted only when its paired document-bootstrap lower confidence bound clears
the frozen improvement margin over matched SQG `R0`; uncertain experts fall
back to `(R0,R0)`.

The initial search evaluates only the 3x3 Cartesian grid

```text
(r13, r2) in {0,1,2} x {0,1,2}.
```

This keeps the full all-expert encode operationally viable while retaining the
independent `w2` decisions that earlier studies showed were important.

## X4 exact endpoint

Official MXFP4 uses four E2M1 bits per weight plus one UE8M0 scale byte per 32
weights, or 4.25 bpw.  X4 changes no represented value:

- the packed E2M1 nibble plane is copied byte-for-byte, including both zero
  codes;
- the complete, unsharded UE8M0 scale plane is coded with per-row palettes,
  adjacent-exponent direction bits, selectors, sparse exceptions, and
  independently decodable 16-row offsets;
- decoding recovers both official tensors exactly before ordinary TP12
  sharding and W4A16 preparation.

Across all experts in layers 1, 24, and 40, the compact scale codec measured
4.03819 bpw in aggregate with exact reconstruction.  Uniform K4 trellis was
slightly smaller at 4.00967 bpw but retained 0.275955% routed NMSE and was off
the K3/X4 lower convex hull for every supported layer-24 expert.  X4 is
therefore the v1 high-quality endpoint; K4 is not treated as a substitute for
the official weights.

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
slices.  X4 is an on-disk encoding; after load, the runtime operand remains the
ordinary production MXFP4 representation.

The reference record/container implementation is `kquant/x4.py`; the scale
codec is `kquant/mxfp4_scale_codec.py`.

### X4T runtime refinement

`X4T` is the GPU-tile-friendly exact alternative to fully expanding X4 at
model load.  It still keeps every official E2M1 nibble unchanged, but stores
each 16-row UE8M0 slab as adjacent per-row bases, fixed-stride selector bits,
and a sorted sparse exception stream.  The compressed scale planes stay
persistent in device memory.  Immediately before the ordinary W4A16 call, one
graph-safe launch expands only the routed experts into a caller-owned packed
scale scratch buffer.  That scratch is reusable across layers on the same
stream; there is no per-call allocation, CPU parsing, prefix scan, or disk
access.

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

X4T is not yet interchangeable with the existing X4 allocation ledger: its
exact serialized byte costs must be indexed over all experts first.  The
runtime result clears the latency plausibility gate; the storage frontier and
checkpoint-derived routed benchmark remain the final choice between fully
expanded X4 and persistent X4T.

## Global allocation

Rate shifting and high-tier selection solve different problems.

1. For each expert, the candidate pool freezes the statistically selected
   `(r13,r2)` at the same three-bit trellis payload.
2. X4 then competes against that selected lossy candidate.  Promoting an
   expert removes its measured routed damage and incurs that expert's exact X4
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

and sweeps `lambda` to meet the checkpoint budget.  Since X4 sizes vary by
expert, the old fixed-count top-damage rule is not the final SQRT-C allocator.
Candidate generation and X4 cost indexing are reusable, so changing the target
budget does not require another trellis encode.

An additive solution supplies the initial frontier.  A later layerwise routed-
mixture replay may refine borderline choices to account for error cancellation
or reinforcement among co-routed experts.

## Evidence at the v1 freeze point

The production-path SQG study used 24 official-source experts across layers 1,
24, and 40.  At fixed K2/K3/K4 endpoints, SQG normal beat both MUL1-E4M3 and
FP16 MCG for all 24 experts and all 216 matrix/rate dense-H comparisons.

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

The small 21-document confirmation fold is deliberately conservative and is
not the final model-quality claim.  It does establish that SQG survives the
real Hadamard, dense-H, LDLQ, Viterbi, official-weight, and routed-replay path;
it also confirms that independent `w2` selection remains active.  A separate
128K-token validation capture is now sealed with 226 whole documents and no
document-hash overlap with the training capture.  It remains untouched until
the all-expert pool is sealed, at which point it supplies both the X4 damage
ranking and the matched-R0 policy audit.

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

Accordingly, the running all-expert v1 pool remains immutable and continues
using `sqg-l16-normal-r44-v1`.  A later candidate pool may test clipped-R44 at
K2 with SQG-Cheb at K3/K4, or a small shared source-shape law family, but only
after full mixed-mode counterfactual validation.  The provisional
normal/mild/spike/zero bank is not frozen: it was tuned on synthetic sources,
has no supplied K2 design, and adds a mode-selection problem vulnerable to
winner's curse.  Any such family must use the same graph and phase rule at all
rates.

## Execution checklist

- [x] Implement and unit-test L16 SQG-normal E4M3 labels for K2/K3/K4.
- [x] Integrate SQG into dense-H mixed-rate encoding and stored-state decode.
- [x] Validate SQG endpoints and the separate `(r13,r2)` R0/R1/R2 gate.
- [x] Start the resumable all-82,432-expert SQG candidate pool on 12 GPUs.
- [x] Freeze and unit-test exact X4 matrix records and sparse layer sidecars.
- [x] Build and seal the all-expert exact X4 byte-cost index.
- [x] Add exact X4 load-time reconstruction and TP12 W4A16 preparation.
- [x] Implement exact fixed-stride X4T records and the one-launch, graph-safe
      routed TP12 W4A16 scale predecoder.
- [x] Benchmark X4T inside the complete routed W4A16 path across M=1/2/4 and
      1/2/4/8/16 active-expert densities.
- [ ] Build the all-expert X4T byte-cost index and compare its storage frontier
      with X4 before freezing the exact endpoint representation.
- [x] Close representative SQG K2/K3/K4, P24/P33, fused-SiTU, and graph-replay
      execution through the production B12X API.
- [x] Close native SQG W4A8 dense/routed execution and measure its full-path
      activation-quantization error and latency against matched W4A16.
- [x] Seal the 226-document, 128K-token document-disjoint validation capture.
- [x] Validate the fixed-graph SQG-Cheb normal staircase at K2/K3/K4 on the
      24-expert production panel; retain it as a refinement branch rather than
      mutating the running all-expert pool.
- [ ] Complete the unified K2/K3/K4 reconstruction-law study defined in
      [`sqg-unified-k234-investigator-brief.md`](sqg-unified-k234-investigator-brief.md),
      with K2 included during fitting and selection rather than extrapolated
      from K3/K4.
- [ ] Complete and seal the all-expert SQG candidate pool.
- [ ] Score selected candidates on the untouched validation capture and run
      the matched-R0 mixed-mode policy audit.
- [ ] Freeze the global SQRT-C allocation at the target checkpoint budget.
- [ ] Materialize compressed slabs plus selected X4 sidecars into a fresh
      artifact.
- [ ] Close the materialized artifact's structural validation, malformed-input
      rejection, exact state decode, exact X4 source reconstruction, and exact
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
