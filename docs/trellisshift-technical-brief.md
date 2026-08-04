# TrellisShift (TSH): fixed-budget mixed-rate trellis quantization

Status: phase-1 implementation plus adopted X4 successor direction,
2026-08-03. The initial implementation and runtime format target Kimi-K3 at
TP12. End-to-end quality validation is still in progress.

## Summary

**TrellisShift (TSH)** is an expert-static mixed-rate codec for routed MoE
weights. It replaces uniform K3 trellis quantization with an unequal allocation
of K2, K3, and K4 records while holding every compressed expert at the same
nominal three trellis bits per weight.

The encoder removes one bit from relatively insensitive intermediate-neuron
records and gives that bit to more sensitive records. A shared, exactly
function-preserving neuron permutation makes those records contiguous across
the gate, up, and down projections. Dense activation-Hessian LDLQ performs the
actual error-aware rounding, Viterbi supplies the trellis code at each chosen
K, and a document-disjoint routed-function test decides whether a shift is
accepted. The result is a fixed-stride payload with a small per-expert mode,
not an irregular per-channel bit map.

The current production candidate uses the 16-bit EXL state graph with the
procedural MUL1 state hash (`0x83DCD12D`) and rounds each reconstructed value
to E4M3. It has no learned reconstruction table. This makes the compressed
weights suitable for a direct E4M3 block-scaled MMA path while retaining the
same K2/K3/K4 path syntax used by TSH.

`Kimi-K3-TSH` names a checkpoint whose routed-expert representation is TSH.
The in-flight phase-1 checkpoint still has an MXFP4 keep tier so that it can
be completed and validated without restarting its all-expert encode. The
successor target replaces that raw 4.25-bpw keep representation with `X4`, an
exact index-plus-compressed-scale representation of the same MXFP4 weights at
about 4.038 bpw. It retains TSH for the lossy tier and allocates X4 only where
its measured zero-distortion endpoint earns the additional bytes.

## Adopted TSH/X4 rate-control target

The production rate-control frontier has two representation families:

1. mixed-`R` K3 TSH for the lossy tier; and
2. exact `X4` for the high-quality tier.

Ordinary MXFP4 stores one four-bit E2M1 index per weight and one eight-bit
UE8M0 scale per 32 weights, for 4.25 bpw. X4 leaves every E2M1 index unchanged
and losslessly codes the redundant scale plane with a row palette, one-bit
selectors, sparse exceptions, and independently decodable 16-row offsets.
Across every expert in layers 1, 24, and 40 it reconstructed the official
MXFP4 values bit-for-bit at 4.03819 bpw aggregate.

On the held-out all-expert layer-24 study, uniform K4 TSH occupied 4.00967 bpw
but retained 0.275955% routed-expert NMSE. X4 cost only another 0.02820 bpw and
had zero source-relative distortion. K4 was off the K3/X4 lower convex hull
for all 887 supported experts under both confirmation and validation routed
SSE. The successor allocator must therefore compare each selected TSH
candidate with X4 using exact stored bytes; uniform K4 is not a production
high endpoint.

This decision does not alter the already-running phase-1 candidate pool. That
artifact remains useful and retains raw MXFP4 exactly as frozen. X4 enters the
successor schema only after its TP12 materialization, decoder, and performance
gates close.

## Optional all-TSH research branch

An all-TSH curve spanning three through four trellis bits remains a systems
experiment, not the adopted quality frontier. It is worth pursuing only if a
single TSH serving representation produces enough measured simplicity or
latency benefit to justify falling below the K3/X4 rate--distortion hull.

For one 24-record matrix, let `B` be the number of path bits added to uniform
K3, where `0 <= B <= 24`. A legal schedule has record counts
`(n2, n3, n4, n5)` satisfying

```text
n2 + n3 + n4 + n5 = 24
2*n2 + 3*n3 + 4*n4 + 5*n5 = 72 + B.
```

Its nominal mean is therefore `3 + B/24` trellis bits per weight. Records stay
ordered from low to high importance, so a mode is a few segment counts rather
than an arbitrary 24-entry rate map. `w1` and `w3` share their schedule while
`w2` chooses independently; the whole-expert mean is weighted by the two
gate/up matrices and one down matrix.

The two endpoint transfer families are:

```text
three-bit endpoint:  r*K2 + (24 - 2r)*K3 + r*K4
four-bit endpoint:   r*K3 + (24 - 2r)*K4 + r*K5
```

Thus four-bit `R1`, `R2`, and `R3` are K3/K5 exchanges around uniform K4.
They do not save bytes relative to uniform K4; they test whether moving bits
from insensitive records to sensitive records reduces error at the same
four-bit path budget. At intermediate budgets the encoder searches a compact
set of legal count tuples, prunes dominated candidates, and authorizes a mode
only with a complete dense-H encode and routed replay.

The four-bit experimental endpoint is a real transformed E4M3 trellis encode,
not an MXFP4-nibble bypass. K5 is useful only if a K4-to-K5 recipient gain
exceeds the K4-to-K3 donor cost after sensitivity and dense-H coupling are
included. Any such candidate must be compared against X4, not merely against
raw 4.25-bpw MXFP4.

For the adopted path, global allocation selects either the validated TSH
candidate or exact X4 for each expert by minimizing `D + lambda * bytes` and
sweeps `lambda` to hit the checkpoint budget. This replaces the old blind
top-7,007 raw-MXFP4 keep decision with a measured, exact-byte frontier across
all 82,432 experts.

## Core construction

A Kimi-K3 routed expert has

```text
w1: [3072, 3584]
w3: [3072, 3584]
w2: [3584, 3072]
```

The shared 3072-dimensional intermediate axis consists of matching `w1` and
`w3` output rows, post-SiTU coordinates, and `w2` input columns. TSH divides
that axis into 24 records of 128 neurons. Each record contains ordinary 16x16
trellis coding tiles; 16x16 is the coding unit, not a restriction on the MMA
atom used by the decoder.

The records are ordered from low to high estimated importance. Mode `R_r`
assigns

```text
first r records       -> K2
middle 24 - 2r        -> K3
last r records        -> K4
```

so the K2 and K4 counts are equal. Every `R_r` therefore averages exactly K3:

```text
R0:  0 K2 + 24 K3 + 0 K4
R1:  1 K2 + 22 K3 + 1 K4
R2:  2 K2 + 20 K3 + 2 K4
R3:  3 K2 + 18 K3 + 3 K4
R4:  4 K2 + 16 K3 + 4 K4
R5:  5 K2 + 14 K3 + 5 K4
```

The current all-expert build searches the complete Cartesian `R0` through
`R5` grid. `w1` and `w3` share one rate-transfer count, `r13`, while `w2`
chooses `r2` independently. Thus an expert format is written `TSH(r13, r2)`,
such as `TSH(0, 5)`. All three matrices still use one physical neuron order;
separate rate schedules do not introduce a gather at the SiTU boundary.

## Exact shared-neuron permutation

For a permutation matrix `P`, TSH stores

```text
W1' = P W1
W3' = P W3
W2' = W2 P^T.
```

Because SiTU is coordinatewise,

```text
SiTU(Pg, Pu) = P SiTU(g, u),
```

and therefore

```text
W2' SiTU(W1' x, W3' x) = W2 SiTU(W1 x, W3 x).
```

The permutation is consequently part of the stored expert, not an
approximation or runtime shuffle. Its main benefit is structural: similarly
ranked neurons become whole rate records, record K is derived from the expert
mode and record index, TP shards own complete records, and the decoder needs
no per-channel rate metadata.

The existing EXL Hadamard/sign transforms are a separate gauge. They are
cancelled at the appropriate linear boundaries so SiTU operates in canonical
coordinates; only the plain permutation above spans the three semantic
matrices.

## Hessian-aware encoding and selection

TSH does not quantize K2, K3, and K4 partitions as unrelated matrices. For
`w2`, the full transformed dense covariance and BlockLDLQ feedback are
preserved across the heterogeneous-rate traversal. Changing one record can
change the compensation applied to later records, so a rate shift is judged
from a complete counterfactual encode rather than from additive per-record
error estimates.

The phase-1 encoder operates as follows:

1. Capture routes, applied gates, expert inputs, and post-SiTU activations from
   the resident interim EXL3 teacher. Stream official checkpoint weights as
   the source to be encoded.
2. On fit documents, rank four-neuron groups by gate-square-weighted
   official-source post-SiTU energy, then form 24 balanced 128-neuron records.
3. Build expert Hessian approximations with fixed shrinkage. `H13` is 25%
   expert-local input covariance and 75% layer-global covariance; `H2` is 75%
   expert-local post-SiTU covariance and 25% layer-global covariance.
4. Encode the Cartesian `R0`--`R5` grid for `(r13, r2)` with dense-H LDLQ and
   the K-specific Viterbi trellis quantizer. Reconstruction uses the fixed
   L16 MUL1-to-E4M3 codebook function for every rate.
5. Reconstruct the complete expert function and measure applied-gate-square
   weighted output SSE against the decoded official source expert on
   document-disjoint confirmation data.
6. Accept the best nonzero mode only when its paired document-bootstrap lower
   confidence bound beats `TSH(0, 0)`; otherwise fall back to uniform K3.
   Experts with inadequate support also fall back to `TSH(0, 0)`.

Mode selection is entirely offline and static. It is not recomputed per token
or per request. A separate untouched capture is reserved for external
validation of the completed pool.

## Fixed TP12 payload

Two 128-neuron records form one 256-neuron pair:

```text
P24 = one K2 record + one K4 record
P33 = two K3 records.
```

Both pair types contain exactly the same trellis payload: 344,064 bytes per
pair in each Kimi-K3 expert matrix. Each matrix contains twelve pairs and every
TP12 rank owns one pair, so mixed rate retains constant-stride addressing and
equal stored bytes. A deterministic layer/expert rotation distributes P24
ownership across physical ranks. The provisional format uses one byte per
expert: a four-bit `r13`, a four-bit `r2`, and `0xff` for retained MXFP4.

This arrangement is the codec's key systems property: the encoder can make a
meaningful rate-distortion choice while the serving kernel sees only two legal
fixed-size decode cases. There are no offset streams, entropy parsing,
per-channel K tables, or runtime neuron permutations.

The reference serving implementation has exact K2/K3/K4 MUL1-to-E4M3 decode
closure. A compatibility path widens those values for the existing W4A16 MMA;
the throughput target instead keeps the decoded bytes in E4M3 and feeds an
SM120 block-scaled MMA directly. Sparse EXL GEMV measurements favor MUL1 over
MCG, but the native E4M3 fused-MoE path still requires its own TP12 benchmark;
the sparse-kernel ratio is not assumed to transfer unchanged.

## Relationship to model-wide rate allocation

TSH answers the within-compressed-tier question: **where should a fixed K3
budget be spent inside this expert?** It does not decide which experts deserve
more than three bits on average.

After all 82,432 routed experts have TSH candidates, the global allocator
compares each candidate with its official MXFP4 expert using routed functional
damage and exact stored bytes. It then chooses the MXFP4 keep tier at the
target checkpoint budget. Because candidate generation precedes allocation,
the keep fraction can change without requantizing every expert.

## Preliminary signal and remaining gates

Across the first 10,792 atomically sealed assignments in the current
MUL1-E4M3 build, the conservative selector accepted a nonuniform format for
1,067 experts (9.89%). It selected nonzero `r13` for 1,042 and nonzero `r2`
for 547; the `r2` selections include 98 `R4` and 152 `R5` cases. The median
confirmation-fold relative improvement among accepted experts is 2.52% (mean
5.08%). This is an early, scheduler-ordered slice rather than a random sample,
and confirmation data are part of mode selection rather than the untouched
external validation set. The result establishes that separate decisions and
deep `w2` shifts are active; it is not yet an end-to-end quality claim.

In a separate reconstruction study, projecting procedural MUL1 to E4M3
slightly beat procedural FP16 MCG on aggregate error, while an 8.25-KiB
learned correction recovered only about another 0.34% relative SSE. The
correction remains an oracle, not part of the production format: its small
measured gain does not presently justify shared-memory residency or a second
decode branch.

The remaining gates are completion and external validation of the candidate
pool, global MXFP4 allocation, exact materialization and decoder closure,
TP12 routed-kernel performance, streamed full-model comparison, and final KLD
and serving evaluation. The current format is intentionally TP12-specific; a
TP-independent persistent representation is a later design phase.

For the full design rationale and experiment history, see
[`mode-adaptive-trellis-codec-plan.md`](mode-adaptive-trellis-codec-plan.md)
and [`codec-transfer-research.md`](codec-transfer-research.md).
