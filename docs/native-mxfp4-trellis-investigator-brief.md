# Native-MXFP4 trellis coding for Kimi-K3

Status: layer-24 source-domain study complete, 2026-08-02. This document
defines the hypothesis, reference codec, results, and remaining integration
questions.

The separate
[`mxfp4-reconstruction-alphabet-foundations-brief.md`](mxfp4-reconstruction-alphabet-foundations-brief.md)
frames the underlying rate-distortion, finite-state-coding, requantization,
and hardware questions for a broader investigator-led literature review.

## Executive question

Kimi-K3's official expert weights are already MXFP4: an E2M1 value nibble and
an E8M0 scale for each 32-value block. The existing EXL3 encoder first expands
those weights, applies scaling and incoherence transforms, and searches a
large procedural codebook whose reconstructed values are FP16. It stores three
new branch bits per weight in the K3 case.

That is a reasonable general-purpose quantizer, but Kimi-K3 presents an unusual
source-coding problem. The source has only fifteen distinct numerical values
per scale block: the sixteen E2M1 wire codes include both `+0` and `-0`. A
codec that emits native E2M1 may give its limited path bits more freedom to
select trajectories among values the source can actually contain, close
exactly at K4, and feed Blackwell's native MXFP4 weight MMA without expanding a
dense FP16 weight tile.

The hypothesis is therefore:

> A small-state trellis whose output alphabet is exactly the source E2M1 grid,
> while preserving the official K32 scales, may offer a better
> quality/decoder-throughput tradeoff than a general FP16-valued procedural
> codebook for this already-quantized source.

This is not yet a claim that native output is better. In particular, dense-H
LDLQ changes the encoder target from an original E2M1 point to a compensated,
generally continuous residual target. That is the strongest reason an FP16
output alphabet may still win.

## Relevant K3 geometry and formats

A routed Kimi-K3 expert contains:

```text
w1: [3072, 3584]  gate projection
w3: [3072, 3584]  up projection
w2: [3584, 3072]  down projection
```

The official checkpoint stores every matrix in logical `[out, in]` order:

```text
weight_packed: uint8 [out, in/2]
weight_scale:  uint8 [out, in/32]
```

The low and high nibbles of `weight_packed` are consecutive E2M1 values. Each
scale byte represents `2^(byte-127)` and applies to 32 consecutive input-axis
values in one output row. The exact E2M1 decoding table is:

```text
wire code:  0   1    2    3    4   5   6   7    8    9    10    11    12   13   14   15
value:     +0  +.5  +1  +1.5  +2  +3  +4  +6   -0   -.5   -1   -1.5   -2   -3   -4   -6
```

Consequently the official representation costs `4 + 8/32 = 4.25` bits per
weight. Codes 0 and 8 are distinct wire symbols but are identical for ordinary
matrix multiplication.

The current EXL MCG codebook is different. It carries a 16-bit shift-register
state, appends K new bits per coefficient, multiplies the state by
`0xCBAC1FED`, applies bit operations, interprets two half-precision fields, and
adds them to obtain an FP16 reconstruction value. The 16-bit state gives
65,536 procedural reconstruction entries while the path rate remains K bits
per coefficient. This native-MX experiment intentionally does **not** inherit
that 16-bit window.

## Minimal native-E2M1 trellis

For rate K, define:

```text
carried bits m = 4 - K
states       S = 2^m
branches     B = 2^K
```

At coefficient `i`, the decoder has state `s_i`, reads branch payload `b_i`,
and indexes one of exactly sixteen labelled transitions:

```text
t_i     = s_i * B + b_i
code_i  = pi_K[t_i]
s_(i+1) = b_i & (S - 1)
```

`pi_K` is a permutation of the sixteen native E2M1 wire codes. It is a tiny,
global, rate-specific compile-time table; it is not per-expert metadata. The
three rates are:

| Rate | Carried state | Outgoing branches | Interpretation |
| --- | ---: | ---: | --- |
| K2 | 2 bits / 4 states | 4 | Four four-entry state-conditioned palettes |
| K3 | 1 bit / 2 states | 8 | Two eight-entry state-conditioned palettes |
| K4 | no state | 16 | Every source nibble is selected directly |

Paths are tail-bitten over a 256-coefficient 16x16 tile. Since the next state
is just the low carried-state bits of the branch, the decoder derives the
initial state from the final branch. No initial-state header is stored. K4 is
raw-symbol exact for every source tile because `pi_4` is invertible.

The 16x16 unit is retained to match the current tile/lane layout and provide a
clean systems comparison. It is a codec unit, not an assertion that all SM120
MMA atoms have 16x16 granularity.

## The duplicated zero is a graph resource

The second zero should not simply be discarded. Both zero wire labels occupy
real transitions, and Viterbi may choose either whenever they are numerically
equivalent. Their benefit can take two forms:

1. **State coverage:** zero can be reconstructed exactly from more than one
   current state.
2. **Future-state freedom:** equivalent zero transitions can lead toward
   different successor-state trajectories, reducing later error without
   increasing current error.

Placing the zeros at bitwise-opposite transition indices `0000` and `1111` is
a useful topology hypothesis: it maximizes their separation in the
current/successor-state table. It is not automatically optimal. In the first
small layer-24 smoke test, the simple opposite-corner seed improved K2 over
native nibble order but regressed K3. The full search therefore retains both
zeros but learns their positions rather than fixing them by symmetry.

This distinction matters. The right object is not a numerically sorted scalar
codebook; it is a labelled transition graph. Ordinary E2M1 nibble order makes
the K3 current state select positive versus negative palettes, effectively
asking prior coefficients to predict weight sign. A sampled layer-24 `w2`
showed essentially balanced signs and negligible adjacent bit mutual
information, so that particular graph has no obvious statistical basis.

## Learning the transition table

For source codes `c_i`, unchanged scale-square weights `a_i`, and a candidate
table `pi_K`, the training objective is:

```text
min over pi_K  sum over tiles [
    min over closed paths sum_i a_i *
        (E2M1[c_i] - E2M1[pi_K(s_i, b_i)])^2
]
```

The implementation compares four table families on experts excluded from the
table fit:

- native nibble order;
- a deterministic opposite-corner-zero seed;
- an optimized table constrained to opposite-corner zeros; and
- an optimized unconstrained permutation that still contains both zero
  labels.

K2 and K3 may use different global tables. The runtime already dispatches on
K, so this adds no per-record bits and only two sixteen-byte constant tables.
K4 can use any permutation and remains exact; preserving the raw source nibble
is done by encoding through its inverse table.

The current reference search uses random restarts followed by swap-based local
improvement. A deeper investigation should compare this with simulated
annealing, assignment relaxations, and direct graph optimization. Training and
selection must stay separated by expert or document to avoid selecting a
fortuitous permutation.

## Transform and scale isolation

The first experiment deliberately uses:

```text
Hadamard rotation: none
sign rotation:     none
scale refit:       none
scale format:      original E8M0, one per output-row/K32 block
tile orientation:  official [out,in] block transposed to GEMM [K,N], then
                   current tensor-core lane order
```

Keeping the original scale makes K4 exact and isolates whether sequence coding
can exploit the already-discrete source. It does **not** establish that scales
and Hadamard rotations are theoretically interchangeable. Scales normalize
local dynamic range; a Hadamard mixes coordinates and changes distributional
shape. Later ablations must test record-aligned Hadamard, no Hadamard, and
possibly K32-local transforms under the actual dense-H encoder.

No full floating-point source matrix is created on CPU. The experiment expands
only source nibbles plus the scale-square weights for sampled tiles.

## Exact rate accounting

For a 256-coefficient tile, the path payload is exactly `32*K` bytes. A
16-column tile consumes half of each row's K32 scale block, so eight scale
bytes are attributed to it. The accounting is:

| Variant | Path bytes/tile | Scale bytes/tile | Effective rate |
| --- | ---: | ---: | ---: |
| K2 | 64 | 8 | 2.25 bpw |
| K3 | 96 | 8 | 3.25 bpw |
| K4 | 128 | 8 | 4.25 bpw |
| one K2 + one K4 | 192 | 16 | 3.25 bpw average |
| two K3 | 192 | 16 | 3.25 bpw average |

Thus K2/K4 and K3/K3 have exact equal payload and scale bytes. Native-scale K3
is 3.25 bpw, not the approximately 3.0 bpw of the current EXL representation;
scale storage or derivation would need separate work if the final target must
remain at 3.0 bpw.

## What the layer-24 experiment measures

The completed study uses official source weights from all 896 experts in layer
24. It samples 64 16x16 tiles from each of `w1`, `w3`, and `w2`, or about 44
million held-out coefficients. Separate experts supply 393,216 coefficients
for transition-table training, and another disjoint set compares table
generalization.

For each sampled matrix it reports:

- scale-weighted source energy;
- uniform K2 and K3 weight-space SSE/NMSE;
- exact-zero K4 closure;
- source code and numerical-zero frequencies;
- a fixed alternating K2/K4 diagnostic;
- an oracle that selects the lower-K2-error tile within each pair; and
- an oracle that assigns K2 to the best half of all sampled tiles.

The oracle allocations deliberately estimate a ceiling. They would require a
fine-grained rate map and are **not** the fixed-record TrellisShift format. A
useful native alphabet should first make the oracle K2/K4 error lower than
uniform K3 error at identical 3.25 bpw. If even the oracle loses, no importance
ranking can rescue this particular K2/K4 construction.

These are source-weight metrics, not dense-H or routed-function metrics. A win
is permission to proceed, not evidence of end-to-end quality. A loss is more
decisive for the raw-source version because the oracle already has privileged
knowledge of K2 error.

## Layer-24 result

The run completed all 2,688 expert matrices and 44,040,192 held-out
coefficients. The mapping-holdout aggregate excludes the eight experts used to
fit the tables; it contains 888 experts and 43,646,976 coefficients. The
holdout and all-expert figures are effectively identical.

The table search mattered substantially:

| Rate | Native nibble-order NMSE | Selected-table NMSE | Relative reduction |
| --- | ---: | ---: | ---: |
| K2 | 15.893% | 11.906% | 25.1% |
| K3 | 4.918% | 3.205% | 34.8% |

K2 selected an unconstrained table with zero labels at transition indices 6
and 11. K3 selected the optimized opposite-zero family, with zeros at indices
0 and 15. The selected transition values are:

```text
K2, as a 4x4 [current state, branch/next state] table:
[-3,  -0.5, +1.5, +4]
[+2,  -6,    +0,  -2]
[+0.5,+3,  -1.5,  -0]
[-1,  +1,    -4,  +6]

K3, as two 8-entry current-state palettes:
[+0,  -1, +1.5, -6, -1.5, +0.5, +3, -3]
[-0.5,+2, +1,  -2, -4,   +4,   +6, -0]
```

On disjoint mapping-validation paths, K3 reconstructed 67.80% of coefficients
at their exact numerical E2M1 value. It reconstructed 91.16% of source zeros
exactly, versus 64.34% of nonzero values. Both zero labels were actively used:
`+0` represented 11.99% and `-0` 5.46% of decoded coefficients. The source
itself was 12.90% numerical zero in this diagnostic sample, so Viterbi also
used zero as the least-error approximation to some small nonzero weights.

The unequal-rate result is negative and unusually decisive:

| Equal-average-rate assignment | Error relative to uniform K3 |
| --- | ---: |
| Fixed alternating K2/K4 | 1.857x |
| Best K2 tile within every pair | 1.731x |
| Globally easiest half of tiles assigned K2 | 1.682x |

Even the global oracle did not beat K3 for a single one of the 2,688 sampled
expert matrices. Aggregating all three matrices per expert, it did not win for
any of the 896 experts. The result was consistent by matrix family:

| Matrix | K2 NMSE | K3 NMSE | Global K2/K4 oracle vs K3 |
| --- | ---: | ---: | ---: |
| `w1` | 11.738% | 3.196% | 1.655x |
| `w3` | 11.739% | 3.198% | 1.655x |
| `w2` | 12.195% | 3.208% | 1.738x |

This establishes two different conclusions:

1. The constrained output alphabet really does create useful route freedom.
   A learned graph and the duplicated zero reduce distortion materially over
   naïve nibble ordering.
2. That benefit is nowhere near large enough to make a native-grid K2 donor
   pay for an exact native-grid K4 recipient. K2 damage dominates K4 recovery,
   even with an unrealistically informed tile oracle.

The current 3p09 layer-24 MCG pack reports an approximately 1.744% identity-H
encoder proxy, and prior canonical sampled cases are generally around
1.7--2.0% NMSE. Those numbers are not a fully controlled comparison because
MCG uses its fitted scales and Hadamard/sign pipeline while this experiment
fixes official MX scales and removes transforms. Nevertheless, native K3's
3.20% NMSE at 3.25 bpw is not an encouraging replacement signal. The
unchanged-scale/no-Hadamard hypothesis should therefore not be integrated into
the main TrellisShift encoder.

What remains worth studying is narrower: whether a native-output table can be
combined with captured-H compensation, a local transform or scale adjustment,
or a larger but still inexpensive state while retaining a direct MXFP4 MMA
consumer. Any such variant must first beat an identically configured MCG/MUL1
control. The raw K2/K4 native-grid branch is closed unless a materially
different construction changes the donor distortion.

## Interaction with LDLQ and TrellisShift

The main TrellisShift pipeline uses a full Hessian-aware counterfactual encode.
For `w2`, BlockLDLQ modifies later quantizer targets to compensate for earlier
error through dense off-diagonal covariance. Those adjusted targets generally
do not lie on the original E2M1 grid. The native codebook could therefore lose
the apparent source-grid advantage as soon as dense-H feedback is enabled.

The required progression is:

1. Raw source, unchanged scales, no transforms: the current falsification
   test.
2. Same codec with captured-H LDLQ targets and candidate-specific residuals.
3. Compare native E2M1 output with MCG and MUL1 under identical transforms,
   Hessian, scales, tile order, and exact bytes.
4. Evaluate uniform K3 and full-record R1/R2 K2/K4 shifts with the existing
   routed expert-function objective.
5. Only then consider this alphabet for the production TSH candidate pool.

The native experiment does not invalidate or replace the in-progress MCG TSH
encode. That run remains the algorithmic baseline and produces reusable
Hessian/routing evidence.

## Hardware opportunity

SM120 exposes MX MMA forms that consume E2M1 weights, E4M3 activations, and
UE8M0/E8M0 block scales. B12X already contains a W4A8 MX path using an
`m16n8k32` family instruction. A successful native-output trellis could decode
directly into the MMA's E2M1 operand layout and retain the original K32 scale,
avoiding a dense FP16 decoded weight tile and FP16-weight MMA feed.

That potential throughput benefit must be measured, not assumed. The decoder
still performs path reconstruction and bit extraction, activations may need
BF16-to-E4M3 block quantization, and a K2 transition can be more control-heavy
than a K4 direct lookup. Kernel gates should measure isolated K2/K3/K4 decode,
mixed routed workloads, registers, occupancy, graph capture, and end-to-end
MoE latency at TP12.

## Broader extensions to keep on the table

- Learn one table per rate globally, per matrix family, or per layer, charging
  exact cold-table bytes in each case.
- Optimize traversal order within the fixed tile rather than assuming the
  current lane order is statistically best.
- Compress or derive E8M0 scales to recover the 0.25-bpw native-scale overhead.
- Test native FP4 output with record-local Hadamard while preserving exact
  transform cancellation around SiTU.
- Test a K3/K4/K5 blend as a possible replacement for retained MXFP4 experts,
  targeting original-weight quality at slightly below 4.25 average bpw.
- Consider a larger state only if the measured distortion justifies it. State
  size is an empirical coding knob, not a reason to inherit MCG's 16-bit
  window by default.

## Implementation and reproduction

The reference codec is in `kquant/e2m1_trellis.py`, with exhaustive small-path,
packing, duplicate-zero, and K4 closure tests in
`tests/test_e2m1_trellis.py`. The resumable layer study is
`scripts/experiment_e2m1_trellis.py`.

The completed layer-24 invocation was equivalent to:

```bash
.venv/bin/python scripts/experiment_e2m1_trellis.py \
  --layer 24 \
  --experts all \
  --mapping-train-expert-count 8 \
  --mapping-validation-expert-count 8 \
  --mapping-tiles-per-matrix 64 \
  --eval-tiles-per-matrix 64 \
  --mapping-random-trials 128 \
  --mapping-hill-steps 128 \
  --output out/e2m1-trellis-layer24-native-v1.json
```

The output records the exact source revision, table entries, zero positions,
training and disjoint-validation scores, per-expert/matrix metrics, exact rate
contract, progress, and completion state.
