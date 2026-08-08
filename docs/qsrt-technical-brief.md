# Kimi-K3-QSRT technical brief

Status: TP-independent canonical storage and TP12 runtime qualification,
2026-08-07.

`QSRT` means **Quantile-Stratified Rate-shifted Trellis codec**. QSRT is an
expert-static, fixed-payload mixed-rate trellis codec for gated
mixture-of-experts weights. The Kimi-K3 construction combines three
independently useful ideas:

1. an L16 stratified-quantile graph whose transitions reconstruct finite E4M3
   values (`SQG-E4M3`);
2. equal-byte K2/K4 exchanges around K3 (`R0`, `R1`, and `R2`), selected
   separately for fused `w1`/`w3` and for `w2`; and
3. `X4T`, an exact high-quality endpoint that preserves the official MXFP4
   nibble plane and losslessly compresses its UE8M0 scale plane.

The intended artifact name is `Kimi-K3-QSRT`. The first usable checkpoint will
contain QSRT experts selected from a fresh all-expert candidate pool and X4T
experts chosen by an exact-byte global allocator. There is no raw-MXFP4 keep
tier in the storage contract.

## Frozen scope

The initial production experiment is intentionally narrow:

```text
canonical storage              TP-independent 32-channel balanced atoms
qualified runtime              TP12 first; TP is not serialized in the codec
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

The current performance gate is TP12 because that is the available Kimi-K3
deployment. TP4, TP8, TP16, TP24, and TP32 are storage-valid direct-load views
of the same artifact and require kernel qualification, not re-encoding.
Alternate companders, wider rate ladders, learned per-layer tables, and
entropy-coded hot streams remain outside the supported surface.

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

## QSRT-E4M3 reconstruction

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

The public runtime profile is **QSRT-E4M3**. Its definition is the composition
of an SQG rank map and a finite reconstruction staircase:

$$
\text{codeword}
\xrightarrow{G_K}
r
\xrightarrow{Y_{12}}
\text{finite E4M3}.
$$

The graph $G_K$ and scalar law $Y_{12}$ are independent mathematical objects.
In particular, the carry-mixed graph does not approximate an inverse CDF or a
Chebyshev polynomial.

### Carry-mixed SQG rank map

For $L=16$, rate $K\in\{2,3,4\}$, and $w=16-K$, split a codeword $t$ into
history and physical branch:

$$
h=t\mathbin{\gg}K,
\qquad
b=t\mathbin{\&}(2^K-1).
$$

With $M_w=2^w-1$, define

$$
\begin{aligned}
x_0 &= h\oplus(h\gg11),\\
x_1 &= x_0\oplus((x_0\ll11)\mathbin{\&}M_w),\\
p &= (\mathtt{0x3FA7D929}\,x_1+\mathtt{0xC928FD8E})\bmod2^{32},\\
\phi &= p\mathbin{\&}M_w,\\
s_K &= p\gg(32-K),\\
j &= \operatorname{rev}_K(b)\oplus s_K,\\
G_K(h,b)=r &= (j\ll w)\mathbin{|}\phi.
\end{aligned}
$$

Here $\operatorname{rev}_K$ reverses the $K$ branch bits. Both xorshifts are
triangular bijections on $w$ bits, and `0x3FA7D929` is odd, so multiplication
by it is invertible modulo $2^w$. Therefore $h\mapsto\phi$ is a permutation.
For fixed $h$, $b\mapsto j$ is also a permutation. It follows that

$$
(h,b)\longleftrightarrow(j,\phi)
$$

is a bijection over all $2^{16}$ directed edges. Every state has exactly one
outgoing branch in each of its $2^K$ strata, and every global rank occurs once.

The rank bijection does not by itself specify sequence behavior. If $P(h)=\phi$
is the phase permutation, $\pi_h(b)=j$ is the branch permutation, and $T$ is
the de Bruijn successor, then logical stratum $j$ induces the continuation map

$$
F_j(\phi)=P\!\left(
T\!\left(P^{-1}(\phi),\pi_{P^{-1}(\phi)}^{-1}(j)\right)
\right).
$$

The family $\{F_j\}$ is the branch-conditioned phase-transition geometry used
by Viterbi. It is part of $G_K$, not part of the scalar reconstruction law.

### Chebyshev-derived finite staircase

The exact normal staircase is defined on every global rank $r$ by

$$
u_r=\frac{r+\tfrac12}{65536},
\qquad
z_r=\Phi^{-1}(u_r),
\qquad
Y(r)=\operatorname{RNE}_{\mathrm{E4M3FN}}(1.5z_r).
$$

An E4M3-aware piecewise Chebyshev construction is a compact synthesis of this
discrete map. For the target byte $Y(r)$, its polynomial output is constrained
to lie inside the real interval that rounds to $Y(r)$. Exhaustive evaluation
over all 65,536 ranks proves byte identity. Chebyshev therefore derives the
rank-to-byte staircase; it does not participate in $G_K$.

### Twelve-bit execution staircase

QSRT-E4M3 compresses the exact staircase from 65,536 rank labels to 4,096
bytes. For $q\in\{0,\ldots,4095\}$, define

$$
Y_{12}(q)=
\operatorname{mode}\{Y(16q),Y(16q+1),\ldots,Y(16q+15)\},
$$

with the lower unsigned E4M3 byte selected on a modal tie. Runtime
reconstruction is

$$
\widehat Y_K(h,b)=Y_{12}\!\left(G_K(h,b)\gg4\right).
$$

Thus the 12-bit table is a piecewise-constant approximation to the discrete
Chebyshev-derived E4M3 staircase. It is not a Chebyshev evaluator and does not
change the reference distribution. The approximation chain is

$$
\text{normal equal-probability rank}
\longrightarrow
\text{Chebyshev-derived finite-E4M3 label}
\longrightarrow
\text{modal 16-rank execution label}.
$$

The authoritative construction is implemented independently in
`kquant/sqg_e4m3.py` and B12X. kquant generates the 4,096-byte staircase and
complete K2/K3/K4 direct encoder labels and passes them through
`kquant/sqg_quantizer.py`; B12X evaluates the same immutable construction at
runtime. Frozen SHA-256 checks over the T12 table and all three 65,536-byte
direct tables make cross-repository drift fail in unit tests. A payload
encoded under a different graph cannot be relabelled in place because
$\{F_j\}$ and the selected Viterbi paths differ.

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
physical size. Pair placement is rotated over a global 96-slot atom axis by
layer and expert so P24 work remains balanced at every supported TP view.

The mode is `(r13, r2)`. `w1` and `w3` share `r13` for fused execution;
`w2` selects `r2` independently. The common physical neuron permutation does
not require the three matrices to share a rate schedule.

## TP-independent balanced-atom storage

Tensor parallelism is a view over the checkpoint, not part of the codec. The
canonical sharding unit is a balanced 32-channel atom. For logical mirrored
record pair $i\in\{0,\ldots,11\}$ and 16-channel stripe
$s\in\{0,\ldots,7\}$, define

$$
a=8i+s.
$$

The encoder serializes logical donor/recipient pair $i$ as physical records
$2i$ and $2i+1$. Atom $a$ owns stripe $s$ from both of those physical
records. In mode `Rr`, its rate pair is

$$
(K_\mathrm{low},K_\mathrm{high})=
\begin{cases}
(2,4),&i<r,\\
(3,3),&i\ge r.
\end{cases}
$$

Both cases contain exactly six trellis bits per coefficient pair. For one
matrix, one atom therefore occupies exactly

$$
32\cdot3584\cdot\frac{3}{8}=43{,}008\ \text{bytes}.
$$

The atom bundle stores the fixed trellis fragments for `w1`, `w3`, and `w2`
plus their three 32-value FP16 intermediate-side scale fragments:

$$
B_\mathrm{atom}
=3(43{,}008+32\cdot2)
=129{,}216\ \text{bytes}.
$$

There are 96 atoms per compressed expert, so atomization preserves the exact
payload:

$$
96B_\mathrm{atom}=12{,}404{,}736\ \text{bytes per expert}.
$$

It adds no rate padding and cannot separate coupled coordinates: both sides
of a P24/P33 pair, all three expert matrices, and all three local scale
fragments have one atom owner.

### Physical atom order

Let

$$
\rho_{\ell,e}=(5e+\ell)\bmod12.
$$

The physical slot of logical atom $a$ is

$$
p=(a+8\rho_{\ell,e})\bmod96.
$$

The rotation is defined over the model-global atom axis and contains no TP
rank. It rotates complete record pairs, leaves the stripe index unchanged,
and is bijective for every layer/expert. The on-disk compressed tensor is
atom-major:

```text
[96 physical atom slots, compressed experts, 129216 bytes]
```

The layer file is a standards-valid safetensors container. Its tensors are the
atom slab above, a 4 KiB expert-format section, and a 24 KiB shared-scale
section. The safetensors header itself occupies a fixed 4 KiB. Each physical
atom row is padded once to a 4 KiB boundary, at most 4,095 bytes per slot per
layer rather than per expert. These fixed offsets permit direct range loading
of physical atom rows without parsing or copying unrelated payload bytes.

### Shard views

For any TP size $T$ dividing 96, rank $q$ owns

$$
A=96/T
$$

consecutive physical atom slots beginning at $qA$. Its local intermediate
width is $32A=3072/T$. Consequently one rank loads one aligned contiguous
extent per layer; no trellis bit is decoded, shifted, concatenated, or
repacked. All practical Kimi-K3 views through TP32 are exact direct views:

```text
TP = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32
```

TP48 and TP96 are also equal-width views. A shard count that does not divide
96 still receives one aligned, contiguous range of complete atoms. The
quotient/remainder partition covers every atom once and differs by at most one
atom, or 32 intermediate channels, between shards. A runtime requiring equal
local shapes pads only its disposable prepared cache; canonical bytes remain
unchanged. Thus arbitrary resharding never separates a P24/P33 atom or
requires trellis re-encoding.

At TP12, a rank owns eight atoms, exactly 256 intermediate channels. This is
a consequence of the canonical layout, not a serialized TP12 contract.

### Load preparation

The loader reads the small layer metadata and uses InstantTensor to transfer
only its atom-row range. It then performs one GPU preparation pass that removes
slot padding and transposes atom-major
storage into the fused-MoE operand layout. It also derives P24/P33 work queues
from `(layer, expert, physical_slot, r13, r2)`. Rate metadata is per
expert/atom during preparation; the fused coefficient loop has no TP-dependent
addressing and no coefficient-level rate branch. Rank-local prepared buffers
are disposable caches and are never checkpoint files.

The canonical implementation and byte-accounting reference are in
`kquant/qsrt_storage.py`.

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
weights, or 4.25 bpw. X4T changes no represented value:

- every E2M1 nibble is preserved exactly, including both zero codes;
- each scale row chooses the adjacent UE8M0 pair that covers the most entries;
- selector bits and out-of-pair exceptions reproduce the complete official
  scale plane; and
- load preparation partitions the decoded exact matrix on 32-channel storage
  groups, with the same equal or quotient/remainder shard rule as QSRT.

The selector stream is directly indexable and needs no tile offset table,
prefix sum, or exception search.  X4T is therefore the high-quality
endpoint; uniform K4 remains lossy and is not treated as a substitute for the
official weights. The all-expert X4T index stores each expert's exact tensor
payload contribution rather than relying on a nominal bpw estimate.

The canonical X4T layer is a TP-independent safetensors file. It contains:

```text
expert_ids: int32[E]
w1/w3.packed: uint8[E, output_rows, input_columns / 2]
w2.packed: uint8[E, 32-channel input groups, output_rows, 16]
matrix.scale_fixed: uint8[E, fixed_stream_bytes]
matrix.scale_exceptions: uint8[concatenated exception bytes]
matrix.scale_exception_offsets: int64[E + 1]
```

The safetensors JSON directory is padded to 4 KiB, which makes storage exactly
additive: each layer has 4,128 fixed bytes, and each expert contributes its
three matrix tensor payloads plus 28 bytes for its ID and exception-offset
entries. Artifact SHA-256 closure receipts authenticate the complete files;
there is no private record header, directory, CRC, or padding convention.

The stored matrix is the source of truth; rank-local W4A16 tensors are a
load-time cache. `w1`/`w3` partition on 32-row groups and `w2` on 32-column
groups. Equal divisors produce identical local shapes; other shard counts use
the same bounded uneven partition and optional cache padding as the QSRT atom
reader. The checkpoint never stores a rank count or rank-local X4T copy.

The scalar scale codec remains `kquant/mxfp4_scale_codec.py`. The existing
full-matrix `kquant/x4t.py` layer container is the canonical exact endpoint.

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
   safetensors payload bytes rather than a fixed nominal four-bit cost.

The comparison byte cap inherited from the validated 3p09 allocation is:

```text
target container bytes = 1,058,586,247,168
```

The final cap must be restated in canonical atom-container bytes before the
next allocation; TP-rank padding is not a valid budget component. The global
allocator minimizes

```text
sum_e D_e(choice_e) + lambda * sum_e bytes_e(choice_e)
```

and sweeps `lambda` to meet the checkpoint budget. Its endpoint alphabet is
`qsrt_all` and `x4t_all`; exact trellis and X4T safetensors bytes are
charged by the same optimizer. Since X4T sizes vary by expert, candidate
generation and X4T cost indexing remain reusable when the target budget
changes.

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

The mature B12X W4A16 kernel now has one QSRT serving reconstruction:
QSRT-E4M3. The slow exact profile-5 graph, R44, MUL1, and MCG codebook branches
have been removed from that kernel; the exact variants remain offline teachers
where comparisons require them. QSRT-E4M3 passes dense K2/K3/K4 reconstruction,
P24, P33, dynamic pair selection, and CUDA graph replay closure.

The current mature split-K W4A16 implementation measures about 56.90 us for
P33 and 65.09 us for P24 on the production-shaped benchmark and reproduces the
CPU decoder for P33, P24, and dynamic pair layouts. P24 remains above the
current latency target.

## Supported reconstruction path

QSRT exposes one serving profile:

| Profile | Role | Contract |
| --- | --- | --- |
| `qsrt_sqg_e4m3` | sole runtime encoding profile | `sqg_xor_cheb_t12`: the two-round XOR/odd-multiply bijective SQG graph plus the shared 12-bit approximation to the Chebyshev-derived finite-E4M3 staircase at K2/K3/K4 |

There is no runtime R44, MUL1, MCG, exact profile-5 graph, alternate K2
staircase, or per-expert codebook selector. Those names may appear in archived
measurements and offline research controls, but they are not valid payload or
kernel profile identities.

The preceding profile-5 decision was supported by a confirmation study on 384 unseen,
support-stratified experts from layers 1, 24, and 40. It used production
Hadamard ordering, TF32 dense-H BlockLDLQ, decoded-upstream conditional `H2`,
the complete `R0/R1/R2` grid, and document-disjoint confirmation and external
validation:

| Complete mode | Validation SSE improvement | Wins | Layer-stratified expert 95% interval |
| --- | ---: | ---: | ---: |
| fixed `R0/R1` | +0.04249% | 236 / 384 | crosses zero |
| fixed `R0/R2` | **+0.13345%** | **269 / 384** | **+0.07707% to +0.19738%** |
| fixed `R2/R2` | +0.06896% | 260 / 384 | +0.01682% to +0.12203% |

The document-cluster bootstrap for fixed `R0/R2` used 223 active documents
and produced a +0.07074% to +0.20312% interval. Every fit-support quartile
improved. The native K2 profile's aggregate fixed `R0/R2` exchange was
slightly harmful relative to its own `R0/R0`; Q8H4 made it beneficial.

That study remains evidence for QSRT's rate-shift architecture and calibration
policy, but its stored paths are not reusable under the new graph. The next
encoder contract uses the `qsrt_sqg_e4m3` encoding profile with the
`sqg_xor_cheb_t12` codebook, rotation draw zero, `h2_reverse`,
folded-scale power zero, decoded-upstream conditional expert-local `H2`, and
TF32 dense-H LDLQ. The profile-5 pool is now a teacher/comparison artifact;
the serving candidate pool must be re-encoded because changing continuation
geometry changes Viterbi paths.

### Offline trellis-encoder optimization

The SM120 offline tile encoder keeps the authoritative 128-symbol context on
both sides of each 256-value tile.  Its optimized implementation transposes
each state-indexed SQG byte table into predecessor-major groups so one thread
loads all K2 or K3 predecessor labels in one vector transaction and all K4
labels in two.  K2 traceback stores four two-bit decisions per byte, K3 uses a
768-thread forward pass while preserving the established 512-thread final
reduction tree, K4 uses a 512-thread/maximum-L1 configuration, and paired
half-precision comparisons update both paths together.

On 512 production-codebook tiles at C128, median kernel time changed as
follows on SM120:

| Rate | Previous encoder | Optimized encoder | Reduction |
| --- | ---: | ---: | ---: |
| K2 | 7.270 ms | 4.825 ms | 33.64% |
| K3 | 6.378 ms | 4.015 ms | 37.04% |
| K4 | 6.054 ms | 3.345 ms | 44.75% |

Safety was checked directly against the preceding CUDA extension in 63 cases
covering K2/K3/K4, C1/C32/C128, Gaussian/heavy/structured inputs, and the
production and control E4M3 tables.  Reconstructed values and trellis indices
were bit-identical in every case.  A complete 20-expert layer-24 endpoint
study also produced the identical serialized candidate-payload SHA-256 and
the same selected modes, while wall time fell from about 136 to 89 seconds.

A shorter C32 primer is not part of this optimization.  The initial
20-expert screening study put every C128 confirmation winner in C32's top
three, but that is not sufficient evidence for the all-expert pool or a future
arbitrary-record search.  The current production build remains C128 end to
end; C32 may be revisited only as a shortlist generator followed by exact C128
re-encoding after a substantially broader audit.

### Interim all-expert mode-selection evidence

The profile-ID-5 production pool was built at
`/models/Kimi-K3-QSRT-CHEB-Q8H4-CANDIDATES-v1`.  At the 2026-08-05 04:59 PDT
snapshot, 44 complete atomic selection sidecars contained 27,176 unique
experts across 33 partly or fully represented layers.  A nonzero mode was
retained only when its paired, document-clustered confirmation lower bound
cleared the zero-improvement margin.

| Selected `(r13,r2)` | Experts | Share |
| --- | ---: | ---: |
| `R0/R0` | 20,485 | 75.379% |
| `R0/R1` | 266 | 0.979% |
| `R0/R2` | 894 | 3.290% |
| `R1/R0` | 125 | 0.460% |
| `R1/R1` | 974 | 3.584% |
| `R1/R2` | 2,763 | 10.167% |
| `R2/R0` | 9 | 0.033% |
| `R2/R1` | 18 | 0.066% |
| `R2/R2` | 1,642 | 6.042% |

In aggregate, 6,691 experts, or 24.621%, selected a confirmed nonzero shift.
The down projection selected R1+ in 6,557 experts (24.128%), while the coupled
gate/up pair selected R1+ in 5,531 (20.353%).  R2 appeared on at least one
axis in 5,326 experts (19.598%).  The large `R1/R2` and `R2/R2` populations
show that the recovered effect is not confined to a few marginal R1 choices:
the down axis remains the more frequent shifter, and joint `w13`/`w2` shifts
are also common.

This snapshot is historical confirmation-stage incidence, not a final model-wide mode
distribution or a quality result.  The work-balanced schedule makes the set
of completed layers nonuniform, incomplete sidecars are excluded, untouched
validation did not turn it into a QSRT-E4M3 pool, and X4T endpoint allocation
is a separate exact-byte optimization. A new all-expert encode must recompute
the rate modes under the current graph and staircase.

## Execution checklist

- [x] Implement and unit-test L16 SQG-normal E4M3 labels for K2/K3/K4.
- [x] Integrate SQG into dense-H rate-shifted encoding and stored-state decode.
- [x] Validate SQG endpoints and the separate `(r13,r2)` R0/R1/R2 gate.
- [x] Start the resumable all-82,432-expert SQG candidate pool on 12 GPUs.
- [x] Complete and seal the all-82,432-expert R44 SQG candidate pool.
- [x] Freeze and unit-test the exact X4T numerical representation.
- [x] Add exact X4T load-time reconstruction and TP12 W4A16 preparation.
- [x] Store exact X4T layers as TP-independent safetensors and implement the
      one-launch, graph-safe routed W4A16 scale predecoder.
- [x] Benchmark X4T inside the complete routed W4A16 path across M=1/2/4 and
      1/2/4/8/16 active-expert densities.
- [ ] Build and seal the all-expert X4T byte-cost index.
- [x] Close representative SQG K2/K3/K4, P24/P33, fused-SiTU, and graph-replay
      execution through the production B12X API.
- [x] Close native SQG W4A8 dense/routed execution and measure its full-path
      activation-quantization error and latency against matched W4A16.
- [x] Seal the 223-document, 128K-token document-disjoint validation capture.
- [x] Validate R44 and the shared SQG-Cheb normal staircase at K2/K3/K4 on the
      production path.
- [x] Confirm and freeze the `w2`-only bit-4 K2 graph on 384 unseen,
      support-stratified experts with both expert- and document-clustered
      positive confidence intervals for fixed `R0/R2`.
- [ ] Score selected candidates on the untouched validation capture and run
      the matched-R0 rate-shift policy audit.
- [ ] Freeze the global QSRT allocation at the target checkpoint budget.
- [ ] Materialize QSRT and X4T physical-slot extents into a fresh atom-major
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
