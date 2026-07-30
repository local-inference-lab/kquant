# Kimi-K3 12-GPU Serving — Handoff Brief

*Written 2026-07-30 ~10:30 by the outgoing agent. Read the whole thing before
touching anything. The experiment that may contain the answer is still
running — see §1.*

## 0. Mission

Serve moonshotai/Kimi-K3 (2.8T MoE, 92 MoE layers × 896 experts, MXFP4 QAT,
SiTU activation, KDA+MLA hybrid attention) on Luke's 12× RTX PRO 6000
Blackwell (SM120, 95.6 GiB each), TP12, via the local vLLM fork + b12x
(sparkinfer) kernels. Quality target: keep-hot-experts-exact + EXL3-3.0 for
the rest. Goal state: coherent generation, breakable CUDA graphs
(PIECEWISE), optimal decode.

**Current blocker: every configuration on THIS box emits deterministic
garbage (frequency-prior tokens, sharp logits after the serialized-MXFP8
fix). This includes 100% original-weight configurations. Martin's 16-GPU box
(vm1.voipmonitor.org:13212, /root/k3-serve, launch-graphs.sh) serves the
same model lineage coherently — it is the working reference.**

## 1. THE IN-FLIGHT EXPERIMENT — READ THIS FIRST

A 4-layer truncated K3 (`/models/k3-trunc-stock` = 100% original weights:
BF16 attention + stock MXFP4 experts, layers 0-3 + embed/lm_head/norms)
reproduces the garbage in a **3-minute boot cycle**.

At handoff, this was running: the same truncated stock model served via its
**native compressed-tensors quantization config** (original config.json
quant block restored; NO hybrid config; b12x MoE backend ON) vs the
already-recorded run via the **nvfp4_nf3_hybrid all-kept path**
(`scratchpad/trunc-stock-tp12.json`). Token-level logprob diff prints
automatically (background task chain; results land in
`$SCRATCH/trunc-native-tp12.json` and the task output).

**RESULT (landed at handoff): native == hybrid EXACTLY — max |Δlogprob|
0.0000, identical completions.** Both routes drive the same b12x kernels,
so the hybrid quant-method plumbing is exonerated. Combined with everything
in §2, the fault is in what every config on this box shares: the
KDA/MLA attention stack or model glue as executed HERE.

Your next discriminator: run the same truncated model on Martin's box
(tensors+config+tokenizer already staged at `/root/k3-trunc-stock` there —
his GPUs were free at handoff, his production server down; restore it
after: `/root/k3-serve/launch-graphs.sh`) and diff token logprobs across
boxes. USE HIS STACK AS-IS (/opt/venv). If his stack gives different
logprobs on the same 4-layer weights, instrument both sides with
KIMI_DEBUG_NORMS-style per-module dumps and walk layer 0 → 3 until the
first divergent module; it will be one of: dense-L0 MLP, KDA internals
(Triton kernels), MLA prefill, or the attn_res combine. If his stack gives
the SAME logprobs, the truncated harness is a red herring for absolute
quality (a 4-layer amputee is legitimately incoherent) and the comparison
must move to full-model per-layer traces vs his box.

`$SCRATCH = /tmp/claude-1000/-home-luke-projects-kquant/ea779040-a191-46b7-8e07-61d45f78bfc5/scratchpad`
(session-scoped; copy anything you need out of it early).

## 2. Verified facts (do not re-litigate)

Numerics / integrity — all measured, all clean:
- Local TP invariant HOLDS: truncated stock at TP2 vs TP4 vs TP12 —
  identical greedy completions, max |Δlogprob| ≤ 0.073 (reduction noise).
  TP geometry/padding is NOT the bug.
- EXL3 demoted tier: offline closure through the production
  `sparkinfer.moe.fused_moe` kernels (TP12-sliced, SiTU, top-8) rel err
  0.21 vs dequantized source — healthy quantization noise, kernels sound.
  `kquant/scripts/closure_exl3_layer.py` (--broadcast tests shared-su path).
- Offline MXFP8 bake is BIT-IDENTICAL to the online CUDA op
  (`mxfp8_e4m3_quantize`) — same scales, same 2.66% roundtrip.
- Final-stage tensors (lm_head, final norm, attn_res norm, embed) in the
  serve dirs are byte-identical to the source checkpoint.
- A_log loader narrows the 128-padded flat tensor by logical head prefix —
  correct at TP12 (8 heads/rank).
- Per-layer hidden norms are healthy through all 93 layers (embed 1.8,
  hidden 0.2–21, no explosion/collapse). Corruption is semantic, not
  magnitude. Env-gated instrumentation: `KIMI_DEBUG_NORMS=1` (in
  `kimi_linear.py`, prints on TP rank 0).
- The b12x/vllm unit+contract tests pass on this box (25/25 fused_moe
  trellis, 7/7 w2 realign, 30/30 hybrid+warmup).

Fixed along the way (real bugs, already committed):
1. `_vllm_fa2_C/_vllm_fa3_C` were ABI-dead vs torch 2.12 (precompiled wheel
   vs upgraded torch). Working July-5 source builds found in
   `build/vllm-cmake/vllm-flash-attn/` and swapped in. MLA prefill selector
   restored to Martin's `major == 10` gating (FlashInfer-fa2-on-SM120 was
   our unvalidated NF3-era hack).
2. Trellis tier ported to refactored `sparkinfer.moe.fused_moe` API
   (plan_weights/prepare_weights/Caps(weight_plan)/scratch_specs/bind
   experts=/run→fp32-cast). Activation MUST come from `self.moe.activation
   .value` (a getattr default silently ran silu instead of SiTU).
3. Kept tier at TP12 defaults to `w4a8_mx` in refactored b12x (local I=256
   divides 128; TP16's 192 does not) — pinned with `B12X_MOE_FORCE_A16=1`.
4. Serialized MXFP8 checkpoint loading: `dense_format: "mxfp8"` +
   `ignored_layers` in the checkpoint quant config routes dense linears to
   the existing `Mxfp8SerializedLinearMethod` (same convention as
   `Fp8Config`). Saves ~2.7 GiB/rank resident + the whole online-quantize
   boot phase. Bake: `kquant/scripts/bake_mxfp8_nonexpert.py` →
   `/models/Kimi-K3-mxfp8-nonexpert` (71 GiB).
5. Martin's hard-won constraint (his README): NEVER instanttensor +
   online-quant overlay (`copy=False` ring-buffer race → nondeterministic
   corruption ~2/3 boots). Use `--load-format safetensors` until his
   loader-hardening commits are validated with instanttensor again.

## 3. Artifacts (on /models, all verified)

- `Kimi-K3-EXL3-3p14` / `-serve`: keep 0.11 (12,530→9,068... see configs),
  shared-su. Superseded by 3p09 but intact.
- `Kimi-K3-EXL3-3p09` / `-serve`: keep 0.085 (7,007 keeps), shared-su,
  proxy 0.0174–0.0175 uniform, serve dir uses baked MXFP8 non-expert +
  `dense_format`. Loads at **87.97 GiB/rank**, boots, generates (garbage —
  see §1; artifact itself is NOT the cause).
- `Kimi-K3-mxfp8-nonexpert`: baked fp8+e8m0 non-expert shards (targets:
  KDA q/k/v/o, MLA q_a/q_b/kv_a_with_mqa, shared experts, dense-L0 MLP;
  BF16 kept: kv_b_proj, g_proj, f_a/f_b, b_proj, conv1d, norms, lm_head).
- `k3-trunc-stock` (55G): 4-layer original — THE repro. Config currently
  carries the ORIGINAL compressed-tensors quant block.
- `k3-trunc-ours` (42G): 4-layer version of the 3p09 artifact.
- Source checkpoint: HF cache has TWO snapshot dirs.
  `c5d1dd4c...` has all 96 weight shards; `9f62e4e9...` is config-only and
  is what `kquant.io.hf_cache.resolve()` returns. FIX resolve() or pass
  explicit paths. (I burned an hour + a pointless 24 GB scp on this.)
- Shared-su A/B test dirs (`Kimi-K3-EXL3-shared-su-test*`): can be deleted.

Effective-bpw ledger at 3p09: ~88 GiB/rank weights + ~2.3 non-torch + KV.
Boot with `--kv-cache-memory-bytes 268435456`-ish caps until the KDA
q/k/v/o MXFP8 + graph-capture memory plan is finalized.

## 4. Repos (all local, all committed)

- `~/projects/vllm` branch `k3-pp2tp6`, REBASED onto `origin/dev/gg-k3`
  (Martin's branch; his newest loader hardening + chat renderer included).
  Our 8 commits on top: TP12 padding (router gate + latent projections —
  keep `quant_config=None` on latents unless you also bake them), EXL3
  integration, shared-su broadcast registration, fused_moe port, FA
  gating revert, dense_format. Serve script: `serve-kimi-k3-exl3.sh`
  (safetensors, B12X_MOE_FORCE_A16, FP8 GEMM, 1800s execute timeout,
  slim profile).
- `~/projects/b12x` on master (36cade0b). Martin's deployed sparkinfer
  (d4dc4ad) is an ANCESTOR — our master is 9 commits ahead (indexer tests,
  compile-cache-by-UUID, shared-su broadcast tables). Broadcast H-side
  rotation tables: `[1,H]` accepted end-to-end, zero-expert-stride compile
  flag, tests green. sparkinfer `origin/dev/gg-k3` has ONE unported perf
  commit (64a90970 "accelerate K3 hybrid decode", 297 lines vs pre-refactor
  kernel.py) — port later, carefully.
- `~/projects/exllamav3`: shared-su encoder mode (`KQUANT_EXL3_SHARED_SU=1`:
  shared channel-scale profile + g_scale folded into sharded sv). Committed
  locally.
- `~/projects/kquant`: pack pipeline + scripts (pack_exl3_12gpu.py with
  --layers, bake_mxfp8_nonexpert.py, package_exl3_serve_dir.py
  (parameterized), make_truncated_k3.py, run_trunc_matrix.sh,
  closure_exl3_layer.py).

## 5. Environment facts & footguns (each cost real time)

- **pkill/pgrep self-match**: your own wrapper shell's command line contains
  your pattern. Kill ONLY GPU-attested PIDs
  (`nvidia-smi --query-compute-apps=pid`) or exact stored PIDs. This bit us
  four times (exit code 144 = you killed yourself).
- **Port 8011 hygiene**: check `ss -tlnp | grep 8011` before boot; a stale
  API server causes silent probe-routing to the WRONG server (an entire A/B
  was invalidated this way).
- **CUDA_VISIBLE_DEVICES dedups duplicates** — no TP16 on 12 GPUs that way.
  vLLM `--nnodes 2` + `--headless` node also fails: NCCL refuses two ranks
  per device ("invalid usage"). TP16 comparisons need Martin's box. TP2/TP4
  are the local clean-geometry references (TP1 overflows int32 in the b12x
  host interface; TP8 violates trellis I%256 for EXL3 artifacts).
- **First-request CuTe compiles** take minutes-to-tens-of-minutes and the
  engine watchdog kills at `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (default
  300; serve script sets 1800). Compiles disk-cache by device UUID —
  second cycle is fast. A rank that finished compiling spins at 100% GPU in
  a barrier; a rank still compiling shows ~100% CPU. **0% CPU + 0% GPU on
  one rank + 11 spinning = deadlock/shutdown, not compiling** — check
  py-spy (`sudo env PATH=$PATH py-spy dump --pid <worker>`), don't wait.
- **Memory**: startup precheck needs free ≥ util×total at worker init
  (context ~1.06 GiB already resident → util ≤ ~0.985). instanttensor
  staging is `MAX_FREE_MEM_USAGE` × free and must leave room for load-time
  temporaries. ~2 GiB of unattributed post-profile allocations exist —
  explicit `--kv-cache-memory-bytes` caps are more reliable than util math.
- **Host RAM**: 12 pack workers ≈ 20 GB each + slow heap creep → OOM killer.
  Swapfiles at `/data/kquant-swapfile{,2}` (96G+128G) — keep for pack runs.
- **Monitors**: use the Monitor tool (events stream); a `run_in_background`
  bash loop only notifies on EXIT — alerts written to its stdout are
  invisible until then. This delayed a pack-worker-death detection by 25
  minutes.
- The GPU that fell off the bus (Xid 120→79, bus E3:00) recovered only via
  full reboot. If it happens again: don't sysfs-reset a live-workload GPU;
  reboot. Enumeration shifts silently break `CUDA_VISIBLE_DEVICES=<index>`
  pinning — pin by UUID.
- Kernel/attention probe noise: the `fa_utils` FA2-unavailable ERROR line
  is history (fixed .so) — if it reappears, the ABI broke again.

## 6. Martin's box (reference oracle)

`ssh -p 13212 root@vm1.voipmonitor.org`. `/root/k3-serve/README.md` is
required reading (his garbage-hunting diary + hard-won constraints).
Production launch: `launch-graphs.sh` — PIECEWISE breakable CUDA graphs
(`VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `cudagraph_mode: PIECEWISE`, capture
[1,2,4]) ≈ 30 tok/s vs 3.5 eager. FULL graph mode bakes KDA metadata →
garbage (his attempts 5-8). His box's production server was STOPPED during
our session — relaunch it when done, and generally: read-only by default,
his box is a favor.

For the goal's graph-capture phase on OUR box: his launch flags are the
recipe; K3 is not in AUTO_BREAKABLE_CUDAGRAPH_ARCHITECTURES so the env is
mandatory.

## 7. Recommended sequence for you

1. Read the native-vs-hybrid diff (§1). Follow its verdict with the
   3-minute truncated repro. This is the critical path; everything else is
   staged and waiting on coherent output.
2. When output goes coherent on truncated: full 3p09 boot → same probe →
   then graphs (Martin's flags) → then perf (W4A8 kept tier, the sparkinfer
   perf-commit port, instanttensor revival via Martin's loader hardening,
   keep-frac restore once KDA-MXFP8 lands the headroom).
3. Quality validation once serving is sane: port Martin's
   `probe_replay.py` battery; KL/logprob eval vs his box's full-model
   output (his box = the reference, modulo expected quant noise).
4. Longer arc (user-approved roadmap): Martin's L1 measured routing stats →
   keep-set re-rank; L2 → Hessian-weighted EXL3 requant.

## 8. Why I was fired — read carefully

Every item below is a real incident from this session, not hypothetical:

1. **I called a deadlock "still compiling" for 20 minutes** because I
   watched an aggregate signal (11 GPUs at 100%) instead of checking the
   one idle rank's CPU. Verify liveness per-component before waiting on
   anything. Attach hard deadlines to every wait, with "diagnose" as the
   timeout action, never "wait more".
2. **I speculated when I could have measured.** The operator's instincts
   were right every time: bake conversions offline; build a truncated
   model; assert TP-invariance; "vLLM obviously loads MXFP8 from disk —
   find the existing mechanism". When he says something is a common recipe,
   the correct move is `grep`, not designing a new subsystem.
3. **I declared the source checkpoint lost** without checking for a second
   snapshot directory, then started a 24 GB network copy to work around a
   problem that didn't exist. Exhaust local evidence before acting on a
   dramatic conclusion.
4. **I used internal jargon** ("kept tier", "serialized-load integration")
   that obscured meaning from the person who understands the system best.
   Say "the original MXFP4 weights". Say "teach vLLM to load pre-converted
   weights". Plain language is a debugging tool.
5. **I left a dozen zombie shells/monitors** accumulating across boot
   cycles. Kill what you spawn; keep an exact inventory; the operator
   noticed before I did.
6. **I burned boot cycles on knife-edge memory configs** instead of taking
   the structural fix first. When the margin is <1 GiB and boots are
   15 minutes, spend the hour on the real gigabytes.
7. Sequencing discipline: fix → verify the fix landed (my gpu_worker
   revert silently failed on a dirty tree and cost a full boot cycle) →
   then boot. Never launch with an unverified tree.
8. The operator wants terse, high-frequency status when things are moving
   (30s cadence was explicitly requested at one point) and zero
   philosophizing. Lead with the number, the verdict, the next action.

Good luck. The bug is close: it reproduces in 4 layers, in 3 minutes, with
100% original weights, and there is a working reference one ssh away.
