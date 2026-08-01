# GLM-5.2 calibrated EXL3 shared-H recipe

This recipe produces a TP4, calibrated EXL3-Trellis checkpoint whose three
hidden-sized rotation vectors are stored once per MoE layer and TP rank instead
of once per routed expert:

- `gate_proj.suh`: shared across 256 experts;
- `up_proj.suh`: shared across 256 experts;
- `down_proj.svh`: shared across 256 experts.

The intermediate-side vectors remain expert-local. At GLM-5.2 dimensions this
removes exactly `705,024,000` persistent bytes, or `672.36 MiB`, from every GPU
without expanding the shared rows during load.

## Source and scope

The input is the public `calibration_encoder` bundle from
[`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw/tree/main/calibration_encoder).
The preparation script verifies these inputs before changing anything:

| Input | SHA-256 |
| --- | --- |
| `encode_tr3_v31.py` | `e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032` |
| `encode_b300.py` | `f378817b212dc9f4a8c9dc049803542e7c91748283f6e8ec1ebe0427be96aaf1` |
| `calibration/reap_recall_calib.jsonl` | `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4` |

The BF16 source remains [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2).
The quantized scope is unchanged from the published recipe: all 256 routed
experts in layers 3 through 77 use calibrated 3-bit MCG Trellis; attention,
dense layers, shared experts, routers, MTP, embeddings and `lm_head` remain
byte-identical BF16.

This is a new encode from BF16. Existing expert-local EXL3 checkpoints cannot
be deduplicated losslessly because their H-side values are not identical.

## Why two passes are required

The shared magnitudes depend on each expert's BF16 weights, the calibrated
Hessian and the automatic output-scale decision. The patched encoder therefore
does two passes for every layer:

1. Run the normal calibrated pre-regularization for all experts and form a
   signed geometric-mean profile for each `(projection, TP rank)`.
2. Rerun the original calibrated LDLQ/Trellis encode with the profile forced on
   the H side.

Gate/up keep expert-local output vectors and move each expert's scalar
`g_scale` from shared `SU` to expert-local `SV`. Down keeps expert-local `SU`,
so its standard `g_scale` placement is unchanged. These placements are
algebraically equivalent; tests cover the equality. Profile artifacts are
bound to the source recipe and per-layer capture SHA, so stale resume data is
rejected.

## Prepare the encoder

Download the published bundle, then apply the small, reviewable patches:

```bash
export ORIGINAL_BUNDLE=/workspace/original/calibration_encoder
export SHARED_ENCODER=/workspace/tr3-shared-h

python3 recipes/glm52_exl3_shared_h/prepare_shared_h_encoder.py \
  --bundle "$ORIGINAL_BUNDLE" \
  --output "$SHARED_ENCODER"
```

The command refuses modified input files and verifies the patched outputs:

| Output | SHA-256 |
| --- | --- |
| `encode_tr3_v31.py` | `400c0df1c95c81c30a2ce31e060f0445a798fd29ad9339923d4e02e3ee40f6f7` |
| `encode_b300.py` | `b41f9397a1754e67f41b2356db413b5da18228bcd3961c97023b4e1cabd01010` |

## Full conversion

Use the same B300 environment as the published recipe: exllamav3 `0.0.43`,
CUDA 12.9, a BF16 model checkout, sufficient tmpfs for captures and roughly
0.5 TB assembly scratch. The reference capture used eight B300 GPUs and packs
the artifact for TP4.

```bash
export SCRIPT_DIR="$SHARED_ENCODER"
export WORK_ROOT=/workspace/tr3-shared-h-work
export BF16_SRC=/workspace/bf16
export OWNER_CORPUS="$SHARED_ENCODER/calibration/reap_recall_calib.jsonl"
export BASE_ENCODER_PY="$SHARED_ENCODER/encode_tr3_v31.py"
export OUT_DIR=/workspace/output/GLM-5.2-EXL3-TR3-Shared-H
export CUDA_HOME=/usr/local/cuda-12.9

"$SHARED_ENCODER/convert_b300.sh" preflight
"$SHARED_ENCODER/convert_b300.sh" ext
"$SHARED_ENCODER/convert_b300.sh" plan

for window in 3-10 11-18 19-26 27-34 35-42 43-50 51-58 59-66 67-74 75-77; do
  LAYERS="$window" "$SHARED_ENCODER/convert_b300.sh" capture-window
  LAYERS="$window" "$SHARED_ENCODER/convert_b300.sh" encode-window
done

"$SHARED_ENCODER/convert_b300.sh" assemble
```

`encode-window` automatically creates and verifies the profile before LDLQ.
No extra quantization switch is required.

## Artifact contract

New checkpoints declare:

```json
{
  "rotation_layout": "shared_h_v1",
  "shared_h_tensor_schema": "model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}"
}
```

The ordinary expert tensor schema remains unchanged for Trellis and the
expert-local rotation side. A loader that does not recognize `shared_h_v1`
must reject the artifact. A checkpoint without `rotation_layout` is legacy
`per_expert_v1` and loads exactly as before.

## Validation gates

Run unit tests before a conversion:

```bash
pytest -q tests/test_glm52_exl3_shared_h_recipe.py
```

A release checkpoint still needs all of these full-model gates:

1. `MANIFEST.sha256` and source-payload audit pass.
2. Tensor schema contains 9,228 EXL3 tensors per MoE layer.
3. BF16-reference teacher-forced KLD is measured with the standard GLM logits.
4. Legacy and shared-H checkpoints both boot with the release image.
5. TP4 decode, prefill, CUDA-graph replay and tool-call sanity pass.

The layer-40 POC measured `672.36 MiB/GPU` saved, no CUDA-graph latency
regression, and essentially unchanged activation error (`-0.057%` relative),
but it is not a substitute for the full-checkpoint KLD gate.
