#!/usr/bin/env bash
# Boot each (model, tp) config of the truncated-K3 matrix, probe, record.
set -u
SCRATCH=/tmp/claude-1000/-home-luke-projects-kquant/ea779040-a191-46b7-8e07-61d45f78bfc5/scratchpad
OUT=$SCRATCH/trunc-matrix-results.txt
: > $OUT

probe() {
  local tag=$1
  timeout 600 curl -s http://127.0.0.1:8011/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"trunc","prompt":"The capital of France is Paris. The capital of Germany is","max_tokens":8,"temperature":0,"logprobs":5,"echo":true}' \
    > $SCRATCH/trunc-$tag.json
  /home/luke/projects/kquant/.venv/bin/python - "$tag" <<'EOF' >> $OUT
import json, sys
tag = sys.argv[1]
try:
    r = json.load(open(f"/tmp/claude-1000/-home-luke-projects-kquant/ea779040-a191-46b7-8e07-61d45f78bfc5/scratchpad/trunc-{tag}.json"))
    c = r["choices"][0]
    lp = c["logprobs"]
    comp = "".join(lp["tokens"][-8:])
    prompt_lp = [x for x in lp["token_logprobs"][1:12] if x is not None]
    import statistics
    print(f"{tag}: completion={comp!r} prompt_lp_mean={statistics.mean(prompt_lp):.2f}")
except Exception as e:
    print(f"{tag}: PROBE FAILED {e}")
EOF
}

run_cfg() {
  local tag=$1 model=$2 tp=$3
  echo "=== $tag (tp=$tp) ===" | tee -a $OUT
  cd /home/luke/projects/vllm
  KIMI_DEBUG_NORMS=1 PYTHONPATH=/home/luke/projects/vllm \
  CUTE_DSL_ARCH=sm_120a CUDA_DEVICE_MAX_CONNECTIONS=32 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_B12X_MOE=1 \
  VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  KDA_DISABLE_AUTOTUNE=1 B12X_MOE_FORCE_A16=1 VLLM_USE_B12X_FP8_GEMM=1 \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
  ./.venv/bin/python -m vllm.entrypoints.cli.main serve "$model" \
    --served-model-name trunc --trust-remote-code \
    --host 127.0.0.1 --port 8011 \
    --max-model-len 2048 --tensor-parallel-size "$tp" --enforce-eager \
    --load-format safetensors \
    --gpu-memory-utilization 0.35 \
    --max-num-batched-tokens 512 --max-num-seqs 1 \
    > $SCRATCH/trunc-$tag.log 2>&1 &
  local srv=$!
  local t0=$(date +%s)
  until grep -q 'Application startup complete' $SCRATCH/trunc-$tag.log 2>/dev/null; do
    if ! kill -0 $srv 2>/dev/null; then echo "$tag: BOOT FAILED" | tee -a $OUT; return 1; fi
    if [ $(( $(date +%s) - t0 )) -gt 1500 ]; then echo "$tag: BOOT TIMEOUT" | tee -a $OUT; kill $srv; return 1; fi
    sleep 10
  done
  probe "$tag"
  grep '\[norms\]' $SCRATCH/trunc-$tag.log | tail -12 >> $OUT
  kill $srv 2>/dev/null
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill $p 2>/dev/null; done
  sleep 10
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 $p 2>/dev/null; done
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | sort -rn | head -1 | awk '{print $1}')" -lt 100 ]; do sleep 5; done
}

true # ours-tp12 done
true # ours-tp4 done
if [ -d /models/k3-trunc-stock ] && [ -f /models/k3-trunc-stock/model.safetensors.index.json ]; then
  run_cfg stock-tp12 /models/k3-trunc-stock 12
  run_cfg stock-tp4  /models/k3-trunc-stock 4
fi
echo "MATRIX DONE"
cat $OUT
