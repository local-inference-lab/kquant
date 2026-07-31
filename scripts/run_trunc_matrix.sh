#!/usr/bin/env bash
# Run deterministic truncated-model serving probes. Process ownership, port/GPU
# preflight, manifests, and cleanup are handled by run_k3_correctness.py.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
RUNNER="${SCRIPT_DIR}/run_k3_correctness.py"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/out/correctness}"
VLLM_DIR="${VLLM_DIR:-/home/luke/projects/vllm}"
MODEL_DIR="${MODEL_DIR:-/models/k3-trunc-stock}"
MODEL_TAG="${MODEL_TAG:-stock}"
TP_SIZES="${TP_SIZES:-4 12}"

mkdir -p "${OUT_ROOT}"
declare -a RESULTS=()

for tp in ${TP_SIZES}; do
  tag="${MODEL_TAG}-tp${tp}"
  run_dir="${OUT_ROOT}/$(date +%Y%m%dT%H%M%S%N)-${tag}"
  "${PYTHON_BIN}" "${RUNNER}" run \
    --tag "${tag}" \
    --model "${MODEL_DIR}" \
    --tp "${tp}" \
    --vllm-dir "${VLLM_DIR}" \
    --run-dir "${run_dir}"
  RESULTS+=("${run_dir}")
done

reference="${RESULTS[0]}"
for result in "${RESULTS[@]:1}"; do
  "${PYTHON_BIN}" "${RUNNER}" compare "${reference}" "${result}" \
    --output "${result}/comparison-to-$(basename "${reference}").json"
done

printf '%s\n' "${RESULTS[@]}"
