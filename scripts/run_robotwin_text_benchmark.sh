#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAMPLE="${SAMPLE:-data/real_samples/robotwin_unified_ep000000_frame000000.npz}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-data/text_embeds_cache/robotwin}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
WARMUP_BATCHES="${WARMUP_BATCHES:-2}"
MEASURE_BATCHES="${MEASURE_BATCHES:-5}"

if [[ -f scripts/fastwam_agx_env.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/fastwam_agx_env.sh
fi

if [[ ! -f "${SAMPLE}" ]]; then
  echo "Missing RoboTwin sample: ${SAMPLE}" >&2
  exit 1
fi

if [[ ! -d "${TEXT_CACHE_DIR}" ]]; then
  echo "Missing FastWAM text cache dir: ${TEXT_CACHE_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/benchmark_fastwam.py \
  --sample "${SAMPLE}" \
  --text-cache-dir "${TEXT_CACHE_DIR}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --warmup-batches "${WARMUP_BATCHES}" \
  --measure-batches "${MEASURE_BATCHES}" \
  --output outputs/benchmarks/robotwin_uncond_3cam_384_real_sample_text_cache_latency.json

"${PYTHON_BIN}" scripts/benchmark_trt_partitioned_runtime.py \
  --sample "${SAMPLE}" \
  --text-cache-dir "${TEXT_CACHE_DIR}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --warmup-batches "${WARMUP_BATCHES}" \
  --measure-batches "${MEASURE_BATCHES}" \
  --output outputs/trt/partitioned_trt_runtime_real_sample_text_cache.json

"${PYTHON_BIN}" scripts/compare_partitioned_trt_drift.py \
  --sample "${SAMPLE}" \
  --text-cache-dir "${TEXT_CACHE_DIR}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --output outputs/validation/partitioned_trt_drift_real_text_cache.json
