#!/usr/bin/env bash
set -euo pipefail

export TINYACTION_FASTWAM_ROOT="${TINYACTION_FASTWAM_ROOT:-/home/airbot/wujinlong}"
export FASTWAM_DEPLOY_ROOT="${FASTWAM_DEPLOY_ROOT:-$TINYACTION_FASTWAM_ROOT/fastwam-deploy}"
export FASTWAM_UPSTREAM_DIR="${FASTWAM_UPSTREAM_DIR:-$TINYACTION_FASTWAM_ROOT/FastWAM}"
export PYTHON_BIN="${PYTHON_BIN:-$TINYACTION_FASTWAM_ROOT/pi0.5/.venv/bin/python}"

cd "$FASTWAM_DEPLOY_ROOT"
source scripts/fastwam_agx_env.sh

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$FASTWAM_DEPLOY_ROOT/outputs/dgx_spark_runtime_runs/$RUN_ID}"
SAMPLE="${SAMPLE:-$FASTWAM_DEPLOY_ROOT/data/real_samples/robotwin_unified_ep000000_frame000000.npz}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-$FASTWAM_DEPLOY_ROOT/data/text_embeds_cache/robotwin}"

mkdir -p "$RUN_DIR" "$RUN_DIR/logs" "$FASTWAM_DEPLOY_ROOT/outputs/trt"

python_cmd() {
  PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" "$@"
}

if [ ! -f "$SAMPLE" ]; then
  python_cmd scripts/prepare_robotwin_unified_sample.py \
    --repo-id lerobot/robotwin_unified \
    --episode-index 0 \
    --frame-offset 0
fi

INSTRUCTION="$(python_cmd - <<'PY'
from pathlib import Path
import numpy as np
sample = Path("data/real_samples/robotwin_unified_ep000000_frame000000.npz")
with np.load(sample, allow_pickle=True) as data:
    print(str(data["instruction"].item() if data["instruction"].shape == () else data["instruction"]))
PY
)"

if ! find "$TEXT_CACHE_DIR" -maxdepth 1 -name '*.t5_len128.wan22ti2v5b.pt' -print -quit | grep -q .; then
  (
    cd "$FASTWAM_UPSTREAM_DIR"
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" scripts/precompute_text_embeds.py \
      task=robotwin_uncond_3cam_384_1e-4 \
      "override_instruction=$INSTRUCTION" \
      "data.train.text_embedding_cache_dir=$TEXT_CACHE_DIR" \
      "data.val.text_embedding_cache_dir=$TEXT_CACHE_DIR" \
      overwrite=false
  ) 2>&1 | tee "$RUN_DIR/logs/precompute_text_embeds.log"
fi

python_cmd scripts/export_vae_image_encoder_onnx.py \
  --output outputs/trt/vae_image_encoder_fp16_patched.onnx \
  --status-output "$RUN_DIR/vae_image_encoder_export_status_fp16_patched.json" \
  --device cuda --mixed-precision fp16 2>&1 | tee "$RUN_DIR/logs/export_vae_image_encoder.log"

python_cmd scripts/export_video_prefill_onnx.py \
  --output outputs/trt/video_prefill_fp16.onnx \
  --status-output "$RUN_DIR/video_prefill_export_status_fp16.json" \
  --device cuda --mixed-precision fp16 2>&1 | tee "$RUN_DIR/logs/export_video_prefill.log"

python_cmd scripts/export_action_step_dynamic_kv_onnx.py \
  --output outputs/trt/action_step_dynamic_kv_fp16.onnx \
  --status-output "$RUN_DIR/action_step_dynamic_kv_export_status_fp16.json" \
  --device cuda --mixed-precision fp16 2>&1 | tee "$RUN_DIR/logs/export_action_step_dynamic_kv.log"

python_cmd scripts/fix_onnx_double_initializers.py \
  outputs/trt/video_prefill_fp16.onnx \
  --summary-json "$RUN_DIR/video_prefill_double_init_fix.json" \
  2>&1 | tee "$RUN_DIR/logs/fix_video_prefill_double_initializers.log"

python_cmd scripts/fix_onnx_double_initializers.py \
  outputs/trt/action_step_dynamic_kv_fp16.onnx \
  --summary-json "$RUN_DIR/action_step_double_init_fix.json" \
  2>&1 | tee "$RUN_DIR/logs/fix_action_step_double_initializers.log"

python_cmd scripts/build_trt_engine_python.py \
  --onnx outputs/trt/vae_image_encoder_fp16_patched.onnx \
  --engine outputs/trt/vae_image_encoder_fp16_patched.engine \
  --precision fp16 --builder-optimization-level 5 --workspace-gib 16 \
  --summary-json "$RUN_DIR/vae_image_encoder_engine_summary.json" 2>&1 | tee "$RUN_DIR/logs/build_vae_image_encoder.log"

python_cmd scripts/build_trt_engine_python.py \
  --onnx outputs/trt/video_prefill_fp16.onnx \
  --engine outputs/trt/video_prefill_fp16.engine \
  --precision fp16 --builder-optimization-level 5 --workspace-gib 32 \
  --summary-json "$RUN_DIR/video_prefill_engine_summary.json" 2>&1 | tee "$RUN_DIR/logs/build_video_prefill.log"

python_cmd scripts/build_trt_engine_python.py \
  --onnx outputs/trt/action_step_dynamic_kv_fp16.onnx \
  --engine outputs/trt/action_step_dynamic_kv_fp16.engine \
  --precision fp16 --builder-optimization-level 5 --workspace-gib 16 \
  --summary-json "$RUN_DIR/action_step_dynamic_kv_engine_summary.json" 2>&1 | tee "$RUN_DIR/logs/build_action_step_dynamic_kv.log"

python_cmd scripts/benchmark_fastwam.py \
  --sample "$SAMPLE" \
  --text-cache-dir "$TEXT_CACHE_DIR" \
  --num-inference-steps 1 \
  --warmup-batches 2 \
  --measure-batches 5 \
  --output "$RUN_DIR/eager_real_text_latency.json" 2>&1 | tee "$RUN_DIR/logs/benchmark_eager.log"

python_cmd scripts/benchmark_trt_partitioned_runtime.py \
  --sample "$SAMPLE" \
  --text-cache-dir "$TEXT_CACHE_DIR" \
  --num-inference-steps 1 \
  --warmup-batches 2 \
  --measure-batches 5 \
  --output "$RUN_DIR/partitioned_trt_real_text_latency.json" 2>&1 | tee "$RUN_DIR/logs/benchmark_partitioned_trt.log"

python_cmd scripts/compare_partitioned_trt_drift.py \
  --sample "$SAMPLE" \
  --text-cache-dir "$TEXT_CACHE_DIR" \
  --num-inference-steps 1 \
  --output "$RUN_DIR/partitioned_trt_drift_real_text.json" 2>&1 | tee "$RUN_DIR/logs/compare_partitioned_trt_drift.log"

python_cmd scripts/build_fastwam_summary_figures.py \
  --agx-eager results/agx_real_text/eager_real_text_latency.json \
  --agx-trt results/agx_real_text/partitioned_trt_real_text_latency.json \
  --agx-drift results/agx_real_text/partitioned_trt_drift_real_text.json \
  --dgx-eager "$RUN_DIR/eager_real_text_latency.json" \
  --dgx-trt "$RUN_DIR/partitioned_trt_real_text_latency.json" \
  --dgx-drift "$RUN_DIR/partitioned_trt_drift_real_text.json" \
  --output-dir "$RUN_DIR" \
  --docs-output docs/fastwam_deployment_pipeline_agx_orin.png

echo "$RUN_DIR"
