# FastWAM Deploy

Public deployment layer for running the official FastWAM RoboTwin checkpoint
with PyTorch eager bring-up and partitioned TensorRT benchmarking.

This repository keeps upstream FastWAM unchanged. Clone upstream FastWAM next
to this repo, download the public assets, prepare a RoboTwin sample and text
cache, then run the export, TensorRT build, benchmark, and drift validation
commands below.

Upstream reference: [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM).

## File Structure

```text
FastWAM-deploy/
├── docs/
│   └── fastwam_sample_contract.md        # Offline RoboTwin sample and runtime tensor contract
├── fixtures/
│   └── fastwam_minimal_sample.json       # Synthetic public fixture for CLI validation
├── scripts/
│   ├── check_minimal_fixture.py
│   ├── download_fastwam_assets.py
│   ├── check_fastwam_assets.py
│   ├── prepare_robotwin_unified_sample.py
│   ├── benchmark_fastwam.py
│   ├── export_vae_image_encoder_onnx.py
│   ├── export_video_prefill_onnx.py
│   ├── export_action_step_dynamic_kv_onnx.py
│   ├── fix_onnx_double_initializers.py
│   ├── build_trt_engine_python.py
│   ├── benchmark_trt_partitioned_runtime.py
│   ├── compare_partitioned_trt_drift.py
│   └── fastwam_agx_env.sh
└── src/tinyaction_fastwam/               # Runtime adapters and deploy helpers
```

Large private or generated assets are intentionally not tracked:

- FastWAM released checkpoints and dataset stats.
- Wan / FastWAM weights.
- ONNX exports and TensorRT engines.
- Real RoboTwin samples.
- Precomputed text embedding caches.
- Raw benchmark output files and internal reports.

## Runtime Layout

The current deployment path uses three TensorRT engines plus host-side scheduler
and action denormalization:

```text
RoboTwin sample + FastWAM preprocessing
  -> vae_image_encoder.engine
  -> video_prefill.engine
  -> action_step_dynamic_kv.engine
  -> host-side flow-matching scheduler
  -> host-side action decode / denormalization
```

Default public preset:

- preset: `robotwin_uncond_3cam_384`
- upstream task: `robotwin_uncond_3cam_384_1e-4`
- checkpoint repo: `yuanty/fastwam`
- cameras: `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- composed image tensor: `[1, 3, 384, 320]`
- text context tensor: `[1, 128, 4096]`
- action horizon: `32`
- action dimension: `14`
- benchmark batch size: `1`

## Environment Setup

Use Python 3.10+. On Jetson, keep the NVIDIA-provided PyTorch and TensorRT
stack instead of installing upstream x86 CUDA wheels.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements-agx.txt
```

Clone upstream FastWAM next to this repository:

```bash
git clone https://github.com/yuantianyuan01/FastWAM ../FastWAM
export FASTWAM_UPSTREAM_DIR="$(pwd)/../FastWAM"
```

On AGX, the generic helper sets workspace, cache, and import paths from
environment variables:

```bash
source scripts/fastwam_agx_env.sh
```

Validate the synthetic public fixture without model weights or private data:

```bash
python3 scripts/check_minimal_fixture.py
```

## Asset Download And Check

Download the official released RoboTwin checkpoint and dataset stats:

```bash
PYTHONPATH=src python scripts/download_fastwam_assets.py
PYTHONPATH=src python scripts/check_fastwam_assets.py
```

Expected checkpoint layout:

```text
artifacts/checkpoints/fastwam_release/
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

Place the Wan/FastWAM model files used by upstream FastWAM under the directory
referenced by `DIFFSYNTH_MODEL_BASE_PATH`.

## Public RoboTwin Sample

Prepare one public `lerobot/robotwin_unified` frame as a local offline sample:

```bash
PYTHONPATH=src python scripts/prepare_robotwin_unified_sample.py \
  --repo-id lerobot/robotwin_unified \
  --episode-index 0 \
  --frame-offset 0
```

The offline sample contract is documented in
[`docs/fastwam_sample_contract.md`](docs/fastwam_sample_contract.md).

## Text Cache Preparation

FastWAM uses a T5 text context. This deploy repo loads a prompt-specific cache
entry at runtime; it does not keep the T5 encoder resident in the TensorRT
runtime.

Generate text embeddings with upstream FastWAM, then place the generated `.pt`
cache files under `data/text_embeds_cache/robotwin`:

```bash
cd "$FASTWAM_UPSTREAM_DIR"
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
cd -
```

The cache filename is derived from the FastWAM prompt string and should match:

```text
<sha256(prompt)>.t5_len128.wan22ti2v5b.pt
```

## PyTorch Eager Benchmark

Use eager mode for bring-up and reference checks before TensorRT export:

```bash
PYTHONPATH=src python scripts/benchmark_fastwam.py \
  --sample data/real_samples/robotwin_unified_ep000000_frame000000.npz \
  --text-cache-dir data/text_embeds_cache/robotwin \
  --num-inference-steps 1 \
  --warmup-batches 2 \
  --measure-batches 5 \
  --output outputs/benchmarks/robotwin_uncond_3cam_384_text_cache_latency.json
```

## ONNX Export

Export the three runtime partitions:

```bash
PYTHONPATH=src python scripts/export_vae_image_encoder_onnx.py \
  --output outputs/trt/vae_image_encoder_fp16_patched.onnx \
  --status-output outputs/trt/vae_image_encoder_export_status_fp16_patched.json \
  --device cuda --mixed-precision fp16

PYTHONPATH=src python scripts/export_video_prefill_onnx.py \
  --output outputs/trt/video_prefill_fp16.onnx \
  --status-output outputs/trt/video_prefill_export_status_fp16.json \
  --device cuda --mixed-precision fp16

PYTHONPATH=src python scripts/export_action_step_dynamic_kv_onnx.py \
  --output outputs/trt/action_step_dynamic_kv_fp16.onnx \
  --status-output outputs/trt/action_step_dynamic_kv_export_status_fp16.json \
  --device cuda --mixed-precision fp16
```

If TensorRT rejects DOUBLE initializers in an exported graph, patch that graph:

```bash
PYTHONPATH=src python scripts/fix_onnx_double_initializers.py \
  outputs/trt/action_step_dynamic_kv_fp16.onnx \
  --summary-json outputs/trt/action_step_dynamic_kv_double_fix_summary.json
```

## TensorRT Engine Build

Build FP16 engines with `trtexec` when it is available:

```bash
trtexec --onnx=outputs/trt/vae_image_encoder_fp16_patched.onnx \
  --saveEngine=outputs/trt/vae_image_encoder_fp16_patched.engine \
  --fp16 --builderOptimizationLevel=3 --memPoolSize=workspace:4096

trtexec --onnx=outputs/trt/video_prefill_fp16.onnx \
  --saveEngine=outputs/trt/video_prefill_fp16.engine \
  --fp16 --builderOptimizationLevel=3 --memPoolSize=workspace:4096

trtexec --onnx=outputs/trt/action_step_dynamic_kv_fp16.onnx \
  --saveEngine=outputs/trt/action_step_dynamic_kv_fp16.engine \
  --fp16 --builderOptimizationLevel=3 --memPoolSize=workspace:4096
```

When `trtexec` is unavailable, use the TensorRT Python builder wrapper:

```bash
PYTHONPATH=src python scripts/build_trt_engine_python.py \
  --onnx outputs/trt/video_prefill_fp16.onnx \
  --engine outputs/trt/video_prefill_fp16.engine \
  --precision fp16 \
  --builder-optimization-level 5 \
  --workspace-gib 32
```

The wrapper also exposes `--precision int8` for experimental TensorRT builds.
The public summary below uses FP16 engines; uncalibrated INT8 engine builds
should not be treated as accuracy-valid PTQ baselines.

## Partitioned Runtime Benchmark

Run the TensorRT partitioned runtime on the same sample and text cache:

```bash
PYTHONPATH=src python scripts/benchmark_trt_partitioned_runtime.py \
  --sample data/real_samples/robotwin_unified_ep000000_frame000000.npz \
  --text-cache-dir data/text_embeds_cache/robotwin \
  --num-inference-steps 1 \
  --warmup-batches 2 \
  --measure-batches 5 \
  --output outputs/trt/partitioned_trt_runtime_text_cache.json
```

## Drift Validation

Compare eager FP16 reference tensors with the partitioned TensorRT runtime on
the same sample, text cache, seed, and scheduler:

```bash
PYTHONPATH=src python scripts/compare_partitioned_trt_drift.py \
  --sample data/real_samples/robotwin_unified_ep000000_frame000000.npz \
  --text-cache-dir data/text_embeds_cache/robotwin \
  --num-inference-steps 1 \
  --output outputs/validation/partitioned_trt_drift_text_cache.json
```

## Public Latency Summary

These numbers are compact summaries copied from prior AGX Orin and DGX Spark
benchmark runs. The raw benchmark files, private sample, text cache, engines,
and internal reports are not bundled in this public repository.

| Runtime | AGX Orin mean E2E | DGX Spark mean E2E | DGX speedup | Notes |
| --- | ---: | ---: | ---: | --- |
| PyTorch eager BF16 `.pth` | 323.97 ms | 150.66 ms | 2.15x | Bring-up/reference path |
| Partitioned TensorRT FP16 | 134.57 ms | 80.79 ms | 1.67x | 3 engines plus host scheduler/decode |

TensorRT stage summary from the same benchmark protocol:

| Stage | AGX Orin mean | DGX Spark mean | Speedup |
| --- | ---: | ---: | ---: |
| VAE image encoder | 39.35 ms | 9.45 ms | 4.17x |
| Video prefill + K/V cache | 74.65 ms | 57.74 ms | 1.29x |
| K/V cast | 0.55 ms | 0.002 ms | 231.04x |
| Action denoise loop | 19.39 ms | 13.34 ms | 1.45x |
| Scheduler | 0.09 ms | 0.03 ms | 3.51x |
| Action decode / denorm | 0.21 ms | 0.04 ms | 5.12x |

Drift caveat:

| Check | AGX Orin | DGX Spark | Interpretation |
| --- | ---: | ---: | --- |
| Compared tensors | 15 | 15 | Same validation scope |
| Worst max abs vs eager FP16 | 115.63 | 115.63 | Numerical drift gate remains open |
| Decoded action max abs vs eager FP16 | 1.60 | 1.60 | Not a simulator success-rate metric |
| Finite checks | pass | pass | Runtime produced finite tensors |

Use these summary values for public orientation only. For a new machine or
software stack, rerun the commands above and compare the freshly generated
local benchmark files.
