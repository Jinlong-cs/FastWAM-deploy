# FastWAM Deploy

Deployment-oriented FastWAM runtime for RoboTwin on NVIDIA Jetson AGX Orin
and NVIDIA GB10 / DGX Spark.

This repository keeps upstream FastWAM unchanged and provides the small deploy
layer needed to run the official released RoboTwin checkpoint with:

- PyTorch eager inference from the original `.pth` checkpoint.
- Partitioned TensorRT FP16 inference for VAE image encoder, video prefill, and
  dynamic-KV action denoise step.
- Real RoboTwin offline sample input and precomputed FastWAM T5 text cache.
- Latency and drift reports for the current AGX and DGX Spark real-text baselines.

Upstream reference: [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM).

## File Structure

```text
FastWAM-deploy/
├── docs/
│   ├── fastwam_5module_pipeline.png
│   ├── fastwam_5module_pipeline.svg
│   ├── fastwam_deployment_pipeline_agx_orin.png
│   ├── fastwam_agx_deploy_report.pdf
│   └── fastwam_sample_contract.md
├── results/
│   ├── agx_real_text/                  # Final AGX real-text JSON results
│   └── dgx_spark_real_text/            # Final DGX Spark JSON results and figures
├── scripts/
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
│   ├── build_fastwam_summary_figures.py
│   └── run_dgx_spark_real_text_benchmark.sh
└── src/tinyaction_fastwam/              # Runtime adapters and deploy helpers
```

Large assets are intentionally not tracked:

- FastWAM released checkpoint and dataset stats.
- Wan/FastWAM model weights.
- ONNX files and TensorRT engines.
- RoboTwin data samples and text embedding caches.
- Intermediate benchmark outputs.

## Environment Setup

Use Python 3.10+. On Jetson, keep the NVIDIA-provided PyTorch/TensorRT stack and
do not install upstream FastWAM's x86 CUDA wheels.

```bash
pip install -e .
pip install -r requirements-agx.txt
```

Clone upstream FastWAM next to this repository or point `FASTWAM_UPSTREAM_DIR` to
your upstream checkout:

```bash
git clone https://github.com/yuantianyuan01/FastWAM ../FastWAM
export FASTWAM_UPSTREAM_DIR="$(pwd)/../FastWAM"
```

On AGX, the helper script sets NVMe-backed paths and Python import paths:

```bash
source scripts/fastwam_agx_env.sh
```

## Asset Preparation

Download the official FastWAM RoboTwin released checkpoint:

```bash
PYTHONPATH=src python scripts/download_fastwam_assets.py
PYTHONPATH=src python scripts/check_fastwam_assets.py
```

Expected files:

```text
artifacts/checkpoints/fastwam_release/
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

The VAE/model assets used by upstream FastWAM should be placed under the
directory referenced by `DIFFSYNTH_MODEL_BASE_PATH`.

## Real RoboTwin Sample and Text Cache

Prepare one RoboTwin unified frame as a local `.npz` sample:

```bash
PYTHONPATH=src python scripts/prepare_robotwin_unified_sample.py \
  --repo-id lerobot/robotwin_unified \
  --episode-index 0 \
  --frame-offset 0
```

The deploy sample contract is documented in
[`docs/fastwam_sample_contract.md`](docs/fastwam_sample_contract.md).

FastWAM text cache is generated with upstream `scripts/precompute_text_embeds.py`.
At runtime this repository only loads the prompt-specific `.pt` cache entry; the
T5 encoder is not resident in the TensorRT runtime.

## PyTorch Eager Benchmark

```bash
PYTHONPATH=src python scripts/benchmark_fastwam.py \
  --sample data/real_samples/robotwin_unified_ep000000_frame000000.npz \
  --text-cache-dir data/text_embeds_cache/robotwin \
  --num-inference-steps 1 \
  --warmup-batches 2 \
  --measure-batches 5 \
  --output outputs/benchmarks/robotwin_uncond_3cam_384_real_sample_text_cache_latency.json
```

## TensorRT Export and Runtime

The current deploy path uses three FP16 TensorRT engines plus host-side scheduler
and action denormalization:

```text
VAE image encoder engine
  -> video prefill + K/V cache engine
  -> dynamic-KV action denoise-step engine
  -> host-side flow-matching scheduler
  -> host-side action decode / denormalization
```

Export the three ONNX partitions:

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

Build TensorRT engines with `trtexec`:

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

If `trtexec` is not available, such as on the tested DGX Spark setup, use the
Python TensorRT builder wrapper:

```bash
PYTHONPATH=src python scripts/build_trt_engine_python.py \
  --onnx outputs/trt/video_prefill_fp16.onnx \
  --engine outputs/trt/video_prefill_fp16.engine \
  --precision fp16 \
  --builder-optimization-level 5 \
  --workspace-gib 32
```

Run the partitioned real-text benchmark:

```bash
PYTHONPATH=src python scripts/benchmark_trt_partitioned_runtime.py \
  --sample data/real_samples/robotwin_unified_ep000000_frame000000.npz \
  --text-cache-dir data/text_embeds_cache/robotwin \
  --num-inference-steps 1 \
  --warmup-batches 2 \
  --measure-batches 5 \
  --output outputs/trt/partitioned_trt_runtime_real_sample_text_cache.json
```

Run eager-vs-TRT drift validation:

```bash
PYTHONPATH=src python scripts/compare_partitioned_trt_drift.py \
  --sample data/real_samples/robotwin_unified_ep000000_frame000000.npz \
  --text-cache-dir data/text_embeds_cache/robotwin \
  --num-inference-steps 1 \
  --output outputs/validation/partitioned_trt_drift_real_text_cache.json
```

For convenience, the full real-text benchmark sequence can be run with:

```bash
PYTHONPATH=src PYTHON_BIN=python bash scripts/run_robotwin_text_benchmark.sh
```

## Current AGX and DGX Spark Results

Measured with the same FastWAM released RoboTwin checkpoint, real RoboTwin
sample, precomputed T5 cache semantics, batch size 1, action horizon 32, action
dimension 14, and `num_inference_steps=1`.

| Runtime | AGX mean E2E | DGX Spark mean E2E | Speedup | Notes |
| --- | ---: | ---: | ---: | --- |
| PyTorch eager `.pth` | 323.97 ms | 150.66 ms | 2.15x | BF16 eager baseline |
| Partitioned TensorRT FP16 | 134.57 ms | 80.79 ms | 1.67x | 3 engines + host scheduler/decode |

TensorRT stage means:

| Stage | AGX mean | DGX Spark mean | AGX/DGX speedup |
| --- | ---: | ---: | ---: |
| VAE image encoder | 39.35 ms | 9.45 ms | 4.17x |
| Video prefill + K/V cache | 74.65 ms | 57.74 ms | 1.29x |
| K/V cast | 0.55 ms | 0.002 ms | 231.04x |
| Action denoise loop | 19.39 ms | 13.34 ms | 1.45x |
| Scheduler | 0.09 ms | 0.03 ms | 3.51x |
| Action decode / denorm | 0.21 ms | 0.04 ms | 5.12x |

Numerical drift is still an accuracy gate, not solved by this runtime benchmark:

- 15 tensors compared; all compared tensors are finite.
- Worst selected video K/V `max_abs`: 115.629.
- Final decoded action `max_abs`: 1.598, `mean_abs`: 0.277.

This means the current TensorRT path is runtime-valid and finite-output-valid,
but simulator success-rate or accuracy-equivalence claims require further
drift reduction, likely via VAE/video-prefill precision localization or selective
FP32.

Final JSON outputs are kept under [`results/agx_real_text/`](results/agx_real_text/)
and [`results/dgx_spark_real_text/`](results/dgx_spark_real_text/). Summary
figures are:

- [`docs/fastwam_deployment_pipeline_agx_orin.png`](docs/fastwam_deployment_pipeline_agx_orin.png)
- [`results/dgx_spark_real_text/fastwam_deployment_pipeline_dgx_spark.png`](results/dgx_spark_real_text/fastwam_deployment_pipeline_dgx_spark.png)
- [`results/dgx_spark_real_text/fastwam_agx_vs_dgx_spark_comparison.png`](results/dgx_spark_real_text/fastwam_agx_vs_dgx_spark_comparison.png)

The AGX deployment report is
[`docs/fastwam_agx_deploy_report.pdf`](docs/fastwam_agx_deploy_report.pdf).

## Acknowledgements

This deployment wrapper builds on the official FastWAM implementation:

- FastWAM repository: <https://github.com/yuantianyuan01/FastWAM>
- FastWAM released model: <https://huggingface.co/yuanty/fastwam>
- RoboTwin unified dataset: <https://huggingface.co/datasets/lerobot/robotwin_unified>

## License

This repository is released under the MIT License. Upstream FastWAM and
third-party assets retain their original licenses.
