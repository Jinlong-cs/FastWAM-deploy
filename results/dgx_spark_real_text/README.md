# DGX Spark Real-Text Baseline

Final DGX Spark reproduction results for the FastWAM deploy baseline.

Environment:

- Device: NVIDIA GB10 / DGX Spark
- Python: 3.12.3
- PyTorch: 2.11.0+cu130
- TensorRT Python: 10.16.1.11
- Sample: same real RoboTwin unified offline frame as AGX
- Text: same precomputed FastWAM T5 cache semantics as AGX
- Batch size: 1
- Action horizon: 32
- Action dimension: 14
- Inference steps: 1
- Warmup / measure batches: 2 / 5

Latency:

| Runtime | DGX mean E2E | DGX P95 E2E | AGX mean E2E | Speedup |
| --- | ---: | ---: | ---: | ---: |
| PyTorch eager `.pth` | 150.66 ms | 152.03 ms | 323.97 ms | 2.15x |
| Partitioned TensorRT FP16 | 80.79 ms | 81.95 ms | 134.57 ms | 1.67x |

Partitioned TensorRT stage means:

| Stage | AGX mean | DGX mean | AGX/DGX speedup |
| --- | ---: | ---: | ---: |
| VAE image encoder | 39.35 ms | 9.45 ms | 4.17x |
| Video prefill + K/V cache | 74.65 ms | 57.74 ms | 1.29x |
| K/V cast | 0.55 ms | 0.002 ms | 231.04x |
| Action denoise loop | 19.39 ms | 13.34 ms | 1.45x |
| Scheduler | 0.09 ms | 0.03 ms | 3.51x |
| Action decode / denorm | 0.21 ms | 0.04 ms | 5.12x |

Drift caveat:

- 15 tensors compared.
- All compared tensors are finite.
- Worst selected video K/V max absolute drift is 115.628.
- Final decoded action max absolute drift is 1.599 and mean absolute drift is 0.277.

Implementation notes:

- DGX did not have `trtexec`, so TensorRT engines were built with the Python builder wrapper.
- PyTorch 2.11 ONNX export rejected upstream complex RoPE expansion; export uses an ONNX-friendly real-valued RoPE patch without modifying upstream FastWAM.
- TensorRT parser rejected DOUBLE external initializers from the PyTorch 2.11 exporter; those ONNX initializers were converted to FLOAT before engine build.
- The upstream text cache filename uses `t5_len128`, while the saved payload may contain only the effective token length. Runtime now pads shorter cached text embeddings to the fixed 128-token context.

Figures:

- `fastwam_deployment_pipeline_dgx_spark.png`
- `fastwam_agx_vs_dgx_spark_comparison.png`

The AGX pipeline figure is stored at `docs/fastwam_deployment_pipeline_agx_orin.png`.
