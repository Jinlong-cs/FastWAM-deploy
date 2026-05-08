# AGX Real-Text Baseline

Final published JSON results for the current FastWAM deploy baseline.

Environment:

- Device: NVIDIA Jetson AGX Orin
- Sample: real RoboTwin unified offline frame
- Text: precomputed FastWAM T5 cache
- Batch size: 1
- Action horizon: 32
- Action dimension: 14
- Inference steps: 1

Latency:

| Runtime | Mean E2E | P95 E2E | Output finite |
| --- | ---: | ---: | --- |
| PyTorch eager `.pth` | 323.97 ms | 324.40 ms | n/a |
| Partitioned TensorRT FP16 | 134.57 ms | 134.69 ms | true |

Partitioned TensorRT stage means:

| Stage | Mean latency |
| --- | ---: |
| VAE image encoder | 39.35 ms |
| Video prefill + K/V cache | 74.65 ms |
| K/V cast | 0.55 ms |
| Action denoise loop | 19.39 ms |
| Scheduler | 0.09 ms |
| Action decode / denorm | 0.21 ms |

Drift caveat:

- 15 tensors compared.
- All compared tensors are finite.
- Worst selected video K/V max absolute drift is 115.629.
- Final decoded action max absolute drift is 1.598 and mean absolute drift is 0.277.

The TensorRT path is runtime-valid and finite-output-valid, but simulator
accuracy validation is still pending.
