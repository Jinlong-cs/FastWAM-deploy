#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tinyaction_fastwam.paths import OUTPUTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a TensorRT FastWAM video prefill engine on AGX.")
    parser.add_argument("--engine", type=Path, default=OUTPUTS_DIR / "trt/video_prefill_fp16.engine")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "trt/video_prefill_trt_runtime_fp16.json")
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--measure-batches", type=int, default=50)
    parser.add_argument("--precision", default="fp16_engine")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def trt_dtype_to_torch(dtype: object) -> torch.dtype:
    import tensorrt as trt

    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.bool:
        return torch.bool
    raise ValueError(f"Unsupported TensorRT dtype for this benchmark: {dtype}")


def make_input_tensor(name: str, dtype: torch.dtype, shape: tuple[int, ...], device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + sum(ord(ch) for ch in name))
    if dtype == torch.bool:
        return torch.ones(shape, dtype=dtype, device=device)
    data = torch.randn(shape, generator=generator, dtype=torch.float32)
    return data.to(device=device, dtype=dtype)


def benchmark_engine(args: argparse.Namespace) -> dict[str, object]:
    import tensorrt as trt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT benchmark.")
    if not args.engine.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {args.engine}")

    logger = trt.Logger(trt.Logger.WARNING)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"TensorRT benchmark requires a CUDA device, got: {device}")
    device_index = 0 if device.index is None else int(device.index)
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    load_t0 = time.perf_counter()
    with args.engine.open("rb") as handle, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(handle.read())
    deserialize_ms = (time.perf_counter() - load_t0) * 1000.0
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {args.engine}")

    context = engine.create_execution_context()
    tensors: dict[str, torch.Tensor] = {}
    tensor_meta: list[dict[str, object]] = []
    output_names: list[str] = []
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        mode = engine.get_tensor_mode(name)
        trt_dtype = engine.get_tensor_dtype(name)
        shape = tuple(int(dim) for dim in engine.get_tensor_shape(name))
        torch_dtype = trt_dtype_to_torch(trt_dtype)
        if mode == trt.TensorIOMode.INPUT:
            if not context.set_input_shape(name, shape):
                raise RuntimeError(f"TensorRT failed to set input shape for {name}: {shape}")
            tensor = make_input_tensor(name=name, dtype=torch_dtype, shape=shape, device=device, seed=args.seed)
        else:
            tensor = torch.empty(shape, dtype=torch_dtype, device=device)
            output_names.append(name)
        tensors[name] = tensor
        context.set_tensor_address(name, int(tensor.data_ptr()))
        tensor_meta.append(
            {
                "index": idx,
                "name": name,
                "mode": str(mode),
                "trt_dtype": str(trt_dtype),
                "torch_dtype": str(torch_dtype),
                "shape": list(shape),
            }
        )

    missing_tensors = list(context.infer_shapes())
    if missing_tensors:
        raise RuntimeError(f"TensorRT context shape inference is incomplete: {missing_tensors}")

    stream = torch.cuda.Stream(device=device)

    def run_once() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
        end.record(stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False.")
        end.synchronize()
        return float(start.elapsed_time(end))

    for _ in range(args.warmup_batches):
        run_once()

    latencies = [run_once() for _ in range(args.measure_batches)]
    finite_outputs = []
    output_shapes = {}
    for name in output_names:
        tensor = tensors[name]
        output_shapes[name] = list(tensor.shape)
        if tensor.is_floating_point():
            finite_outputs.append(bool(torch.isfinite(tensor).all().item()))

    return {
        "runtime_mode": "tensorrt_video_prefill",
        "backend": "tensorrt",
        "precision": args.precision,
        "engine": str(args.engine.resolve()),
        "engine_size_bytes": args.engine.stat().st_size,
        "deserialize_ms": deserialize_ms,
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
        "mean_policy_ms": statistics.fmean(latencies),
        "p50_policy_ms": percentile(latencies, 0.50),
        "p95_policy_ms": percentile(latencies, 0.95),
        "min_policy_ms": min(latencies),
        "max_policy_ms": max(latencies),
        "num_outputs": len(output_names),
        "all_outputs_finite": all(finite_outputs) if finite_outputs else None,
        "output_shapes_sample": dict(list(output_shapes.items())[:4]),
        "tensors": tensor_meta,
        "note": "This benchmark covers only the TensorRT video prefill/KV cache partition, not full FastWAM end-to-end inference.",
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = benchmark_engine(args)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
