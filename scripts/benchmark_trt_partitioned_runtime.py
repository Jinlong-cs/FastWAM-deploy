#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tinyaction_fastwam.paths import FASTWAM_RELEASE_DIR, OUTPUTS_DIR
from tinyaction_fastwam.presets import DEFAULT_PRESET, PRESETS, get_preset
from tinyaction_fastwam.sample_adapter import load_offline_robotwin_sample
from tinyaction_fastwam.text_cache import load_text_context_from_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark partitioned FastWAM TensorRT runtime on AGX.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--vae-engine", type=Path, default=OUTPUTS_DIR / "trt/vae_image_encoder_fp16_patched.engine")
    parser.add_argument("--video-prefill-engine", type=Path, default=OUTPUTS_DIR / "trt/video_prefill_fp16.engine")
    parser.add_argument("--action-step-engine", type=Path, default=OUTPUTS_DIR / "trt/action_step_dynamic_kv_fp16.engine")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=FASTWAM_RELEASE_DIR / "robotwin_uncond_3cam_384.pt",
        help="FastWAM checkpoint used to extract the proprio encoder for real-state samples.",
    )
    parser.add_argument(
        "--proprio-encoder-cache",
        type=Path,
        default=OUTPUTS_DIR / "runtime/proprio_encoder.pt",
        help="Small cached proprio_encoder weight/bias extracted from the released checkpoint.",
    )
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=FASTWAM_RELEASE_DIR / "robotwin_uncond_3cam_384_dataset_stats.json",
    )
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "trt/partitioned_trt_runtime_fp16.json")
    parser.add_argument("--sample", type=Path, help="Optional real/offline RoboTwin sample .npz/.json.")
    parser.add_argument("--instruction", default=None, help="Override sample instruction for text-cache lookup.")
    parser.add_argument("--text-cache-dir", type=Path, help="Optional precomputed FastWAM T5 text embedding cache.")
    parser.add_argument("--text-cache-encoder-id", default="wan22ti2v5b")
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--context-len-with-proprio", type=int, default=129)
    parser.add_argument("--text-dim", type=int, default=4096)
    parser.add_argument("--scheduler-shift", type=float, default=5.0)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


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
    raise ValueError(f"Unsupported TensorRT dtype: {dtype}")


def torch_randn(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)


def torch_rand_image(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    data = torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0
    return data.to(device=device, dtype=dtype)


def cuda_memory() -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {
            "mem_get_info_free_bytes": None,
            "mem_get_info_total_bytes": None,
            "torch_allocated_bytes": None,
            "torch_reserved_bytes": None,
            "torch_max_allocated_bytes": None,
            "torch_max_reserved_bytes": None,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "mem_get_info_free_bytes": int(free_bytes),
        "mem_get_info_total_bytes": int(total_bytes),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        "torch_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def rss_kib() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None


class TrtEngineRunner:
    def __init__(self, *, name: str, engine_path: Path, device: torch.device) -> None:
        import tensorrt as trt

        self.name = name
        self.engine_path = engine_path
        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        load_t0 = time.perf_counter()
        with engine_path.open("rb") as handle, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        self.deserialize_ms = (time.perf_counter() - load_t0) * 1000.0
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.tensors: dict[str, torch.Tensor] = {}
        self.io_meta: list[dict[str, object]] = []
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self._allocate_default_tensors()

    def _allocate_default_tensors(self) -> None:
        import tensorrt as trt

        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(tensor_name)
            trt_dtype = self.engine.get_tensor_dtype(tensor_name)
            shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(tensor_name))
            torch_dtype = trt_dtype_to_torch(trt_dtype)
            if mode == trt.TensorIOMode.INPUT:
                if not self.context.set_input_shape(tensor_name, shape):
                    raise RuntimeError(f"{self.name}: failed to set input shape for {tensor_name}: {shape}")
                self.input_names.append(tensor_name)
            else:
                self.output_names.append(tensor_name)
            if torch_dtype == torch.bool:
                tensor = torch.ones(shape, dtype=torch_dtype, device=self.device)
            else:
                tensor = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self.bind_tensor(tensor_name, tensor)
            self.io_meta.append(
                {
                    "index": idx,
                    "name": tensor_name,
                    "mode": str(mode),
                    "trt_dtype": str(trt_dtype),
                    "torch_dtype": str(torch_dtype),
                    "shape": list(shape),
                }
            )
        missing = list(self.context.infer_shapes())
        if missing:
            raise RuntimeError(f"{self.name}: TensorRT context shape inference incomplete: {missing}")

    def bind_tensor(self, tensor_name: str, tensor: torch.Tensor) -> None:
        self.tensors[tensor_name] = tensor
        self.context.set_tensor_address(tensor_name, int(tensor.data_ptr()))

    def run(self, stream: torch.cuda.Stream) -> None:
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"{self.name}: TensorRT execute_async_v3 returned False")


def build_inference_schedule(
    *,
    num_inference_steps: int,
    num_train_timesteps: int,
    shift: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[float]]:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
    sigma_steps = shift * u_steps / (1.0 + (shift - 1.0) * u_steps)
    timesteps = (sigma_steps[:-1] * float(num_train_timesteps)).to(dtype=dtype)
    deltas = (sigma_steps[1:] - sigma_steps[:-1]).detach().cpu().tolist()
    return timesteps, [float(delta) for delta in deltas]


def load_action_denorm_stats(path: Path, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, str]:
    if not path.exists():
        mean = torch.zeros((1, 32, 14), dtype=torch.float32, device=device)
        std = torch.ones((1, 32, 14), dtype=torch.float32, device=device)
        return mean, std, "identity_missing_dataset_stats"
    data = json.loads(path.read_text())
    stats = data["action"]["default"]
    mean_values = stats.get("global_mean")
    std_values = stats.get("global_std")
    if mean_values is None or std_values is None:
        raise ValueError(f"Dataset stats missing action.default global_mean/global_std: {path}")
    mean = torch.as_tensor(mean_values, dtype=torch.float32, device=device).view(1, 1, -1)
    std = torch.as_tensor(std_values, dtype=torch.float32, device=device).view(1, 1, -1)
    return mean, std, "z_score_global_action_default"


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def build_fastwam_input_image(
    observation: dict[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    obs_data = observation["observation"]
    head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
    left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
    right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
    bottom = np.concatenate([left, right], axis=1)
    image = np.concatenate([head, bottom], axis=0)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return tensor * (2.0 / 255.0) - 1.0


def normalize_state_from_stats(
    state: np.ndarray,
    *,
    dataset_stats: Path,
    device: torch.device,
) -> torch.Tensor:
    data = json.loads(dataset_stats.read_text())
    stats = data["state"]["default"]
    mean = torch.as_tensor(stats["global_mean"], dtype=torch.float32, device=device).view(1, -1)
    std = torch.as_tensor(stats["global_std"], dtype=torch.float32, device=device).view(1, -1)
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
    return torch.clamp((state_tensor - mean) / (std + 1e-8), -5.0, 5.0)


def load_proprio_encoder_payload(
    *,
    checkpoint: Path,
    cache_path: Path,
) -> tuple[dict[str, torch.Tensor], str]:
    if cache_path.exists():
        payload = torch.load(str(cache_path), map_location="cpu")
        source = "cached_proprio_encoder"
    else:
        if not checkpoint.exists():
            raise FileNotFoundError(f"FastWAM checkpoint not found for proprio encoder extraction: {checkpoint}")
        checkpoint_obj = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint_obj, dict) or "proprio_encoder" not in checkpoint_obj:
            raise KeyError(f"Checkpoint missing top-level proprio_encoder: {checkpoint}")
        payload = {
            key: value.detach().cpu().clone()
            for key, value in dict(checkpoint_obj["proprio_encoder"]).items()
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(cache_path))
        del checkpoint_obj
        gc.collect()
        source = "extracted_from_checkpoint"
    if "weight" not in payload or "bias" not in payload:
        raise KeyError(f"proprio_encoder payload must contain weight and bias: {cache_path}")
    return payload, source


def append_proprio_to_context(
    *,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: torch.Tensor,
    checkpoint: Path,
    cache_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    payload, source = load_proprio_encoder_payload(checkpoint=checkpoint, cache_path=cache_path)
    weight = payload["weight"].to(device=context.device, dtype=context.dtype)
    bias = payload["bias"].to(device=context.device, dtype=context.dtype)
    proprio_token = torch.nn.functional.linear(proprio.to(dtype=context.dtype), weight, bias).unsqueeze(1)
    proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
    return torch.cat([context, proprio_token], dim=1), torch.cat([context_mask, proprio_mask], dim=1), source


def build_real_runtime_inputs(
    args: argparse.Namespace,
    *,
    image_dtype: torch.dtype,
    context_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, object]:
    preset = get_preset(args.preset)
    observation, sample_instruction = load_offline_robotwin_sample(args.sample, preset=preset)
    instruction = args.instruction or sample_instruction
    input_image = build_fastwam_input_image(observation, device=device, dtype=image_dtype)
    raw_state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    proprio = normalize_state_from_stats(raw_state, dataset_stats=args.dataset_stats, device=device)
    if args.text_cache_dir is not None:
        text_context = load_text_context_from_cache(
            cache_dir=args.text_cache_dir,
            instruction=instruction,
            context_len=128,
            encoder_id=args.text_cache_encoder_id,
            device=device,
            dtype=context_dtype,
        )
        context = text_context.context
        context_mask = text_context.context_mask
        text_context_mode = "precomputed_t5_cache"
        semantic_text_encoder = "precomputed_cache"
        text_cache_path = str(text_context.cache_path)
        prompt = text_context.prompt
    else:
        context = torch.zeros((1, 128, args.text_dim), dtype=context_dtype, device=device)
        context_mask = torch.ones((1, 128), dtype=torch.bool, device=device)
        text_context_mode = "synthetic_zero_context"
        semantic_text_encoder = "disabled"
        text_cache_path = None
        prompt = None
    context, context_mask, proprio_source = append_proprio_to_context(
        context=context,
        context_mask=context_mask,
        proprio=proprio,
        checkpoint=args.checkpoint,
        cache_path=args.proprio_encoder_cache,
    )
    return {
        "input_image": input_image,
        "context": context,
        "context_mask": context_mask,
        "sample_source": f"offline_sample:{args.sample.expanduser().resolve()}",
        "instruction": instruction,
        "text_context": text_context_mode,
        "semantic_text_encoder": semantic_text_encoder,
        "text_cache_path": text_cache_path,
        "prompt": prompt,
        "proprio_source": proprio_source,
        "proprio_encoder_cache": str(args.proprio_encoder_cache.expanduser().resolve()),
        "raw_state_shape": [int(dim) for dim in raw_state.shape],
        "normalized_proprio_shape": [int(dim) for dim in proprio.shape],
    }


def time_cuda_stage(stream: torch.cuda.Stream, fn: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    fn()
    end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end))


def finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item()) if tensor.is_floating_point() else True


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT benchmark.")
    for path in (args.vae_engine, args.video_prefill_engine, args.action_step_engine):
        if not path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {path}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"TensorRT benchmark requires a CUDA device, got: {device}")
    device_index = 0 if device.index is None else int(device.index)
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    memory_before = cuda_memory()
    vae = TrtEngineRunner(name="vae_image_encoder", engine_path=args.vae_engine, device=device)
    memory_after_vae = cuda_memory()
    video = TrtEngineRunner(name="video_prefill", engine_path=args.video_prefill_engine, device=device)
    memory_after_video = cuda_memory()
    action = TrtEngineRunner(name="action_step_dynamic_kv", engine_path=args.action_step_engine, device=device)
    memory_after_action = cuda_memory()

    stream = torch.cuda.Stream(device=device)
    real_input_metadata: dict[str, object] = {}
    if args.sample is not None:
        real_inputs = build_real_runtime_inputs(
            args,
            image_dtype=vae.tensors["input_image"].dtype,
            context_dtype=action.tensors["context"].dtype,
            device=device,
        )
        input_image = real_inputs.pop("input_image")
        context = real_inputs.pop("context")
        context_mask = real_inputs.pop("context_mask")
        real_input_metadata = real_inputs
        if tuple(input_image.shape) != tuple(vae.tensors["input_image"].shape):
            raise ValueError(
                f"Real sample input_image shape {tuple(input_image.shape)} does not match VAE engine "
                f"{tuple(vae.tensors['input_image'].shape)}"
            )
        if tuple(context.shape) != tuple(action.tensors["context"].shape):
            raise ValueError(
                f"Real sample context shape {tuple(context.shape)} does not match action engine "
                f"{tuple(action.tensors['context'].shape)}"
            )
        if tuple(context_mask.shape) != tuple(action.tensors["context_mask"].shape):
            raise ValueError(
                f"Real sample context_mask shape {tuple(context_mask.shape)} does not match action engine "
                f"{tuple(action.tensors['context_mask'].shape)}"
            )
    else:
        input_image = torch_rand_image(
            tuple(vae.tensors["input_image"].shape),
            dtype=vae.tensors["input_image"].dtype,
            device=device,
            seed=args.seed,
        )
        context = torch.zeros(
            (1, args.context_len_with_proprio, args.text_dim),
            dtype=action.tensors["context"].dtype,
            device=device,
        )
        context_mask = torch.ones((1, args.context_len_with_proprio), dtype=torch.bool, device=device)
        real_input_metadata = {
            "sample_source": "synthetic_tensor_contract",
            "instruction": None,
            "text_context": "synthetic_zero_context",
            "semantic_text_encoder": "disabled",
            "text_cache_path": None,
            "prompt": None,
            "proprio_source": "synthetic_zero_context_includes_proprio_slot",
            "proprio_encoder_cache": None,
        }
    latents_initial = torch_randn(
        (1, args.action_horizon, args.action_dim),
        dtype=action.tensors["latents_action"].dtype,
        device=device,
        seed=args.seed + 17,
    )
    latents_action = torch.empty_like(latents_initial)
    pred_action_half = torch.empty_like(latents_action)
    action_decoded = torch.empty((1, args.action_horizon, args.action_dim), dtype=torch.float32, device=device)
    action_mean, action_std, denorm_mode = load_action_denorm_stats(args.dataset_stats, device=device)
    timesteps, deltas = build_inference_schedule(
        num_inference_steps=args.num_inference_steps,
        num_train_timesteps=args.num_train_timesteps,
        shift=args.scheduler_shift,
        device=device,
        dtype=action.tensors["timestep_action"].dtype,
    )

    vae.bind_tensor("input_image", input_image)
    video.bind_tensor("first_frame_latents", vae.tensors["first_frame_latents"])
    video.bind_tensor("context", context)
    video.bind_tensor("context_mask", context_mask)
    action.bind_tensor("latents_action", latents_action)
    action.bind_tensor("context", context)
    action.bind_tensor("context_mask", context_mask)

    kv_cast_buffers: dict[str, torch.Tensor] = {}
    direct_kv_bindings = 0
    cast_kv_bindings = 0
    for layer_idx in range(30):
        for suffix in ("k", "v"):
            name = f"video_{suffix}_{layer_idx}"
            source = video.tensors[name]
            expected = action.tensors[name]
            if source.dtype == expected.dtype:
                action.bind_tensor(name, source)
                direct_kv_bindings += 1
            else:
                cast_buffer = torch.empty_like(expected)
                kv_cast_buffers[name] = cast_buffer
                action.bind_tensor(name, cast_buffer)
                cast_kv_bindings += 1

    stage_values: dict[str, list[float]] = {
        "vae_image_encoder_ms": [],
        "video_prefill_ms": [],
        "kv_cast_ms": [],
        "action_denoise_loop_ms": [],
        "action_step_engine_total_ms": [],
        "scheduler_total_ms": [],
        "action_decode_ms": [],
        "end_to_end_ms": [],
    }
    output_cpu_shape: list[int] | None = None

    def run_once(measure: bool) -> dict[str, float]:
        per_iter: dict[str, float] = {}
        torch.cuda.synchronize(device)
        wall_t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            latents_action.copy_(latents_initial)
            per_iter["vae_image_encoder_ms"] = time_cuda_stage(stream, lambda: vae.run(stream))
            per_iter["video_prefill_ms"] = time_cuda_stage(stream, lambda: video.run(stream))

            def cast_kv() -> None:
                for name, target in kv_cast_buffers.items():
                    target.copy_(video.tensors[name])

            per_iter["kv_cast_ms"] = time_cuda_stage(stream, cast_kv)

            action_engine_total = 0.0
            scheduler_total = 0.0
            loop_start = torch.cuda.Event(enable_timing=True)
            loop_end = torch.cuda.Event(enable_timing=True)
            loop_start.record(stream)
            for step_idx, delta in enumerate(deltas):
                action.tensors["timestep_action"].copy_(timesteps[step_idx : step_idx + 1])
                action_engine_total += time_cuda_stage(stream, lambda: action.run(stream))

                def scheduler_step(delta_value: float = delta) -> None:
                    pred_action_half.copy_(action.tensors["pred_action"])
                    latents_action.add_(pred_action_half, alpha=delta_value)

                scheduler_total += time_cuda_stage(stream, scheduler_step)
            loop_end.record(stream)
            loop_end.synchronize()
            per_iter["action_denoise_loop_ms"] = float(loop_start.elapsed_time(loop_end))
            per_iter["action_step_engine_total_ms"] = action_engine_total
            per_iter["scheduler_total_ms"] = scheduler_total

            def decode_action() -> None:
                action_decoded.copy_(latents_action.to(dtype=torch.float32) * action_std + action_mean)

            per_iter["action_decode_ms"] = time_cuda_stage(stream, decode_action)
        stream.synchronize()
        action_cpu = action_decoded.detach().cpu()
        per_iter["end_to_end_ms"] = (time.perf_counter() - wall_t0) * 1000.0
        nonlocal output_cpu_shape
        output_cpu_shape = [int(dim) for dim in action_cpu.shape]
        if measure:
            for key, value in per_iter.items():
                stage_values[key].append(float(value))
        return per_iter

    for _ in range(args.warmup_batches):
        run_once(measure=False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.measure_batches):
        run_once(measure=True)

    stage_summary_ms = {key: summarize(values) for key, values in stage_values.items()}
    final_finite = {
        "first_frame_latents": finite_tensor(vae.tensors["first_frame_latents"]),
        "video_k_0": finite_tensor(video.tensors["video_k_0"]),
        "video_v_0": finite_tensor(video.tensors["video_v_0"]),
        "pred_action": finite_tensor(action.tensors["pred_action"]),
        "latents_action": finite_tensor(latents_action),
        "action_decoded": finite_tensor(action_decoded),
    }
    memory_after_benchmark = cuda_memory()

    return {
        "runtime_mode": "partitioned_tensorrt_fastwam",
        "backend": "tensorrt",
        "precision": "fp16_engines_with_fp32_selected_outputs",
        "sample_source": real_input_metadata.get("sample_source"),
        "instruction": real_input_metadata.get("instruction"),
        "text_context": real_input_metadata.get("text_context"),
        "semantic_text_encoder": real_input_metadata.get("semantic_text_encoder"),
        "text_cache_path": real_input_metadata.get("text_cache_path"),
        "prompt": real_input_metadata.get("prompt"),
        "real_input_metadata": real_input_metadata,
        "vae_engine": str(args.vae_engine.resolve()),
        "video_prefill_engine": str(args.video_prefill_engine.resolve()),
        "action_step_engine": str(args.action_step_engine.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint.exists() else None,
        "dataset_stats": str(args.dataset_stats.resolve()) if args.dataset_stats.exists() else None,
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
        "num_inference_steps": args.num_inference_steps,
        "action_horizon": args.action_horizon,
        "action_dim": args.action_dim,
        "scheduler": {
            "num_train_timesteps": args.num_train_timesteps,
            "shift": args.scheduler_shift,
            "timestep_dtype": str(timesteps.dtype),
            "deltas": deltas,
        },
        "deserialization_ms": {
            "vae_image_encoder": vae.deserialize_ms,
            "video_prefill": video.deserialize_ms,
            "action_step_dynamic_kv": action.deserialize_ms,
            "total": vae.deserialize_ms + video.deserialize_ms + action.deserialize_ms,
        },
        "engine_size_bytes": {
            "vae_image_encoder": args.vae_engine.stat().st_size,
            "video_prefill": args.video_prefill_engine.stat().st_size,
            "action_step_dynamic_kv": args.action_step_engine.stat().st_size,
        },
        "kv_binding": {
            "direct_fp16_bindings": direct_kv_bindings,
            "cast_to_fp16_bindings": cast_kv_bindings,
            "cast_buffer_names": sorted(kv_cast_buffers),
        },
        "stage_summary_ms": stage_summary_ms,
        "mean_end_to_end_ms": stage_summary_ms["end_to_end_ms"]["mean"],
        "p50_end_to_end_ms": stage_summary_ms["end_to_end_ms"]["p50"],
        "p95_end_to_end_ms": stage_summary_ms["end_to_end_ms"]["p95"],
        "output_shape": output_cpu_shape,
        "output_finite": all(final_finite.values()),
        "finite_checks": final_finite,
        "denormalize_mode": denorm_mode,
        "memory": {
            "before_load": memory_before,
            "after_vae_load": memory_after_vae,
            "after_video_prefill_load": memory_after_video,
            "after_action_step_load": memory_after_action,
            "after_benchmark": memory_after_benchmark,
            "process_ru_maxrss_kib": rss_kib(),
        },
        "io": {
            "vae_image_encoder": vae.io_meta,
            "video_prefill_num_tensors": len(video.io_meta),
            "action_step_dynamic_kv_num_tensors": len(action.io_meta),
        },
        "note": "This benchmark loads VAE, video prefill, and dynamic-KV action-step TensorRT engines in one process. Scheduler and action denormalization remain host-side. When --sample is set, image preprocessing and state/proprio-token construction use a real offline RoboTwin sample; semantic text still depends on whether --text-cache-dir is provided.",
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = benchmark(args)
        result["status"] = "success"
    except Exception as exc:
        result = {
            "status": "failed",
            "runtime_mode": "partitioned_tensorrt_fastwam",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("status") != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
