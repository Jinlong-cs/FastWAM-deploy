#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_trt_partitioned_runtime import (
    TrtEngineRunner,
    build_real_runtime_inputs,
    load_action_denorm_stats,
)
from tinyaction_fastwam.paths import FASTWAM_RELEASE_DIR, OUTPUTS_DIR, UPSTREAM_FASTWAM_DIR
from tinyaction_fastwam.presets import DEFAULT_PRESET, PRESETS, get_preset
from tinyaction_fastwam.runtime import load_robotwin_policy, resolve_runtime_paths
from tinyaction_fastwam.sample_adapter import load_offline_robotwin_sample


SELECTED_KV_LAYERS = (0, 15, 29)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one real RoboTwin + text-cache sample between FastWAM eager "
            "and the partitioned TensorRT runtime."
        )
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--upstream-dir", type=Path, default=UPSTREAM_FASTWAM_DIR)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=FASTWAM_RELEASE_DIR / "robotwin_uncond_3cam_384.pt",
    )
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=FASTWAM_RELEASE_DIR / "robotwin_uncond_3cam_384_dataset_stats.json",
    )
    parser.add_argument("--vae-engine", type=Path, default=OUTPUTS_DIR / "trt/vae_image_encoder_fp16_patched.engine")
    parser.add_argument("--video-prefill-engine", type=Path, default=OUTPUTS_DIR / "trt/video_prefill_fp16.engine")
    parser.add_argument("--action-step-engine", type=Path, default=OUTPUTS_DIR / "trt/action_step_dynamic_kv_fp16.engine")
    parser.add_argument(
        "--proprio-encoder-cache",
        type=Path,
        default=OUTPUTS_DIR / "runtime/proprio_encoder.pt",
    )
    parser.add_argument("--sample", type=Path, required=True, help="Real/offline RoboTwin sample .npz/.json.")
    parser.add_argument("--instruction", default=None, help="Override sample instruction for text-cache lookup.")
    parser.add_argument("--text-cache-dir", type=Path, required=True, help="Precomputed FastWAM T5 text embedding cache.")
    parser.add_argument("--text-cache-encoder-id", default="wan22ti2v5b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "validation/partitioned_trt_drift_real_text_cache.json")
    return parser.parse_args()


def sync(device: str | torch.device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "finite": bool(torch.isfinite(tensor).all().item()),
        "min": float(tensor.min().item()) if tensor.numel() else 0.0,
        "max": float(tensor.max().item()) if tensor.numel() else 0.0,
        "mean": float(tensor.mean().item()) if tensor.numel() else 0.0,
        "std": float(tensor.std(unbiased=False).item()) if tensor.numel() else 0.0,
    }


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().to(device="cpu", dtype=torch.float32)
    candidate = candidate.detach().to(device="cpu", dtype=torch.float32)
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "status": "shape_mismatch",
            "reference_shape": [int(dim) for dim in reference.shape],
            "candidate_shape": [int(dim) for dim in candidate.shape],
        }

    diff = candidate - reference
    abs_diff = diff.abs()
    denom = torch.maximum(reference.abs(), candidate.abs()).clamp_min(1.0e-6)
    rel = abs_diff / denom
    rmse = torch.sqrt(torch.mean(diff * diff)).item() if diff.numel() else 0.0
    ref_flat = reference.reshape(-1)
    cand_flat = candidate.reshape(-1)
    ref_norm = torch.linalg.vector_norm(ref_flat)
    cand_norm = torch.linalg.vector_norm(cand_flat)
    cosine = None
    if ref_norm.item() > 0.0 and cand_norm.item() > 0.0:
        cosine = float(torch.dot(ref_flat, cand_flat).item() / (ref_norm.item() * cand_norm.item()))
    return {
        "status": "compared",
        "shape": [int(dim) for dim in reference.shape],
        "reference_finite": bool(torch.isfinite(reference).all().item()),
        "candidate_finite": bool(torch.isfinite(candidate).all().item()),
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "rmse": float(rmse),
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel": float(rel.mean().item()) if rel.numel() else 0.0,
        "cosine": cosine,
    }


def load_real_sample(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    preset = get_preset(args.preset)
    observation, sample_instruction = load_offline_robotwin_sample(args.sample, preset=preset)
    instruction = args.instruction or sample_instruction
    sample_source = f"offline_sample:{args.sample.expanduser().resolve()}"
    return observation, instruction, sample_source


@torch.no_grad()
def run_eager_reference(
    args: argparse.Namespace,
    *,
    observation: dict[str, Any],
    instruction: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    runtime_paths = resolve_runtime_paths(
        preset_name=args.preset,
        upstream_dir=args.upstream_dir,
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    load_start = time.perf_counter()
    policy = load_robotwin_policy(
        preset_name=args.preset,
        upstream_dir=args.upstream_dir,
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        device=args.device,
        mixed_precision=args.mixed_precision,
        num_inference_steps=args.num_inference_steps,
        action_horizon=args.action_horizon,
        replan_steps=24,
        seed=args.seed,
        rand_device="cpu",
        timing_enabled=False,
        text_cache_dir=args.text_cache_dir,
        text_cache_instruction=instruction,
        text_cache_encoder_id=args.text_cache_encoder_id,
    )
    model = policy.model
    sync(model.device)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    run_start = time.perf_counter()
    prepared = policy.preprocess_observation(observation)
    context, context_mask = policy._get_context()
    context, context_mask = model._append_proprio_to_context(
        context=context,
        context_mask=context_mask,
        proprio=prepared.proprio,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents_action = torch.randn(
        (1, args.action_horizon, model.action_expert.action_dim),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    latents_initial = latents_action.clone()

    first_frame_latents = model._encode_input_image_latents_tensor(
        input_image=prepared.image_tensor.to(device=model.device, dtype=model.torch_dtype),
        tiled=False,
    )
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    timestep_video = torch.zeros(
        (first_frame_latents.shape[0],),
        dtype=first_frame_latents.dtype,
        device=model.device,
    )
    video_pre = model.video_expert.pre_dit(
        x=first_frame_latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=fuse_flag,
    )
    video_seq_len = int(video_pre["tokens"].shape[1])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=latents_action.shape[1],
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device,
    )
    video_kv_cache = model.mot.prefill_video_cache(
        video_tokens=video_pre["tokens"],
        video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
        },
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
    )
    infer_timesteps, infer_deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=args.num_inference_steps,
        device=model.device,
        dtype=latents_action.dtype,
        shift_override=None,
    )

    pred_action_first: torch.Tensor | None = None
    pred_action_last: torch.Tensor | None = None
    for step_t, step_delta in zip(infer_timesteps, infer_deltas):
        timestep_action = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=model.device)
        pred_action = model._predict_action_noise_with_cache(
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        if pred_action_first is None:
            pred_action_first = pred_action.clone()
        pred_action_last = pred_action.clone()
        latents_action = model.infer_action_scheduler.step(pred_action, step_delta, latents_action)

    if pred_action_first is None or pred_action_last is None:
        raise RuntimeError("No eager denoise step was executed.")
    action_decoded_np = policy._denormalize_action(latents_action)[0]
    sync(model.device)
    run_ms = (time.perf_counter() - run_start) * 1000.0

    tensors: dict[str, torch.Tensor] = {
        "context_with_proprio": cpu_float(context),
        "latents_initial": cpu_float(latents_initial),
        "first_frame_latents": cpu_float(first_frame_latents),
        "pred_action_first": cpu_float(pred_action_first),
        "pred_action_last": cpu_float(pred_action_last),
        "latents_action_final": cpu_float(latents_action),
        "action_decoded": torch.as_tensor(action_decoded_np, dtype=torch.float32).unsqueeze(0),
        "schedule_timesteps": cpu_float(infer_timesteps),
        "schedule_deltas": cpu_float(infer_deltas),
    }
    for layer_idx in SELECTED_KV_LAYERS:
        tensors[f"video_k_{layer_idx}"] = cpu_float(video_kv_cache[layer_idx]["k"])
        tensors[f"video_v_{layer_idx}"] = cpu_float(video_kv_cache[layer_idx]["v"])

    metadata = {
        "runtime_mode": "eager_reference",
        "backend": "pytorch",
        "mixed_precision": args.mixed_precision,
        "checkpoint": str(runtime_paths.checkpoint_path),
        "dataset_stats": str(runtime_paths.dataset_stats_path),
        "load_ms": load_ms,
        "run_ms": run_ms,
        "text_context": policy.get_text_context_metadata(),
        "video_seq_len": video_seq_len,
        "selected_kv_layers": list(SELECTED_KV_LAYERS),
    }

    del policy, model, video_kv_cache, video_pre
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return tensors, metadata


def bind_video_cache_to_action(
    *,
    video: TrtEngineRunner,
    action: TrtEngineRunner,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
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
    return kv_cast_buffers, {
        "direct_fp16_bindings": direct_kv_bindings,
        "cast_to_fp16_bindings": cast_kv_bindings,
        "cast_buffer_names": sorted(kv_cast_buffers),
    }


@torch.no_grad()
def run_trt_candidate(
    args: argparse.Namespace,
    *,
    eager_tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT drift validation.")
    for path in (args.vae_engine, args.video_prefill_engine, args.action_step_engine):
        if not path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {path}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"TensorRT drift validation requires CUDA device, got {device}")
    device_index = 0 if device.index is None else int(device.index)
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    load_start = time.perf_counter()
    memory_before_load = torch.cuda.mem_get_info(device)
    vae = TrtEngineRunner(name="vae_image_encoder", engine_path=args.vae_engine, device=device)
    video = TrtEngineRunner(name="video_prefill", engine_path=args.video_prefill_engine, device=device)
    action = TrtEngineRunner(name="action_step_dynamic_kv", engine_path=args.action_step_engine, device=device)
    sync(device)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    real_inputs = build_real_runtime_inputs(
        args,
        image_dtype=vae.tensors["input_image"].dtype,
        context_dtype=action.tensors["context"].dtype,
        device=device,
    )
    input_image = real_inputs.pop("input_image")
    context = real_inputs.pop("context")
    context_mask = real_inputs.pop("context_mask")
    latents_initial = eager_tensors["latents_initial"].to(
        device=device,
        dtype=action.tensors["latents_action"].dtype,
        non_blocking=True,
    )
    timesteps = eager_tensors["schedule_timesteps"].to(
        device=device,
        dtype=action.tensors["timestep_action"].dtype,
        non_blocking=True,
    )
    deltas = [float(v) for v in eager_tensors["schedule_deltas"].reshape(-1).tolist()]

    latents_action = torch.empty_like(action.tensors["latents_action"])
    pred_action_half = torch.empty_like(latents_action)
    action_decoded = torch.empty((1, args.action_horizon, 14), dtype=torch.float32, device=device)
    action_mean, action_std, denorm_mode = load_action_denorm_stats(args.dataset_stats, device=device)

    vae.bind_tensor("input_image", input_image)
    video.bind_tensor("first_frame_latents", vae.tensors["first_frame_latents"])
    video.bind_tensor("context", context)
    video.bind_tensor("context_mask", context_mask)
    action.bind_tensor("latents_action", latents_action)
    action.bind_tensor("context", context)
    action.bind_tensor("context_mask", context_mask)
    kv_cast_buffers, kv_binding = bind_video_cache_to_action(video=video, action=action)

    stream = torch.cuda.Stream(device=device)
    run_start = time.perf_counter()
    with torch.cuda.stream(stream):
        latents_action.copy_(latents_initial)
        vae.run(stream)
        video.run(stream)
        for name, target in kv_cast_buffers.items():
            target.copy_(video.tensors[name])
    stream.synchronize()

    pred_action_first: torch.Tensor | None = None
    pred_action_last: torch.Tensor | None = None
    for step_idx, delta in enumerate(deltas):
        with torch.cuda.stream(stream):
            action.tensors["timestep_action"].copy_(timesteps[step_idx : step_idx + 1])
            action.run(stream)
        stream.synchronize()
        if pred_action_first is None:
            pred_action_first = action.tensors["pred_action"].clone()
        pred_action_last = action.tensors["pred_action"].clone()
        with torch.cuda.stream(stream):
            pred_action_half.copy_(action.tensors["pred_action"])
            latents_action.add_(pred_action_half, alpha=delta)
        stream.synchronize()

    if pred_action_first is None or pred_action_last is None:
        raise RuntimeError("No TensorRT denoise step was executed.")
    with torch.cuda.stream(stream):
        action_decoded.copy_(latents_action.to(dtype=torch.float32) * action_std + action_mean)
    stream.synchronize()
    run_ms = (time.perf_counter() - run_start) * 1000.0
    memory_after_run = torch.cuda.mem_get_info(device)

    tensors: dict[str, torch.Tensor] = {
        "context_with_proprio": cpu_float(context),
        "latents_initial": cpu_float(latents_initial),
        "first_frame_latents": cpu_float(vae.tensors["first_frame_latents"]),
        "pred_action_first": cpu_float(pred_action_first),
        "pred_action_last": cpu_float(pred_action_last),
        "latents_action_final": cpu_float(latents_action),
        "action_decoded": cpu_float(action_decoded),
        "schedule_timesteps": cpu_float(timesteps),
        "schedule_deltas": eager_tensors["schedule_deltas"].clone(),
    }
    for layer_idx in SELECTED_KV_LAYERS:
        tensors[f"video_k_{layer_idx}"] = cpu_float(video.tensors[f"video_k_{layer_idx}"])
        tensors[f"video_v_{layer_idx}"] = cpu_float(video.tensors[f"video_v_{layer_idx}"])

    metadata = {
        "runtime_mode": "partitioned_tensorrt_candidate",
        "backend": "tensorrt",
        "precision": "fp16_engines",
        "load_ms": load_ms,
        "run_ms": run_ms,
        "engines": {
            "vae": str(args.vae_engine.resolve()),
            "video_prefill": str(args.video_prefill_engine.resolve()),
            "action_step": str(args.action_step_engine.resolve()),
        },
        "engine_size_bytes": {
            "vae": args.vae_engine.stat().st_size,
            "video_prefill": args.video_prefill_engine.stat().st_size,
            "action_step": args.action_step_engine.stat().st_size,
        },
        "real_input_metadata": real_inputs,
        "denormalize_mode": denorm_mode,
        "kv_binding": kv_binding,
        "memory_free_bytes": {
            "before_load": int(memory_before_load[0]),
            "after_run": int(memory_after_run[0]),
        },
    }
    del vae, video, action
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tensors, metadata


def build_result(
    *,
    args: argparse.Namespace,
    sample_source: str,
    instruction: str,
    eager_tensors: dict[str, torch.Tensor],
    trt_tensors: dict[str, torch.Tensor],
    eager_metadata: dict[str, Any],
    trt_metadata: dict[str, Any],
) -> dict[str, Any]:
    comparisons = {
        name: compare_tensors(eager_tensors[name], trt_tensors[name])
        for name in eager_tensors
        if name in trt_tensors
    }
    tensor_summaries = {
        "eager": {name: tensor_summary(value) for name, value in eager_tensors.items()},
        "trt": {name: tensor_summary(value) for name, value in trt_tensors.items()},
    }
    compared_values = [item for item in comparisons.values() if item.get("status") == "compared"]
    worst = max((float(item.get("max_abs", 0.0)) for item in compared_values), default=math.nan)
    return {
        "status": "success",
        "validation_mode": "eager_fp16_reference_vs_partitioned_trt_fp16_real_robotwin_text_cache",
        "preset": args.preset,
        "sample_source": sample_source,
        "instruction": instruction,
        "text_context": "precomputed_t5_cache",
        "num_inference_steps": args.num_inference_steps,
        "action_horizon": args.action_horizon,
        "seed": args.seed,
        "selected_kv_layers": list(SELECTED_KV_LAYERS),
        "eager_metadata": eager_metadata,
        "trt_metadata": trt_metadata,
        "comparisons": comparisons,
        "tensor_summaries": tensor_summaries,
        "summary": {
            "num_compared_tensors": len(compared_values),
            "worst_max_abs": worst,
            "all_compared_finite": all(
                bool(item.get("reference_finite", False)) and bool(item.get("candidate_finite", False))
                for item in compared_values
            ),
            "note": (
                "This is a numerical drift report, not a simulator success-rate report. "
                "TensorRT FP16 engines are compared against an eager FP16 reference on the same "
                "real RoboTwin sample, real T5 text cache, seed, initial action latent, and scheduler."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    observation, instruction, sample_source = load_real_sample(args)
    eager_tensors, eager_metadata = run_eager_reference(
        args,
        observation=observation,
        instruction=instruction,
    )
    trt_tensors, trt_metadata = run_trt_candidate(
        args,
        eager_tensors=eager_tensors,
    )
    result = build_result(
        args=args,
        sample_source=sample_source,
        instruction=instruction,
        eager_tensors=eager_tensors,
        trt_tensors=trt_tensors,
        eager_metadata=eager_metadata,
        trt_metadata=trt_metadata,
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
