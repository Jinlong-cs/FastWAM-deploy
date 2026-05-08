#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tinyaction_fastwam.paths import OUTPUTS_DIR, UPSTREAM_FASTWAM_DIR
from tinyaction_fastwam.presets import DEFAULT_PRESET, PRESETS, get_preset
from tinyaction_fastwam.runtime import (
    load_robotwin_policy,
    make_synthetic_robotwin_observation,
    resolve_runtime_paths,
)
from tinyaction_fastwam.sample_adapter import load_offline_robotwin_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FastWAM RoboTwin released checkpoint on one offline sample.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--upstream-dir", type=Path, default=UPSTREAM_FASTWAM_DIR)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-stats", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--instruction", default="Pick up the object and place it at the target.")
    parser.add_argument("--sample", type=Path, help="Optional offline RoboTwin sample (.npz/.json).")
    parser.add_argument("--text-cache-dir", type=Path, help="Optional precomputed FastWAM T5 text embedding cache.")
    parser.add_argument("--text-cache-encoder-id", default="wan22ti2v5b")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def maybe_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    runtime_paths = resolve_runtime_paths(
        preset_name=args.preset,
        upstream_dir=args.upstream_dir,
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.sample is not None:
        observation, instruction = load_offline_robotwin_sample(args.sample, preset=preset)
        sample_source = f"offline_sample:{args.sample.expanduser().resolve()}"
    else:
        observation = make_synthetic_robotwin_observation(preset=preset, seed=args.seed)
        instruction = args.instruction
        sample_source = "synthetic_robotwin_contract"

    policy = load_robotwin_policy(
        preset_name=args.preset,
        upstream_dir=args.upstream_dir,
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        device=args.device,
        mixed_precision=args.mixed_precision,
        num_inference_steps=args.num_inference_steps,
        action_horizon=args.action_horizon,
        replan_steps=args.replan_steps,
        seed=args.seed,
        rand_device="cpu",
        timing_enabled=True,
        text_cache_dir=args.text_cache_dir,
        text_cache_instruction=instruction if args.text_cache_dir is not None else None,
        text_cache_encoder_id=args.text_cache_encoder_id,
    )

    with torch.no_grad():
        for _ in range(args.warmup_batches):
            prepared = policy.preprocess_observation(observation)
            policy.infer_preprocessed_action(prepared=prepared, instruction=instruction)
            maybe_sync(args.device)

        if hasattr(policy, "reset_timing_rollout"):
            policy.reset_timing_rollout()

        policy_latencies_ms: list[float] = []
        preprocess_latencies_ms: list[float] = []
        end_to_end_latencies_ms: list[float] = []
        for _ in range(args.measure_batches):
            maybe_sync(args.device)
            start = time.perf_counter()

            preprocess_start = time.perf_counter()
            prepared = policy.preprocess_observation(observation)
            maybe_sync(args.device)
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

            policy_start = time.perf_counter()
            policy.infer_preprocessed_action(prepared=prepared, instruction=instruction)
            maybe_sync(args.device)
            policy_ms = (time.perf_counter() - policy_start) * 1000.0

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            preprocess_latencies_ms.append(preprocess_ms)
            policy_latencies_ms.append(policy_ms)
            end_to_end_latencies_ms.append(elapsed_ms)

    if not end_to_end_latencies_ms:
        raise RuntimeError("No benchmark batches were measured.")

    timing_rollout = policy.get_timing_rollout() if hasattr(policy, "get_timing_rollout") else {}
    text_context_metadata = (
        policy.get_text_context_metadata() if hasattr(policy, "get_text_context_metadata") else {}
    )
    measured_batches = len(end_to_end_latencies_ms)
    stage_ms = {
        "image_preprocess_ms": timing_rollout.get("image_preprocess_s", 0.0) * 1000.0 / measured_batches,
        "state_normalize_ms": timing_rollout.get("state_normalize_s", 0.0) * 1000.0 / measured_batches,
        "context_ms": timing_rollout.get("context_s", 0.0) * 1000.0 / measured_batches,
        "model_forward_ms": timing_rollout.get("model_forward_s", 0.0) * 1000.0 / measured_batches,
        "action_postprocess_ms": timing_rollout.get("action_postprocess_s", 0.0) * 1000.0 / measured_batches,
    }
    result = {
        "preset": args.preset,
        "checkpoint": str(runtime_paths.checkpoint_path),
        "dataset_stats": str(runtime_paths.dataset_stats_path),
        "device": args.device,
        "backend": "pytorch",
        "mixed_precision": args.mixed_precision,
        "dtype": args.mixed_precision,
        "runtime_mode": "eager_pth",
        "batch_size": 1,
        "sample_source": sample_source,
        **text_context_metadata,
        "instruction": instruction,
        "warmup_batches": args.warmup_batches,
        "measure_batches": measured_batches,
        "num_inference_steps": args.num_inference_steps,
        "mean_preprocess_ms": statistics.fmean(preprocess_latencies_ms),
        "mean_policy_ms": statistics.fmean(policy_latencies_ms),
        "mean_end_to_end_ms": statistics.fmean(end_to_end_latencies_ms),
        "p50_end_to_end_ms": percentile(end_to_end_latencies_ms, 0.50),
        "p95_end_to_end_ms": percentile(end_to_end_latencies_ms, 0.95),
        "p50_policy_ms": percentile(policy_latencies_ms, 0.50),
        "p95_policy_ms": percentile(policy_latencies_ms, 0.95),
        **stage_ms,
    }
    output = args.output or (OUTPUTS_DIR / "benchmarks" / f"{args.preset}_latency.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
