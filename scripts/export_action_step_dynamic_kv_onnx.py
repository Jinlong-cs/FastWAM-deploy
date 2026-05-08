#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch

from tinyaction_fastwam.paths import OUTPUTS_DIR, UPSTREAM_FASTWAM_DIR
from tinyaction_fastwam.onnx_patches import install_export_safe_rope
from tinyaction_fastwam.presets import DEFAULT_PRESET, PRESETS, get_preset
from tinyaction_fastwam.runtime import load_robotwin_policy, make_synthetic_robotwin_observation


class DynamicKVActionStepWrapper(torch.nn.Module):
    """One action denoise step with video K/V cache provided as inputs."""

    def __init__(
        self,
        model: Any,
        *,
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.video_seq_len = int(video_seq_len)
        self.register_buffer("attention_mask", attention_mask.detach(), persistent=False)

    def _video_kv_cache(self, kv_inputs: tuple[torch.Tensor, ...], dtype: torch.dtype) -> list[dict[str, torch.Tensor]]:
        cache = []
        expected = self.model.mot.num_layers * 2
        if len(kv_inputs) != expected:
            raise ValueError(f"Expected {expected} K/V tensors, got {len(kv_inputs)}.")
        for layer_idx in range(self.model.mot.num_layers):
            k = kv_inputs[layer_idx * 2].to(dtype=dtype)
            v = kv_inputs[layer_idx * 2 + 1].to(dtype=dtype)
            cache.append({"k": k, "v": v})
        return cache

    def forward(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        *kv_inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.model._predict_action_noise_with_cache(
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=self._video_kv_cache(kv_inputs, latents_action.dtype),
            attention_mask=self.attention_mask,
            video_seq_len=self.video_seq_len,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FastWAM action denoise step with dynamic video K/V inputs.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--upstream-dir", type=Path, default=UPSTREAM_FASTWAM_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "trt/action_step_dynamic_kv_fp16.onnx")
    parser.add_argument(
        "--status-output",
        type=Path,
        default=OUTPUTS_DIR / "trt/action_step_dynamic_kv_export_status_fp16.json",
    )
    parser.add_argument(
        "--constant-folding",
        action="store_true",
        help="Enable PyTorch ONNX constant folding. Disabled by default on Jetson to avoid CPU/CUDA folding failures.",
    )
    return parser.parse_args()


def truncate_text(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated {len(text) - limit} chars>"


def build_export_state(args: argparse.Namespace) -> tuple[DynamicKVActionStepWrapper, tuple[torch.Tensor, ...], dict[str, object]]:
    preset = get_preset(args.preset)
    with torch.inference_mode():
        policy = load_robotwin_policy(
            preset_name=args.preset,
            upstream_dir=args.upstream_dir,
            device=args.device,
            mixed_precision=args.mixed_precision,
            num_inference_steps=1,
            seed=args.seed,
            rand_device="cpu",
            timing_enabled=False,
        )
        observation = make_synthetic_robotwin_observation(preset=preset, seed=args.seed)
        prepared = policy.preprocess_observation(observation)
        model = policy.model
        model.requires_grad_(False)

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        latents_action = torch.randn(
            (1, policy.action_horizon, model.action_expert.action_dim),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=model.device, dtype=model.torch_dtype)
        timestep_action = torch.ones((1,), dtype=latents_action.dtype, device=model.device)

        context, context_mask = policy._get_context()
        if prepared.proprio is not None:
            context, context_mask = model._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=prepared.proprio,
            )
        first_frame_latents = model._encode_input_image_latents_tensor(
            input_image=prepared.image_tensor,
            tiled=False,
        )
        timestep_video = torch.zeros((first_frame_latents.shape[0],), dtype=first_frame_latents.dtype, device=model.device)
        video_pre = model.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)),
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
        kv_inputs = []
        for layer in video_kv_cache:
            kv_inputs.extend([layer["k"].detach(), layer["v"].detach()])

        wrapper = DynamicKVActionStepWrapper(
            model=model,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        ).to(device=model.device, dtype=model.torch_dtype).eval()
        inputs = (
            latents_action.detach(),
            timestep_action.detach(),
            context.detach(),
            context_mask.detach(),
            *kv_inputs,
        )
        with torch.inference_mode():
            eager_out = wrapper(*inputs)
    meta = {
        "latents_action_shape": list(latents_action.shape),
        "timestep_action_shape": list(timestep_action.shape),
        "context_shape": list(context.shape),
        "context_mask_shape": list(context_mask.shape),
        "video_seq_len": video_seq_len,
        "attention_mask_shape": list(attention_mask.shape),
        "num_kv_layers": len(video_kv_cache),
        "num_kv_inputs": len(kv_inputs),
        "kv0_k_shape": list(kv_inputs[0].shape),
        "kv0_v_shape": list(kv_inputs[1].shape),
        "eager_output_shape": list(eager_out.shape),
        "eager_output_finite": bool(torch.isfinite(eager_out).all().item()),
    }
    return wrapper, inputs, meta


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.status_output.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "preset": args.preset,
        "runtime_mode": "trt_export_action_step_dynamic_kv",
        "output": str(args.output),
        "opset": args.opset,
        "mixed_precision": args.mixed_precision,
        "safe_rope_patch": True,
    }
    try:
        install_export_safe_rope()
        wrapper, inputs, meta = build_export_state(args)
        status["meta"] = meta
        input_names = ["latents_action", "timestep_action", "context", "context_mask"]
        for layer_idx in range(int(meta["num_kv_layers"])):
            input_names.extend([f"video_k_{layer_idx}", f"video_v_{layer_idx}"])
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                inputs,
                str(args.output),
                input_names=input_names,
                output_names=["pred_action"],
                opset_version=args.opset,
                do_constant_folding=bool(args.constant_folding),
                dynamic_axes=None,
            )
        status["status"] = "success"
        status["size_bytes"] = args.output.stat().st_size
        status["size_mib"] = round(args.output.stat().st_size / (1024 * 1024), 2)
    except Exception as exc:
        status["status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        status["traceback"] = truncate_text(traceback.format_exc())
    args.status_output.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if status.get("status") != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
