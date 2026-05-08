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


class VideoPrefillWrapper(torch.nn.Module):
    """ONNX wrapper for video pre_dit plus MoT video KV prefill."""

    def __init__(self, model: Any, *, action_seq_len: int) -> None:
        super().__init__()
        self.model = model
        self.action_seq_len = int(action_seq_len)

    def forward(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=first_frame_latents.device,
        )
        video_pre = self.model.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.model.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        # For video-only prefill with one latent frame, first_frame_causal
        # reduces to a full video self-attention mask. Avoid exporting the
        # upstream in-place mask construction because PyTorch ONNX hits an
        # internal index_put shape-inference assertion there.
        video_attention_mask = torch.ones(
            (video_seq_len, video_seq_len),
            dtype=torch.bool,
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
        )
        outputs: list[torch.Tensor] = []
        for layer in video_kv_cache:
            outputs.append(layer["k"])
            outputs.append(layer["v"])
        return tuple(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FastWAM video prefill/KV cache partition to ONNX.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--upstream-dir", type=Path, default=UPSTREAM_FASTWAM_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "trt/video_prefill_fp16.onnx")
    parser.add_argument(
        "--status-output",
        type=Path,
        default=OUTPUTS_DIR / "trt/video_prefill_export_status_fp16.json",
    )
    parser.add_argument(
        "--constant-folding",
        action="store_true",
        help="Enable PyTorch ONNX constant folding. Disabled by default on Jetson for this large graph.",
    )
    return parser.parse_args()


def truncate_text(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated {len(text) - limit} chars>"


def build_export_state(args: argparse.Namespace) -> tuple[VideoPrefillWrapper, tuple[torch.Tensor, ...], dict[str, object]]:
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
        wrapper = VideoPrefillWrapper(model=model, action_seq_len=policy.action_horizon).to(
            device=model.device,
            dtype=model.torch_dtype,
        ).eval()
        outputs = wrapper(first_frame_latents, context, context_mask)
    meta = {
        "first_frame_latents_shape": list(first_frame_latents.shape),
        "context_shape": list(context.shape),
        "context_mask_shape": list(context_mask.shape),
        "num_outputs": len(outputs),
        "num_kv_layers": len(outputs) // 2,
        "kv0_k_shape": list(outputs[0].shape),
        "kv0_v_shape": list(outputs[1].shape),
        "all_outputs_finite": all(bool(torch.isfinite(out).all().item()) for out in outputs),
    }
    return wrapper, (first_frame_latents.detach(), context.detach(), context_mask.detach()), meta


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.status_output.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "preset": args.preset,
        "runtime_mode": "trt_export_video_prefill",
        "output": str(args.output),
        "opset": args.opset,
        "mixed_precision": args.mixed_precision,
        "safe_rope_patch": True,
    }
    try:
        install_export_safe_rope()
        wrapper, inputs, meta = build_export_state(args)
        status["meta"] = meta
        output_names = []
        for layer_idx in range(int(meta["num_kv_layers"])):
            output_names.extend([f"video_k_{layer_idx}", f"video_v_{layer_idx}"])
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                inputs,
                str(args.output),
                input_names=["first_frame_latents", "context", "context_mask"],
                output_names=output_names,
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
