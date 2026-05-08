#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch

from tinyaction_fastwam.paths import OUTPUTS_DIR, UPSTREAM_FASTWAM_DIR
from tinyaction_fastwam.presets import DEFAULT_PRESET, PRESETS, get_preset
from tinyaction_fastwam.runtime import load_robotwin_policy, make_synthetic_robotwin_observation


class VAEImageEncoderWrapper(torch.nn.Module):
    """Trace wrapper for FastWAM first-frame VAE image encoding."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        video = input_image.to(device=self.model.device, dtype=self.model.torch_dtype).unsqueeze(2)
        return self.model.vae.model.encode(video, self.model.vae.scale)


def install_export_safe_vae_attention() -> None:
    """Replace SDPA in Wan VAE attention with explicit matmul for TensorRT parsing.

    PyTorch exports scaled_dot_product_attention as an ONNX If subgraph on this
    path. TensorRT 10.3 rejects that If because the two branches can have
    different ranks after Squeeze. The VAE block uses one attention head, so a
    direct matmul/softmax/matmul implementation is shape-equivalent for the
    fixed first-frame export.
    """

    def safe_attention_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(b * t, 1, c * 3, h * w).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (float(c) ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = x.reshape(b * t, h * w, c).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity

    import fastwam.models.wan22.wan_video_vae as vae_module

    vae_module.AttentionBlock.forward = safe_attention_forward


def install_export_safe_vae_primitives() -> None:
    """Patch VAE primitive ops that export into TensorRT-hostile shape graphs."""

    def safe_rms_norm_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        variance = torch.mean(x * x, dim=dim, keepdim=True)
        # Original FastWAM uses F.normalize(x, dim) * sqrt(dim). Since
        # variance is mean(x^2), rsqrt(variance) already includes sqrt(dim).
        x = x * torch.rsqrt(variance + 1e-12)
        return x * self.gamma + self.bias

    def safe_causal_conv3d_forward(
        self: torch.nn.Conv3d,
        x: torch.Tensor,
        cache_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_t_front, pad_t_back = self._padding
        if cache_x is not None and pad_t_front > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            pad_t_front = max(0, int(pad_t_front) - int(cache_x.shape[2]))

        if pad_t_front > 0:
            zeros = torch.zeros(
                (x.shape[0], x.shape[1], int(pad_t_front), x.shape[3], x.shape[4]),
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([zeros, x], dim=2)
        if pad_t_back > 0:
            zeros = torch.zeros(
                (x.shape[0], x.shape[1], int(pad_t_back), x.shape[3], x.shape[4]),
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, zeros], dim=2)

        if int(pad_w_left) != int(pad_w_right) or int(pad_h_top) != int(pad_h_bottom):
            x = F.pad(x, (int(pad_w_left), int(pad_w_right), int(pad_h_top), int(pad_h_bottom), 0, 0))
            spatial_padding = (0, 0, 0)
        else:
            spatial_padding = (0, int(pad_h_top), int(pad_w_left))
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            spatial_padding,
            self.dilation,
            self.groups,
        )

    import fastwam.models.wan22.wan_video_vae as vae_module

    vae_module.RMS_norm.forward = safe_rms_norm_forward
    vae_module.CausalConv3d.forward = safe_causal_conv3d_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FastWAM first-frame VAE image encoder to ONNX.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--upstream-dir", type=Path, default=UPSTREAM_FASTWAM_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "trt/vae_image_encoder_fp16.onnx")
    parser.add_argument(
        "--status-output",
        type=Path,
        default=OUTPUTS_DIR / "trt/vae_image_encoder_export_status_fp16.json",
    )
    parser.add_argument(
        "--constant-folding",
        action="store_true",
        help="Enable PyTorch ONNX constant folding. Disabled by default on Jetson to avoid mixed-device folding failures.",
    )
    parser.add_argument(
        "--no-safe-attention-patch",
        action="store_true",
        help="Disable the deploy-side VAE attention export patch.",
    )
    parser.add_argument(
        "--no-safe-primitive-patch",
        action="store_true",
        help="Disable deploy-side VAE RMSNorm/CausalConv3d export patches.",
    )
    return parser.parse_args()


def truncate_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated {len(text) - limit} chars>"


def build_export_input(args: argparse.Namespace) -> tuple[VAEImageEncoderWrapper, torch.Tensor, dict[str, object]]:
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
        input_image = prepared.image_tensor.detach().to(device=model.device, dtype=model.torch_dtype)
        wrapper = VAEImageEncoderWrapper(model=model).to(device=model.device, dtype=model.torch_dtype).eval()
        eager_out = wrapper(input_image)
    meta = {
        "input_image_shape": list(input_image.shape),
        "input_image_dtype": str(input_image.dtype),
        "eager_output_shape": list(eager_out.shape),
        "eager_output_dtype": str(eager_out.dtype),
        "eager_output_finite": bool(torch.isfinite(eager_out).all().item()),
    }
    return wrapper, input_image, meta


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.status_output.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "preset": args.preset,
        "runtime_mode": "trt_export_vae_image_encoder",
        "output": str(args.output),
        "opset": args.opset,
        "mixed_precision": args.mixed_precision,
        "safe_attention_patch": not bool(args.no_safe_attention_patch),
        "safe_primitive_patch": not bool(args.no_safe_primitive_patch),
        "direct_vae_model_encode": True,
    }
    try:
        if not args.no_safe_primitive_patch:
            install_export_safe_vae_primitives()
        if not args.no_safe_attention_patch:
            install_export_safe_vae_attention()
        wrapper, input_image, meta = build_export_input(args)
        status["meta"] = meta
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (input_image,),
                str(args.output),
                input_names=["input_image"],
                output_names=["first_frame_latents"],
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
