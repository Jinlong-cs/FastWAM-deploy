from __future__ import annotations

import math

import torch


def install_export_safe_rope() -> None:
    """Patch FastWAM RoPE with an ONNX-friendly real-valued implementation.

    Upstream FastWAM uses ``view_as_complex`` / ``view_as_real`` in the RoPE
    helper. That is valid in eager PyTorch but unsupported by the legacy ONNX
    exporter used on Jetson. PyTorch 2.11's exporter also rejects complex-valued
    RoPE cache expansion before the patched helper is reached. The deployment
    exports use static shapes, so the cos/sin form is sufficient and keeps
    upstream FastWAM unmodified.
    """

    def safe_precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
        del theta
        return torch.zeros((int(end), int(dim) // 2), dtype=torch.float32)

    def safe_precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
        return (
            safe_precompute_freqs_cis(dim - 2 * (dim // 3), end, theta),
            safe_precompute_freqs_cis(dim // 3, end, theta),
            safe_precompute_freqs_cis(dim // 3, end, theta),
        )

    def real_rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
        del freqs
        batch, seq_len, hidden = x.shape
        head_dim = hidden // int(num_heads)
        x_heads = x.reshape(batch, seq_len, int(num_heads), head_dim)
        x_pair = x_heads.reshape(batch, seq_len, int(num_heads), head_dim // 2, 2)
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]
        half_dim = head_dim // 2
        inv_freq = torch.exp(
            -torch.arange(0, half_dim, dtype=torch.float32, device=x.device)
            * (math.log(10000.0) / max(half_dim, 1))
        )
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        angles = positions[:, None] * inv_freq[None, :]
        cos = torch.cos(angles).to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)
        sin = torch.sin(angles).to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return rotated.reshape(batch, seq_len, hidden)

    import fastwam.models.wan22.mot as mot_module
    import fastwam.models.wan22.wan_video_dit as wan_dit_module
    import fastwam.models.wan22.action_dit as action_dit_module

    wan_dit_module.precompute_freqs_cis = safe_precompute_freqs_cis
    wan_dit_module.precompute_freqs_cis_3d = safe_precompute_freqs_cis_3d
    action_dit_module.precompute_freqs_cis = safe_precompute_freqs_cis
    wan_dit_module.rope_apply = real_rope_apply
    mot_module.rope_apply = real_rope_apply
