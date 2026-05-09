from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FastWAMPreset:
    name: str
    task: str
    checkpoint_repo_id: str
    checkpoint_filename: str
    dataset_stats_filename: str
    num_cameras: int
    state_dim: int
    action_dim: int
    num_frames: int
    action_video_freq_ratio: int
    image_height: int
    image_width: int
    description: str


PRESETS = {
    "robotwin_uncond_3cam_384": FastWAMPreset(
        name="robotwin_uncond_3cam_384",
        task="robotwin_uncond_3cam_384_1e-4",
        checkpoint_repo_id="yuanty/fastwam",
        checkpoint_filename="robotwin_uncond_3cam_384.pt",
        dataset_stats_filename="robotwin_uncond_3cam_384_dataset_stats.json",
        num_cameras=3,
        state_dim=14,
        action_dim=14,
        num_frames=33,
        action_video_freq_ratio=4,
        image_height=384,
        image_width=320,
        description="Official released FastWAM RoboTwin 2.0 unconditioned 3-camera checkpoint.",
    )
}

DEFAULT_PRESET = "robotwin_uncond_3cam_384"


def get_preset(name: str) -> FastWAMPreset:
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown FastWAM preset '{name}'. Available presets: {available}")
    return PRESETS[name]
