from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tinyaction_fastwam.presets import FastWAMPreset


DEFAULT_INSTRUCTION = "Pick up the object and place it at the target."


def _as_rgb_uint8(array: Any, *, name: str) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"{name} must be [H,W,3] or [T,H,W,3], got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"{name} must have 3 RGB channels, got {image.shape}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _as_state_vector(array: Any, *, preset: FastWAMPreset) -> np.ndarray:
    state = np.asarray(array, dtype=np.float32)
    if state.ndim == 2:
        state = state[0]
    if state.ndim != 1 or state.shape[0] != preset.state_dim:
        raise ValueError(f"State vector must be [{preset.state_dim}], got {state.shape}")
    return np.ascontiguousarray(state)


def make_robotwin_observation(
    *,
    head_rgb: Any,
    left_rgb: Any,
    right_rgb: Any,
    state: Any,
    preset: FastWAMPreset,
) -> dict[str, Any]:
    return {
        "observation": {
            "head_camera": {"rgb": _as_rgb_uint8(head_rgb, name="head_rgb")},
            "left_camera": {"rgb": _as_rgb_uint8(left_rgb, name="left_rgb")},
            "right_camera": {"rgb": _as_rgb_uint8(right_rgb, name="right_rgb")},
        },
        "joint_action": {
            "vector": _as_state_vector(state, preset=preset),
        },
    }


def load_robotwin_npz_sample(path: Path, *, preset: FastWAMPreset) -> tuple[dict[str, Any], str]:
    with np.load(path.expanduser(), allow_pickle=False) as sample:
        keys = set(sample.files)
        camera_aliases = {
            "head_rgb": ("head_rgb", "head_camera", "cam_high", "observation.images.cam_high"),
            "left_rgb": ("left_rgb", "left_camera", "cam_left_wrist", "observation.images.cam_left_wrist"),
            "right_rgb": ("right_rgb", "right_camera", "cam_right_wrist", "observation.images.cam_right_wrist"),
            "state": ("state", "joint_state", "observation.state", "observation.state.default"),
        }

        def get_first(name: str) -> np.ndarray:
            for key in camera_aliases[name]:
                if key in keys:
                    return sample[key]
            raise KeyError(f"Missing `{name}` in {path}. Tried aliases: {camera_aliases[name]}")

        instruction = str(sample["instruction"].item()) if "instruction" in keys else DEFAULT_INSTRUCTION
        observation = make_robotwin_observation(
            head_rgb=get_first("head_rgb"),
            left_rgb=get_first("left_rgb"),
            right_rgb=get_first("right_rgb"),
            state=get_first("state"),
            preset=preset,
        )
        return observation, instruction


def load_robotwin_json_sample(path: Path, *, preset: FastWAMPreset) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.expanduser().read_text())
    instruction = str(payload.get("instruction", DEFAULT_INSTRUCTION))
    observation = make_robotwin_observation(
        head_rgb=payload["head_rgb"],
        left_rgb=payload["left_rgb"],
        right_rgb=payload["right_rgb"],
        state=payload["state"],
        preset=preset,
    )
    return observation, instruction


def load_offline_robotwin_sample(path: Path, *, preset: FastWAMPreset) -> tuple[dict[str, Any], str]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_robotwin_npz_sample(path, preset=preset)
    if suffix == ".json":
        return load_robotwin_json_sample(path, preset=preset)
    raise ValueError(f"Unsupported sample format: {path}. Expected .npz or .json.")
