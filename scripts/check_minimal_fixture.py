#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "fixtures" / "fastwam_minimal_sample.json"
REQUIRED_CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
REQUIRED_RUNTIME_CONTRACT = {
    "composed_image_shape": [1, 3, 384, 320],
    "composed_image_dtype": "float16_or_float32",
    "composed_image_range": [-1.0, 1.0],
    "text_context_shape": [1, 128, 4096],
    "text_context_dtype": "float16_or_bfloat16",
    "action_horizon": 32,
    "action_dim": 14,
    "trt_stages": ["vae_image_encoder", "video_prefill", "action_step_dynamic_kv"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and validate the public FastWAM minimal fixture.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    return parser.parse_args()


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object.")
    return value


def require_keys(payload: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise KeyError(f"{context} is missing keys: {missing}")


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def validate_shape(value: Any, context: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{context}.shape must be a list.")
    shape = [int(item) for item in value]
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"{context}.shape must contain positive dimensions.")
    return shape


def validate_numeric_list(value: Any, context: str) -> list[int | float]:
    if not isinstance(value, list):
        raise TypeError(f"{context}.data must be a list.")
    data: list[int | float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{context}.data contains a non-numeric value: {item!r}")
        if not math.isfinite(float(item)):
            raise ValueError(f"{context}.data contains a non-finite value: {item!r}")
        data.append(item)
    return data


def validate_tensor(
    value: Any,
    context: str,
    *,
    dtype: str,
    rank: int,
    shape: list[int] | None = None,
    last_dim: int | None = None,
) -> dict[str, Any]:
    tensor = require_mapping(value, context)
    require_keys(tensor, ["shape", "dtype", "data"], context)

    actual_shape = validate_shape(tensor["shape"], context)
    if len(actual_shape) != rank:
        raise ValueError(f"{context}.shape must have rank {rank}, got {actual_shape}.")
    if shape is not None and actual_shape != shape:
        raise ValueError(f"{context}.shape must be {shape}, got {actual_shape}.")
    if last_dim is not None and actual_shape[-1] != last_dim:
        raise ValueError(f"{context}.shape[-1] must be {last_dim}, got {actual_shape[-1]}.")
    if tensor["dtype"] != dtype:
        raise ValueError(f"{context}.dtype must be {dtype}, got {tensor['dtype']!r}.")

    data = validate_numeric_list(tensor["data"], context)
    numel = product(actual_shape)
    if len(data) != numel:
        raise ValueError(f"{context}.data has {len(data)} values, but shape {actual_shape} needs {numel}.")
    if dtype == "uint8":
        bad_values = [item for item in data if int(item) != item or item < 0 or item > 255]
        if bad_values:
            raise ValueError(f"{context}.data must contain uint8 values.")

    return {"shape": actual_shape, "dtype": dtype, "numel": len(data)}


def validate_runtime_contract(value: Any) -> dict[str, Any]:
    contract = require_mapping(value, "runtime_contract")
    require_keys(contract, sorted(REQUIRED_RUNTIME_CONTRACT), "runtime_contract")
    for key, required in REQUIRED_RUNTIME_CONTRACT.items():
        if contract[key] != required:
            raise ValueError(f"runtime_contract.{key} must be {required!r}, got {contract[key]!r}.")
    return contract


def validate_cameras(value: Any) -> dict[str, Any]:
    cameras = require_mapping(value, "sample.observation.cameras")
    if sorted(cameras) != sorted(REQUIRED_CAMERA_NAMES):
        raise ValueError(f"sample.observation.cameras must contain exactly {REQUIRED_CAMERA_NAMES}.")

    summary: dict[str, Any] = {}
    for camera_name in REQUIRED_CAMERA_NAMES:
        camera = require_mapping(cameras[camera_name], f"sample.observation.cameras.{camera_name}")
        require_keys(camera, ["shape", "layout", "dtype", "data"], f"sample.observation.cameras.{camera_name}")
        if camera["layout"] != "CHW_RGB":
            raise ValueError(f"sample.observation.cameras.{camera_name}.layout must be CHW_RGB.")
        summary[camera_name] = validate_tensor(
            camera,
            f"sample.observation.cameras.{camera_name}",
            dtype="uint8",
            rank=3,
        )
        if summary[camera_name]["shape"][0] != 3:
            raise ValueError(f"sample.observation.cameras.{camera_name}.shape[0] must be 3 RGB channels.")
    return summary


def validate_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    require_keys(
        payload,
        ["schema_version", "license", "source", "preset", "instruction", "runtime_contract", "sample"],
        "fixture",
    )
    if payload["schema_version"] != 1:
        raise ValueError(f"schema_version must be 1, got {payload['schema_version']!r}.")
    if payload["license"] != "CC0-1.0":
        raise ValueError(f"license must be CC0-1.0, got {payload['license']!r}.")
    if payload["source"] != "synthetic_public_fixture":
        raise ValueError("source must be synthetic_public_fixture.")
    if payload["preset"] != "robotwin_uncond_3cam_384":
        raise ValueError(f"preset must be robotwin_uncond_3cam_384, got {payload['preset']!r}.")
    if not isinstance(payload["instruction"], str) or not payload["instruction"].strip():
        raise ValueError("instruction must be a non-empty string.")

    contract = validate_runtime_contract(payload["runtime_contract"])
    sample = require_mapping(payload["sample"], "sample")
    require_keys(sample, ["sample_id", "observation", "target"], "sample")

    observation = require_mapping(sample["observation"], "sample.observation")
    require_keys(observation, ["state", "cameras"], "sample.observation")
    state = validate_tensor(observation["state"], "sample.observation.state", dtype="float32", rank=1, shape=[14])
    cameras = validate_cameras(observation["cameras"])

    target = require_mapping(sample["target"], "sample.target")
    require_keys(target, ["action_contract", "action_preview"], "sample.target")
    action_contract = require_mapping(target["action_contract"], "sample.target.action_contract")
    require_keys(action_contract, ["shape", "dtype", "horizon", "action_dim"], "sample.target.action_contract")
    if action_contract["shape"] != [contract["action_horizon"], contract["action_dim"]]:
        raise ValueError("sample.target.action_contract.shape does not match runtime_contract.")
    if action_contract["dtype"] != "float32":
        raise ValueError("sample.target.action_contract.dtype must be float32.")
    if action_contract["horizon"] != contract["action_horizon"]:
        raise ValueError("sample.target.action_contract.horizon does not match runtime_contract.")
    if action_contract["action_dim"] != contract["action_dim"]:
        raise ValueError("sample.target.action_contract.action_dim does not match runtime_contract.")
    action_preview = validate_tensor(
        target["action_preview"],
        "sample.target.action_preview",
        dtype="float32",
        rank=2,
        last_dim=contract["action_dim"],
    )
    if action_preview["shape"][0] > contract["action_horizon"]:
        raise ValueError("sample.target.action_preview has more timesteps than the runtime action horizon.")

    return {
        "fixture": str(DEFAULT_FIXTURE),
        "preset": payload["preset"],
        "instruction": payload["instruction"],
        "sample_id": sample["sample_id"],
        "state": state,
        "cameras": cameras,
        "action_preview": action_preview,
        "runtime_contract": contract,
    }


def main() -> None:
    args = parse_args()
    payload = require_mapping(json.loads(args.fixture.read_text()), "fixture")
    summary = validate_fixture(payload)
    summary["fixture"] = str(args.fixture)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
