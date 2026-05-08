#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyaction_fastwam.paths import ARTIFACTS_DIR, FASTWAM_RELEASE_DIR
from tinyaction_fastwam.presets import DEFAULT_PRESET, PRESETS, get_preset


EXPECTED_SIZES = {
    "robotwin_uncond_3cam_384.pt": 12041813092,
    "robotwin_uncond_3cam_384_dataset_stats.json": 88715,
    "Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth": 2818839170,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check FastWAM deployment asset presence and byte sizes.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--release-dir", type=Path, default=FASTWAM_RELEASE_DIR)
    parser.add_argument("--models-dir", type=Path, default=ARTIFACTS_DIR / "models")
    return parser.parse_args()


def file_status(path: Path, expected_size: int | None = None) -> dict[str, object]:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    return {
        "path": str(path),
        "exists": exists,
        "size": size,
        "expected_size": expected_size,
        "complete": bool(exists and (expected_size is None or size == expected_size)),
    }


def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    release_dir = args.release_dir.expanduser()
    models_dir = args.models_dir.expanduser()
    checks = {
        preset.checkpoint_filename: file_status(
            release_dir / preset.checkpoint_filename,
            EXPECTED_SIZES.get(preset.checkpoint_filename),
        ),
        preset.dataset_stats_filename: file_status(
            release_dir / preset.dataset_stats_filename,
            EXPECTED_SIZES.get(preset.dataset_stats_filename),
        ),
        "Wan2.2_VAE.pth": file_status(
            models_dir / "Wan-AI" / "Wan2.2-TI2V-5B" / "Wan2.2_VAE.pth",
            EXPECTED_SIZES.get("Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
        ),
    }
    result = {
        "preset": args.preset,
        "all_complete": all(item["complete"] for item in checks.values()),
        "checks": checks,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

