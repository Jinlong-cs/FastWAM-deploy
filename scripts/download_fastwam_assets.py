#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from tinyaction_fastwam.paths import FASTWAM_RELEASE_DIR, ensure_dir
from tinyaction_fastwam.presets import DEFAULT_PRESET, PRESETS, get_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official released FastWAM checkpoint assets.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--output-dir", type=Path, default=FASTWAM_RELEASE_DIR)
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    output_dir = ensure_dir(args.output_dir.expanduser())
    allow_patterns = [
        preset.checkpoint_filename,
        preset.dataset_stats_filename,
    ]
    snapshot_path = snapshot_download(
        repo_id=preset.checkpoint_repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=output_dir,
        allow_patterns=allow_patterns,
    )
    result = {
        "preset": preset.name,
        "repo_id": preset.checkpoint_repo_id,
        "revision": args.revision,
        "local_dir": str(output_dir),
        "snapshot_path": str(snapshot_path),
        "checkpoint": str(output_dir / preset.checkpoint_filename),
        "dataset_stats": str(output_dir / preset.dataset_stats_filename),
        "allow_patterns": allow_patterns,
    }
    (output_dir / "download_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

