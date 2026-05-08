#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tinyaction_fastwam.paths import DATA_DIR


CAMERAS = {
    "head_rgb": "observation.images.cam_high",
    "left_rgb": "observation.images.cam_left_wrist",
    "right_rgb": "observation.images.cam_right_wrist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one real RoboTwin unified LeRobot frame as a FastWAM offline .npz sample."
    )
    parser.add_argument("--repo-id", default="lerobot/robotwin_unified")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir", type=Path, default=DATA_DIR / "robotwin_unified_minimal")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "real_samples")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def hf_download(*, repo_id: str, filename: str, local_dir: Path, revision: str | None, force: bool) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
            force_download=force,
        )
    )


def ensure_minimal_files(args: argparse.Namespace) -> dict[str, Path]:
    args.local_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "info": "meta/info.json",
        "tasks": "meta/tasks.parquet",
        "episodes": "meta/episodes/chunk-000/file-000.parquet",
        "data": "data/chunk-000/file-000.parquet",
    }
    paths = {
        key: hf_download(
            repo_id=args.repo_id,
            filename=filename,
            local_dir=args.local_dir,
            revision=args.revision,
            force=args.force_download,
        )
        for key, filename in files.items()
    }
    for alias, camera_key in CAMERAS.items():
        paths[alias] = hf_download(
            repo_id=args.repo_id,
            filename=f"videos/{camera_key}/chunk-000/file-000.mp4",
            local_dir=args.local_dir,
            revision=args.revision,
            force=args.force_download,
        )
    return paths


def task_from_index(tasks_df: Any, task_index: int) -> str:
    if "task" in tasks_df.columns:
        rows = tasks_df[tasks_df["task_index"] == task_index]
        if rows.empty:
            raise KeyError(f"task_index={task_index} not found in tasks.parquet")
        return str(rows.iloc[0]["task"])
    if tasks_df.index.name == "task":
        rows = tasks_df[tasks_df["task_index"] == task_index]
        if rows.empty:
            raise KeyError(f"task_index={task_index} not found in tasks.parquet")
        return str(rows.index[0])
    raise KeyError("Cannot resolve task text: tasks.parquet must contain a task column or task index.")


def select_row(data_df: Any, *, episode_index: int, frame_offset: int) -> Any:
    rows = data_df[data_df["episode_index"] == episode_index]
    if rows.empty:
        raise KeyError(f"episode_index={episode_index} not found in data parquet")
    rows = rows.sort_values("frame_index")
    if frame_offset < 0 or frame_offset >= len(rows):
        raise IndexError(
            f"frame_offset={frame_offset} out of range for episode_index={episode_index}, length={len(rows)}"
        )
    return rows.iloc[frame_offset]


def decode_frame(video_path: Path, frame_index: int) -> np.ndarray:
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for idx, frame in enumerate(container.decode(stream)):
            if idx == frame_index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"Could not decode frame_index={frame_index} from {video_path}")


def to_float32_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {array.shape}")
    return np.ascontiguousarray(array)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    paths = ensure_minimal_files(args)

    import pandas as pd

    episodes_df = pd.read_parquet(paths["episodes"])
    data_df = pd.read_parquet(paths["data"])
    tasks_df = pd.read_parquet(paths["tasks"])
    row = select_row(data_df, episode_index=args.episode_index, frame_offset=args.frame_offset)

    task_index = int(row["task_index"])
    instruction = task_from_index(tasks_df, task_index)
    global_frame_index = int(row["index"])
    video_frame_index = int(global_frame_index)
    state = to_float32_vector(row["observation.state"], name="observation.state")
    action = to_float32_vector(row["action"], name="action")
    images = {
        alias: decode_frame(paths[alias], video_frame_index)
        for alias in ("head_rgb", "left_rgb", "right_rgb")
    }

    output_name = f"robotwin_unified_ep{args.episode_index:06d}_frame{int(row['frame_index']):06d}.npz"
    output_path = args.output_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episode_rows = episodes_df[episodes_df["episode_index"] == args.episode_index].head(1).to_dict(orient="records")
    metadata = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "episode_index": args.episode_index,
        "frame_offset": args.frame_offset,
        "frame_index": int(row["frame_index"]),
        "global_index": global_frame_index,
        "video_frame_index": video_frame_index,
        "timestamp": float(row["timestamp"]),
        "task_index": task_index,
        "instruction": instruction,
        "source_files": {key: str(value) for key, value in paths.items()},
        "episodes_row": jsonable(episode_rows),
    }
    np.savez_compressed(
        output_path,
        head_rgb=images["head_rgb"],
        left_rgb=images["left_rgb"],
        right_rgb=images["right_rgb"],
        state=state,
        action=action,
        instruction=np.asarray(instruction),
        metadata_json=np.asarray(json.dumps(jsonable(metadata), ensure_ascii=False)),
    )
    summary = {
        "status": "success",
        "output": str(output_path.resolve()),
        "instruction": instruction,
        "episode_index": args.episode_index,
        "frame_index": int(row["frame_index"]),
        "video_frame_index": video_frame_index,
        "task_index": task_index,
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "image_shapes": {key: list(value.shape) for key, value in images.items()},
        "sample_size_bytes": output_path.stat().st_size,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
