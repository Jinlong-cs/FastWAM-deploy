from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    return Path(value).expanduser()


UPSTREAM_FASTWAM_DIR = _env_path("FASTWAM_UPSTREAM_DIR", WORKSPACE_ROOT / "FastWAM")
DATA_DIR = _env_path("FASTWAM_DATA_DIR", REPO_ROOT / "data")
OUTPUTS_DIR = _env_path("FASTWAM_OUTPUTS_DIR", REPO_ROOT / "outputs")
ARTIFACTS_DIR = _env_path("FASTWAM_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
CHECKPOINTS_DIR = _env_path("FASTWAM_CHECKPOINTS_DIR", ARTIFACTS_DIR / "checkpoints")
FASTWAM_RELEASE_DIR = _env_path("FASTWAM_RELEASE_DIR", CHECKPOINTS_DIR / "fastwam_release")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

