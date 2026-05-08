from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyaction_fastwam.paths import FASTWAM_RELEASE_DIR, UPSTREAM_FASTWAM_DIR
from tinyaction_fastwam.presets import FastWAMPreset, get_preset
from tinyaction_fastwam.text_cache import TextContext, load_text_context_from_cache


@dataclass(frozen=True)
class FastWAMRuntimePaths:
    upstream_dir: Path
    checkpoint_path: Path
    dataset_stats_path: Path


@dataclass(frozen=True)
class FastWAMPreparedObservation:
    image_tensor: torch.Tensor
    proprio: torch.Tensor


def add_upstream_paths(upstream_dir: Path = UPSTREAM_FASTWAM_DIR) -> None:
    upstream_dir = upstream_dir.expanduser().resolve()
    upstream_src = upstream_dir / "src"
    for path in (upstream_dir, upstream_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def resolve_runtime_paths(
    *,
    preset_name: str,
    upstream_dir: Path = UPSTREAM_FASTWAM_DIR,
    checkpoint_path: Path | None = None,
    dataset_stats_path: Path | None = None,
) -> FastWAMRuntimePaths:
    preset = get_preset(preset_name)
    checkpoint_path = checkpoint_path or (FASTWAM_RELEASE_DIR / preset.checkpoint_filename)
    dataset_stats_path = dataset_stats_path or (FASTWAM_RELEASE_DIR / preset.dataset_stats_filename)
    return FastWAMRuntimePaths(
        upstream_dir=upstream_dir.expanduser().resolve(),
        checkpoint_path=checkpoint_path.expanduser().resolve(),
        dataset_stats_path=dataset_stats_path.expanduser().resolve(),
    )


def load_robotwin_policy(
    *,
    preset_name: str = "robotwin_uncond_3cam_384",
    upstream_dir: Path = UPSTREAM_FASTWAM_DIR,
    checkpoint_path: Path | None = None,
    dataset_stats_path: Path | None = None,
    device: str = "cuda",
    mixed_precision: str = "bf16",
    num_inference_steps: int = 20,
    action_horizon: int | None = None,
    replan_steps: int = 24,
    seed: int | None = 1234,
    rand_device: str = "cpu",
    timing_enabled: bool = True,
    text_cache_dir: Path | None = None,
    text_cache_instruction: str | None = None,
    text_cache_encoder_id: str = "wan22ti2v5b",
) -> Any:
    return load_robotwin_policy_without_text_encoder(
        preset_name=preset_name,
        upstream_dir=upstream_dir,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        device=device,
        mixed_precision=mixed_precision,
        num_inference_steps=num_inference_steps,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        seed=seed,
        rand_device=rand_device,
        timing_enabled=timing_enabled,
        text_cache_dir=text_cache_dir,
        text_cache_instruction=text_cache_instruction,
        text_cache_encoder_id=text_cache_encoder_id,
    )


def _mixed_precision_to_torch_dtype(mixed_precision: str) -> torch.dtype:
    key = mixed_precision.strip().lower()
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision: {mixed_precision}")


def load_robotwin_policy_without_text_encoder(
    *,
    preset_name: str = "robotwin_uncond_3cam_384",
    upstream_dir: Path = UPSTREAM_FASTWAM_DIR,
    checkpoint_path: Path | None = None,
    dataset_stats_path: Path | None = None,
    device: str = "cuda",
    mixed_precision: str = "bf16",
    num_inference_steps: int = 20,
    action_horizon: int | None = None,
    replan_steps: int = 24,
    seed: int | None = 1234,
    rand_device: str = "cpu",
    timing_enabled: bool = True,
    text_cache_dir: Path | None = None,
    text_cache_instruction: str | None = None,
    text_cache_encoder_id: str = "wan22ti2v5b",
) -> "FastWAMRobotwinOfflinePolicy":
    preset = get_preset(preset_name)
    paths = resolve_runtime_paths(
        preset_name=preset_name,
        upstream_dir=upstream_dir,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
    )
    if not paths.checkpoint_path.exists():
        raise FileNotFoundError(f"FastWAM checkpoint not found: {paths.checkpoint_path}")
    if not paths.dataset_stats_path.exists():
        raise FileNotFoundError(f"FastWAM dataset stats not found: {paths.dataset_stats_path}")

    add_upstream_paths(paths.upstream_dir)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(paths.upstream_dir / "configs")):
        cfg = compose(config_name="sim_robotwin.yaml", overrides=[f"task={preset.task}"])

    # The upstream RoboTwin policy wrapper force-loads the T5 text encoder. For AGX
    # baseline we use the model's context/context_mask inference path and avoid that
    # 11GB encoder until we add a real text-embedding cache.
    cfg.model.load_text_encoder = False
    cfg.model.skip_dit_load_from_pretrain = True
    cfg.model.action_dit_pretrained_path = None
    cfg.model.redirect_common_files = False

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model = instantiate(
        model_cfg,
        model_dtype=_mixed_precision_to_torch_dtype(mixed_precision),
        device=device,
    )
    model.load_checkpoint(str(paths.checkpoint_path))
    model = model.to(device).eval()

    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(paths.dataset_stats_path)))

    return FastWAMRobotwinOfflinePolicy(
        preset=preset,
        model=model,
        processor=processor,
        action_horizon=action_horizon or (preset.num_frames - 1),
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        seed=seed,
        rand_device=rand_device,
        timing_enabled=timing_enabled,
        text_cache_dir=text_cache_dir,
        text_cache_instruction=text_cache_instruction,
        text_cache_encoder_id=text_cache_encoder_id,
    )


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _empty_timing_rollout() -> dict[str, float]:
    return {
        "infer_s": 0.0,
        "image_preprocess_s": 0.0,
        "state_normalize_s": 0.0,
        "context_s": 0.0,
        "model_forward_s": 0.0,
        "action_postprocess_s": 0.0,
    }


class FastWAMRobotwinOfflinePolicy:
    def __init__(
        self,
        *,
        preset: FastWAMPreset,
        model: Any,
        processor: Any,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        seed: int | None,
        rand_device: str,
        timing_enabled: bool,
        text_cache_dir: Path | None = None,
        text_cache_instruction: str | None = None,
        text_cache_encoder_id: str = "wan22ti2v5b",
    ) -> None:
        self.preset = preset
        self.model = model
        self.processor = processor
        self.action_horizon = int(action_horizon)
        self.replan_steps = int(replan_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.seed = seed
        self.rand_device = rand_device
        self.timing_enabled = timing_enabled
        self.text_cache_dir = text_cache_dir.expanduser().resolve() if text_cache_dir is not None else None
        self.text_cache_instruction = text_cache_instruction
        self.text_cache_encoder_id = text_cache_encoder_id
        self._timing_rollout = _empty_timing_rollout()
        self._context: torch.Tensor | None = None
        self._context_mask: torch.Tensor | None = None
        self._text_context: TextContext | None = None

    def _record_stage(self, key: str, start_s: float) -> None:
        if self.timing_enabled:
            self._timing_rollout[key] += time.perf_counter() - start_s

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected one merged state key.")
        state_key = state_meta[0]["key"]
        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected one merged action key.")
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()

    def _build_robotwin_image_tensor(self, observation: dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        return tensor * (2.0 / 255.0) - 1.0

    def _get_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._context is None or self._context_mask is None:
            context_len = 128
            if self.text_cache_dir is not None:
                instruction = self.text_cache_instruction
                if instruction is None:
                    raise ValueError("`text_cache_instruction` is required when `text_cache_dir` is set.")
                self._text_context = load_text_context_from_cache(
                    cache_dir=self.text_cache_dir,
                    instruction=instruction,
                    context_len=context_len,
                    encoder_id=self.text_cache_encoder_id,
                    device=self.model.device,
                    dtype=self.model.torch_dtype,
                )
                self._context = self._text_context.context
                self._context_mask = self._text_context.context_mask
                return self._context, self._context_mask

            # Synthetic text context for the first AGX runtime baseline. This preserves
            # model tensor shapes without loading the heavy T5 encoder on Jetson.
            text_dim = int(self.model.text_dim)
            self._context = torch.zeros(
                (1, context_len, text_dim),
                dtype=self.model.torch_dtype,
                device=self.model.device,
            )
            self._context_mask = torch.ones((1, context_len), dtype=torch.bool, device=self.model.device)
        return self._context, self._context_mask

    def get_text_context_metadata(self) -> dict[str, str | None]:
        if self._text_context is not None:
            return {
                "text_context": "precomputed_t5_cache",
                "text_cache_path": str(self._text_context.cache_path),
                "semantic_text_encoder": "precomputed_cache",
                "prompt": self._text_context.prompt,
            }
        if self.text_cache_dir is not None:
            return {
                "text_context": "precomputed_t5_cache",
                "text_cache_path": None,
                "semantic_text_encoder": "precomputed_cache",
                "prompt": None,
            }
        return {
            "text_context": "synthetic_zero_context",
            "text_cache_path": None,
            "semantic_text_encoder": "disabled",
            "prompt": None,
        }

    def preprocess_observation(self, observation: dict[str, Any]) -> FastWAMPreparedObservation:
        stage_t0 = time.perf_counter() if self.timing_enabled else 0.0
        image_tensor = self._build_robotwin_image_tensor(observation)
        self._record_stage("image_preprocess_s", stage_t0)

        stage_t0 = time.perf_counter() if self.timing_enabled else 0.0
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)
        self._record_stage("state_normalize_s", stage_t0)
        return FastWAMPreparedObservation(image_tensor=image_tensor, proprio=proprio)

    @torch.no_grad()
    def infer_preprocessed_action(
        self,
        prepared: FastWAMPreparedObservation,
        instruction: str | None = None,
    ) -> np.ndarray:
        del instruction
        stage_t0 = time.perf_counter() if self.timing_enabled else 0.0
        context, context_mask = self._get_context()
        self._record_stage("context_s", stage_t0)

        stage_t0 = time.perf_counter() if self.timing_enabled else 0.0
        pred = self.model.infer_action(
            prompt=None,
            input_image=prepared.image_tensor,
            action_horizon=self.action_horizon,
            proprio=prepared.proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            rand_device=self.rand_device,
            tiled=False,
        )
        self._record_stage("model_forward_s", stage_t0)

        stage_t0 = time.perf_counter() if self.timing_enabled else 0.0
        action = self._denormalize_action(pred["action"])[0]
        self._record_stage("action_postprocess_s", stage_t0)
        return action

    @torch.no_grad()
    def _infer_action_chunk(self, observation: dict[str, Any], instruction: str | None = None) -> np.ndarray:
        total_t0 = time.perf_counter() if self.timing_enabled else 0.0
        prepared = self.preprocess_observation(observation)
        action = self.infer_preprocessed_action(prepared=prepared, instruction=instruction)
        self._record_stage("infer_s", total_t0)
        return action

    def get_timing_rollout(self) -> dict[str, float]:
        return dict(self._timing_rollout)

    def reset_timing_rollout(self) -> None:
        self._timing_rollout = _empty_timing_rollout()


def make_synthetic_robotwin_observation(
    *,
    preset: FastWAMPreset,
    seed: int = 1234,
    state_scale: float = 0.05,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    def image() -> np.ndarray:
        return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    state = rng.normal(loc=0.0, scale=state_scale, size=(preset.state_dim,)).astype(np.float32)
    return {
        "observation": {
            "head_camera": {"rgb": image()},
            "left_camera": {"rgb": image()},
            "right_camera": {"rgb": image()},
        },
        "joint_action": {
            "vector": state,
        },
    }


def predict_action_chunk(
    *,
    policy: Any,
    observation: dict[str, Any],
    instruction: str,
) -> np.ndarray:
    return policy._infer_action_chunk(observation=observation, instruction=instruction)
