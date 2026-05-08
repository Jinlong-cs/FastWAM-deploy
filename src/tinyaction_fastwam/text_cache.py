from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_ENCODER_ID = "wan22ti2v5b"


@dataclass(frozen=True)
class TextContext:
    context: torch.Tensor
    context_mask: torch.Tensor
    prompt: str
    cache_path: Path


def format_robotwin_prompt(instruction: str) -> str:
    instruction = str(instruction).strip()
    if not instruction:
        raise ValueError("Instruction cannot be empty when resolving a text cache entry.")
    if instruction.startswith("A video recorded from a robot's point of view"):
        return instruction
    return DEFAULT_PROMPT.format(task=instruction)


def resolve_text_cache_path(
    *,
    cache_dir: Path,
    instruction: str,
    context_len: int = DEFAULT_CONTEXT_LEN,
    encoder_id: str = DEFAULT_ENCODER_ID,
) -> tuple[Path, str]:
    prompt = format_robotwin_prompt(instruction)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    filename = f"{digest}.t5_len{int(context_len)}.{encoder_id}.pt"
    return cache_dir.expanduser().resolve() / filename, prompt


def load_text_context_from_cache(
    *,
    cache_dir: Path,
    instruction: str,
    context_len: int = DEFAULT_CONTEXT_LEN,
    encoder_id: str = DEFAULT_ENCODER_ID,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> TextContext:
    cache_path, prompt = resolve_text_cache_path(
        cache_dir=cache_dir,
        instruction=instruction,
        context_len=context_len,
        encoder_id=encoder_id,
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing FastWAM text embedding cache: {cache_path}. "
            "Generate it with upstream scripts/precompute_text_embeds.py on an x86 GPU/Vast box, "
            "then copy the cache directory to AGX."
        )

    payload = torch.load(str(cache_path), map_location="cpu")
    if "context" not in payload or "mask" not in payload:
        raise KeyError(f"Text cache must contain `context` and `mask`: {cache_path}")

    context = payload["context"]
    context_mask = payload["mask"]
    if context.ndim != 2:
        raise ValueError(f"Cached context must be [L,D], got {tuple(context.shape)} in {cache_path}")
    if context_mask.ndim != 1:
        raise ValueError(f"Cached mask must be [L], got {tuple(context_mask.shape)} in {cache_path}")
    if context.shape[0] != int(context_len) or context_mask.shape[0] != int(context_len):
        raise ValueError(
            f"Cached text length mismatch in {cache_path}: "
            f"context={tuple(context.shape)}, mask={tuple(context_mask.shape)}, expected L={context_len}"
        )

    context = context.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
    context_mask = context_mask.unsqueeze(0).to(device=device, dtype=torch.bool, non_blocking=True)
    context = context.clone()
    context[~context_mask] = 0.0
    # Upstream FastWAM zeroes padded text embeddings, then exposes an all-true
    # mask to cross-attention. Match that cache semantics in deploy runtime.
    context_mask = torch.ones_like(context_mask, dtype=torch.bool, device=context_mask.device)
    return TextContext(context=context, context_mask=context_mask, prompt=prompt, cache_path=cache_path)
