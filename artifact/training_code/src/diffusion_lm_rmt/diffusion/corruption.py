"""Deterministic CPU absorbing-state forward corruption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .schedule import LinearSchedule


@dataclass(frozen=True)
class CorruptionBatch:
    """A clean batch, its corruption, and exact reconstruction targets."""

    clean_ids: torch.Tensor
    corrupted_ids: torch.Tensor
    target_mask: torch.Tensor
    eligible_mask: torch.Tensor
    times: torch.Tensor


def eligible_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_ids: Iterable[int],
) -> torch.Tensor:
    """Identify non-padding, non-special tokens eligible for diffusion."""
    eligible = attention_mask.to(dtype=torch.bool).clone()
    for token_id in special_token_ids:
        eligible &= input_ids.ne(int(token_id))
    return eligible


def corrupt_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    times: torch.Tensor,
    *,
    mask_token_id: int,
    special_token_ids: Iterable[int],
    generator: torch.Generator,
    schedule: LinearSchedule | None = None,
    ensure_masked: bool = True,
) -> CorruptionBatch:
    """Apply independent absorbing masks using only an explicit CPU generator.

    Inputs must be CPU tensors so the random-number stream is independent of
    accelerator implementation. At least one eligible token per nonempty row
    is masked when ``ensure_masked`` is true.
    """
    if input_ids.device.type != "cpu" or times.device.type != "cpu":
        raise ValueError("corruption inputs and times must be CPU tensors")
    if times.ndim != 1 or times.shape[0] != input_ids.shape[0]:
        raise ValueError("times must have shape (batch,)")
    active_schedule = schedule or LinearSchedule()
    eligible = eligible_token_mask(input_ids, attention_mask, special_token_ids)
    mask_probability = 1.0 - active_schedule.alpha(times.float())
    uniforms = torch.rand(input_ids.shape, generator=generator, device="cpu")
    targets = eligible & (uniforms < mask_probability[:, None])
    if ensure_masked:
        for row in range(input_ids.shape[0]):
            if eligible[row].any() and not targets[row].any():
                indices = torch.nonzero(eligible[row], as_tuple=False).flatten()
                chosen = torch.randint(indices.numel(), (1,), generator=generator).item()
                targets[row, indices[chosen]] = True
    corrupted = input_ids.clone()
    corrupted[targets] = int(mask_token_id)
    return CorruptionBatch(input_ids.clone(), corrupted, targets, eligible, times.float().clone())

