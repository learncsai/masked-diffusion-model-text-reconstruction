"""Masked diffusion training objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .schedule import LinearSchedule


def mdlm_weight(times: torch.Tensor, *, cap: float | None = None, schedule: LinearSchedule | None = None) -> torch.Tensor:
    """Return ``-alpha'(t)/(1-alpha(t))``; for linear schedules this is ``1/t``."""
    active_schedule = schedule or LinearSchedule()
    denominator = (1.0 - active_schedule.alpha(times)).clamp_min(torch.finfo(times.dtype).eps)
    weights = -active_schedule.derivative(times) / denominator
    return weights.clamp_max(cap) if cap is not None else weights


def diffusion_loss(
    logits: torch.Tensor,
    clean_ids: torch.Tensor,
    target_mask: torch.Tensor,
    eligible_mask: torch.Tensor,
    times: torch.Tensor,
    *,
    mode: str,
    weight_cap: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute simple masked CE or continuous-time MDLM-weighted CE."""
    per_token = F.cross_entropy(logits.float().transpose(1, 2), clean_ids, reduction="none")
    masked = target_mask.to(per_token.dtype)
    if mode == "simple_masked_ce":
        loss = (per_token * masked).sum() / masked.sum().clamp_min(1.0)
    elif mode == "mdlm_weighted":
        per_sequence = (per_token * masked).sum(dim=1) / eligible_mask.sum(dim=1).clamp_min(1)
        loss = (per_sequence * mdlm_weight(times.float(), cap=weight_cap)).mean()
    else:
        raise ValueError(f"unknown loss mode: {mode}")
    masked_count = int(target_mask.sum().item())
    accuracy = ((logits.argmax(dim=-1) == clean_ids) & target_mask).sum().item() / max(masked_count, 1)
    return loss, {"masked_tokens": float(masked_count), "masked_accuracy": float(accuracy), "masked_ce": float((per_token * masked).sum().detach().item() / max(masked_count, 1))}

