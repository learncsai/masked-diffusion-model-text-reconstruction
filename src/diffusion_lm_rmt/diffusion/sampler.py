"""Absorbing-mask reverse diffusion sampling."""

from __future__ import annotations

import torch

from .schedule import LinearSchedule


def reverse_transition_probabilities(s: float, t: float, schedule: LinearSchedule | None = None) -> tuple[float, float]:
    """Return conditional probabilities of remaining masked and unmasking."""
    if not 0 <= s < t <= 1:
        raise ValueError("reverse times must satisfy 0 <= s < t <= 1")
    active = schedule or LinearSchedule()
    alpha_s = float(active.alpha(s))
    alpha_t = float(active.alpha(t))
    denominator = 1.0 - alpha_t
    remain = (1.0 - alpha_s) / denominator
    unmask = (alpha_s - alpha_t) / denominator
    return remain, unmask


@torch.no_grad()
def reverse_sample(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int,
    pad_token_id: int,
    steps: int,
    generator: torch.Generator,
    mode: str = "categorical",
) -> torch.Tensor:
    """Reverse an absorbing-mask chain without causal left-to-right decoding."""
    if generator.device.type != "cpu":
        raise ValueError("sampler requires a CPU generator")
    device = next(model.parameters()).device
    current = input_ids.detach().cpu().clone()
    attention_cpu = attention_mask.detach().cpu()
    times = torch.linspace(1.0, 0.0, steps + 1)
    for index in range(steps):
        t, s = float(times[index]), float(times[index + 1])
        masked = current.eq(mask_token_id) & attention_cpu.bool()
        if not masked.any():
            break
        logits = model(current.to(device), attention_cpu.to(device), torch.full((current.shape[0],), t, device=device)).detach().float().cpu()
        logits[..., mask_token_id] = -torch.inf
        logits[..., pad_token_id] = -torch.inf
        _, unmask_probability = reverse_transition_probabilities(s, t)
        resolve = masked if s == 0.0 else masked & (torch.rand(masked.shape, generator=generator) < unmask_probability)
        if mode == "greedy":
            selected = logits.argmax(dim=-1)
        elif mode == "categorical":
            probabilities = torch.softmax(logits, dim=-1)
            selected = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator).reshape(current.shape)
        else:
            raise ValueError(f"unknown token selection mode: {mode}")
        current[resolve] = selected[resolve]
    if (current.eq(mask_token_id) & attention_cpu.bool()).any():
        raise RuntimeError("reverse sampler failed to resolve every mask")
    return current
