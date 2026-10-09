"""Continuous-time absorbing-mask schedules."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearSchedule:
    """Token survival schedule ``alpha(t)=1-t`` on ``[0,1]``."""

    def alpha(self, t: torch.Tensor | float) -> torch.Tensor:
        value = torch.as_tensor(t)
        return (1.0 - value).clamp(0.0, 1.0)

    def derivative(self, t: torch.Tensor | float) -> torch.Tensor:
        value = torch.as_tensor(t)
        return -torch.ones_like(value)

