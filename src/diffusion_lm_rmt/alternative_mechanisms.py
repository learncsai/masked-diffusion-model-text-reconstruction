"""Post-hoc tests of cross-size mechanism alternatives on existing runs.

The functions here operate on disagreement ratios normalized at a calibration
size.  They intentionally keep the additive-floor assumptions explicit: an
SGD-constrained floor is a sensitivity analysis unless a factorial design
identifies the mapping from the SGD contrast to the floor in the data contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from .diffusion.model import MaskedDiffusionTransformer


@dataclass(frozen=True)
class RatioModelFit:
    name: str
    parameter: float | None
    predictions: np.ndarray
    rmse: float
    mae: float


def root_mean_squared_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted must have matching shapes")
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def normalized_floor_curve(size_ratios: np.ndarray, rho: float) -> np.ndarray:
    """Return ``(rho * n_ref/n + 1)/(rho + 1)``.

    ``rho`` is the signal-to-floor ratio ``A/(n_ref B)``.  The curve is one at
    the reference size, tends to a constant floor for ``rho -> 0``, and tends
    to the pure inverse-size prediction for ``rho -> infinity``.
    """
    if rho < 0:
        raise ValueError("rho must be nonnegative")
    ratios = np.asarray(size_ratios, dtype=np.float64)
    return (rho * ratios + 1.0) / (rho + 1.0)


def fit_normalized_floor(size_ratios: np.ndarray, observed: np.ndarray) -> RatioModelFit:
    size_ratios = np.asarray(size_ratios, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if size_ratios.shape != observed.shape:
        raise ValueError("size_ratios and observed must have matching shapes")
    optimum = minimize_scalar(
        lambda value: np.mean((observed - normalized_floor_curve(size_ratios, value)) ** 2),
        bounds=(0.0, 1.0e4),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    predicted = normalized_floor_curve(size_ratios, float(optimum.x))
    return RatioModelFit(
        name="fitted_additive_floor",
        parameter=float(optimum.x),
        predictions=predicted,
        rmse=root_mean_squared_error(observed, predicted),
        mae=float(np.mean(np.abs(observed - predicted))),
    )


def fit_normalized_power(size_ratios: np.ndarray, observed: np.ndarray) -> RatioModelFit:
    size_ratios = np.asarray(size_ratios, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if size_ratios.shape != observed.shape:
        raise ValueError("size_ratios and observed must have matching shapes")
    optimum = minimize_scalar(
        lambda alpha: np.mean((observed - size_ratios**alpha) ** 2),
        bounds=(-2.0, 5.0),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    predicted = size_ratios ** float(optimum.x)
    return RatioModelFit(
        name="fitted_power_law",
        parameter=float(optimum.x),
        predictions=predicted,
        rmse=root_mean_squared_error(observed, predicted),
        mae=float(np.mean(np.abs(observed - predicted))),
    )


def fixed_prediction_fit(name: str, observed: np.ndarray, predicted: np.ndarray) -> RatioModelFit:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return RatioModelFit(
        name=name,
        parameter=None,
        predictions=predicted,
        rmse=root_mean_squared_error(observed, predicted),
        mae=float(np.mean(np.abs(observed - predicted))),
    )


def downstream_logits_from_hidden(
    model: MaskedDiffusionTransformer,
    hidden: torch.Tensor,
    *,
    layer: int,
    time_embedding: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact model suffix after a saved block output.

    For layer 0 this includes every later attention block and therefore keeps
    cross-token interactions.  For the last layer it is the final normalization
    and tied output projection.  This enables finite-difference sensitivity
    tests along observed cross-training hidden perturbations without
    materializing a full Jacobian.
    """
    if layer < 0 or layer >= len(model.blocks):
        raise ValueError("layer is outside the model block range")
    output = hidden
    for next_layer in range(layer + 1, len(model.blocks)):
        output = model.blocks[next_layer](output, time_embedding, padding_mask)
    output = model.final_norm(output)
    return torch.nn.functional.linear(output, model.token_embedding.weight, model.output_bias)


def centered_logit_mse(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    selected: torch.Tensor,
) -> float:
    """Return softmax-invariant logit displacement at selected tokens.

    Per-token constants are removed because they leave the predicted
    distribution unchanged. The vocabulary dimension is included in the mean.
    """

    if logits_a.shape != logits_b.shape:
        raise ValueError("logit tensors must have the same shape")
    if selected.shape != logits_a.shape[:-1]:
        raise ValueError("selected mask must match the token dimensions")
    displacement = (logits_a.float() - logits_b.float())[selected]
    if displacement.numel() == 0:
        return float("nan")
    displacement = displacement - displacement.mean(dim=-1, keepdim=True)
    return float(displacement.square().mean().item())
