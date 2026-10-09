"""Diagnostics for stochastic-gradient noise sampled at frozen parameters."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis


def standardized_gradient_residuals(
    mean_samples: np.ndarray,
    noise_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract an independent mean-gradient estimate and standardize coordinates."""

    mean_samples = np.asarray(mean_samples, dtype=np.float64)
    noise_samples = np.asarray(noise_samples, dtype=np.float64)
    if mean_samples.ndim != 2 or noise_samples.ndim != 2:
        raise ValueError("gradient samples must be two-dimensional")
    if mean_samples.shape[1] != noise_samples.shape[1]:
        raise ValueError("mean and noise samples must share coordinates")
    estimated_mean = mean_samples.mean(axis=0)
    residuals = noise_samples - estimated_mean[None, :]
    # Tail-shape statistics are translation invariant. Recenter the finite
    # residual sample before pooling standardized coordinates; otherwise a
    # noisy mean estimate divided by a nearly zero coordinate variance can
    # create a purely numerical apparent tail.
    centered = residuals - residuals.mean(axis=0, keepdims=True)
    scales = centered.std(axis=0, ddof=1)
    usable = np.isfinite(scales) & (scales > 1.0e-12)
    return centered[:, usable] / scales[usable][None, :], usable


def hill_tail_index(values: np.ndarray, tail_fraction: float = 0.05) -> float:
    """Estimate a power-law tail index from the largest absolute observations."""

    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must be in (0, 1)")
    absolute = np.sort(np.abs(np.asarray(values, dtype=np.float64).ravel()))
    absolute = absolute[np.isfinite(absolute) & (absolute > 0)]
    k = max(2, int(np.floor(tail_fraction * absolute.size)))
    if absolute.size <= k:
        return float("nan")
    tail = absolute[-k:]
    threshold = absolute[-k - 1]
    denominator = np.log(tail / threshold).sum()
    return float(k / denominator) if denominator > 0 else float("inf")


def gradient_tail_summary(standardized_residuals: np.ndarray) -> dict[str, float | int]:
    """Summarize marginal and pooled tails without asserting a tail family."""

    values = np.asarray(standardized_residuals, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("standardized residuals must be a nonempty matrix")
    coordinate_kurtosis = kurtosis(values, axis=0, fisher=True, bias=False)
    absolute = np.abs(values.ravel())
    q95, q99, q999 = np.quantile(absolute, [0.95, 0.99, 0.999])
    return {
        "noise_batches": int(values.shape[0]),
        "usable_coordinates": int(values.shape[1]),
        "pooled_excess_kurtosis": float(kurtosis(values.ravel(), fisher=True, bias=False)),
        "median_coordinate_excess_kurtosis": float(np.nanmedian(coordinate_kurtosis)),
        "coordinate_excess_kurtosis_q25": float(np.nanquantile(coordinate_kurtosis, 0.25)),
        "coordinate_excess_kurtosis_q75": float(np.nanquantile(coordinate_kurtosis, 0.75)),
        "hill_index_top_5pct": hill_tail_index(values, 0.05),
        "abs_q95": float(q95),
        "abs_q99": float(q99),
        "abs_q999": float(q999),
        "abs_q99_over_q95": float(q99 / q95),
        "abs_q999_over_q99": float(q999 / q99),
    }
