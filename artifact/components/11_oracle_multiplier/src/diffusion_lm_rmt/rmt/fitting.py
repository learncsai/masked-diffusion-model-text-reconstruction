"""Scaling-law fits with replicate-level bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ScalingFit:
    model: str
    amplitude: float
    exponent: float
    offset: float
    exponent_ci_low: float
    exponent_ci_high: float
    r_squared: float
    n_min: int
    n_max: int
    points: int
    bootstrap_successes: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _with_offset(n: FloatArray, amplitude: float, exponent: float, offset: float) -> FloatArray:
    return amplitude * n ** (-exponent) + offset


def _no_offset(n: FloatArray, amplitude: float, exponent: float) -> FloatArray:
    return amplitude * n ** (-exponent)


def _fit_means(n: FloatArray, y: FloatArray, with_offset: bool) -> tuple[np.ndarray, float]:
    if with_offset:
        initial = [max(float(y[0] * n[0]), 1e-12), 1.0, max(float(y[-1] * 0.05), 0.0)]
        parameters, _ = curve_fit(
            _with_offset, n, y, p0=initial, bounds=([0.0, 0.0, 0.0], [np.inf, 5.0, np.inf]), maxfev=20_000
        )
        predicted = _with_offset(n, *parameters)
    else:
        parameters, _ = curve_fit(
            _no_offset, n, y, p0=[max(float(y[0] * n[0]), 1e-12), 1.0],
            bounds=([0.0, 0.0], [np.inf, 5.0]), maxfev=20_000
        )
        predicted = _no_offset(n, *parameters)
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return parameters, 1.0 - residual / total if total > 0 else 1.0


def fit_scaling_law(
    n_values: NDArray[np.int64],
    replicate_values: FloatArray,
    *,
    with_offset: bool,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> ScalingFit:
    """Fit ``a n^-alpha + b`` (or ``b=0``) and bootstrap replicates.

    ``replicate_values`` has shape ``(number of n values, replicates)``.
    """
    n = np.asarray(n_values, dtype=np.float64)
    values = np.asarray(replicate_values, dtype=np.float64)
    if n.size < 4 or values.shape[0] != n.size:
        raise ValueError("at least four n values with matching replicate rows are required")
    means = values.mean(axis=1)
    parameters, r_squared = _fit_means(n, means, with_offset)
    rng = np.random.default_rng(seed)
    exponents: list[float] = []
    for _ in range(bootstrap_samples):
        boot_means = np.empty(n.size, dtype=np.float64)
        for row in range(n.size):
            sample = rng.choice(values[row], size=values.shape[1], replace=True)
            boot_means[row] = sample.mean()
        try:
            boot_parameters, _ = _fit_means(n, boot_means, with_offset)
            exponents.append(float(boot_parameters[1]))
        except (RuntimeError, ValueError):
            continue
    low, high = np.quantile(exponents, [0.025, 0.975]) if exponents else (np.nan, np.nan)
    offset = float(parameters[2]) if with_offset else 0.0
    return ScalingFit(
        "a*n^-alpha+b" if with_offset else "a*n^-alpha",
        float(parameters[0]), float(parameters[1]), offset,
        float(low), float(high), float(r_squared), int(n.min()), int(n.max()),
        int(n.size), len(exponents),
    )

