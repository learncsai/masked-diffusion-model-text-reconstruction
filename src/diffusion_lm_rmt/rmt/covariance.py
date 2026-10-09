"""Positive-definite population covariance families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
CovarianceFamily = Literal["isotropic", "spiked", "power_law", "broad"]
EigenvectorMode = Literal["fixed", "random"]


@dataclass(frozen=True)
class CovarianceModel:
    """A covariance matrix together with its ordered eigensystem."""

    matrix: FloatArray
    eigenvalues: FloatArray
    eigenvectors: FloatArray
    family: str


def _orthogonal_matrix(d: int, rng: np.random.Generator) -> FloatArray:
    raw = rng.standard_normal((d, d))
    q, r = np.linalg.qr(raw)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return np.asarray(q * signs, dtype=np.float64)


def generate_covariance(
    family: CovarianceFamily,
    d: int,
    *,
    seed: int = 0,
    eigenvectors: EigenvectorMode = "random",
    spike_strengths: Sequence[float] = (8.0, 3.0),
    alpha: float = 1.25,
    broad_condition_number: float = 100.0,
    normalize_trace: bool = True,
) -> CovarianceModel:
    """Construct a reproducible symmetric positive-definite covariance.

    Power-law and broad spectra are normalized to mean eigenvalue one by
    default, making noise levels comparable across covariance families.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    rng = np.random.default_rng(seed)
    if family == "isotropic":
        values = np.ones(d, dtype=np.float64)
    elif family == "spiked":
        values = np.ones(d, dtype=np.float64)
        for index, strength in enumerate(spike_strengths[:d]):
            if strength < 0:
                raise ValueError("spike strengths must be nonnegative")
            values[index] += float(strength)
    elif family == "power_law":
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        values = np.arange(1, d + 1, dtype=np.float64) ** (-alpha)
    elif family == "broad":
        if broad_condition_number <= 1:
            raise ValueError("broad_condition_number must exceed one")
        # A geometric spectrum is an empirical-like, broadly distributed
        # positive spectrum without unstable stochastic outliers.
        values = np.geomspace(broad_condition_number, 1.0, d, dtype=np.float64)
    else:
        raise ValueError(f"unknown covariance family: {family}")

    if normalize_trace and family in {"power_law", "broad"}:
        values = values / values.mean()
    basis = np.eye(d, dtype=np.float64) if eigenvectors == "fixed" else _orthogonal_matrix(d, rng)
    matrix = (basis * values) @ basis.T
    matrix = np.asarray((matrix + matrix.T) * 0.5, dtype=np.float64)
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise RuntimeError("covariance construction lost symmetry")
    if np.linalg.eigvalsh(matrix).min() <= 0:
        raise RuntimeError("covariance is not positive definite")
    return CovarianceModel(matrix, values, basis, family)

