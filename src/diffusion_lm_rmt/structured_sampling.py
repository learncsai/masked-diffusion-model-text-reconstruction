"""Matrix-free propagation for the structured covariance theory at full k.

At k=16 the offset-domain covariance of ``Chat`` is 2304 x 2304 and can be
factorised directly.  At k=96 it is 82944 x 82944 and cannot be formed, so the
same law has to be realised without writing it down.

The route used here is to simulate stationary Gaussian sequences with lag
covariance ``C`` and apply the *same* pooling estimator: the resulting draws
have exactly the Isserlis covariance, cross-offset terms included, so no
tensor is needed.  Sampling goes through a Cholesky factor of the block-
Toeplitz covariance, computed once per (replicate, size) and reused for every
draw and every mask rate.

A frequency-domain sampler would be cheaper, but a circulant construction was
tried and rejected: it reproduces the same-offset variances to within a
percent while collapsing the cross-offset covariance to zero, which is exactly
the term that matters here.  Correctness was preferred over the speedup.

The propagation exploits banding instead.  ``dSigma`` inherits the estimator's
block-Toeplitz structure, so multiplying by it costs ``S L (2R+1) k^2`` rather
than ``S (L k)^2`` -- a factor of ``L / (2R+1)`` saved, verified exact against
the dense product.
"""

from __future__ import annotations

import numpy as np

__all__ = ["toeplitz_cholesky", "sample_sequences", "pooled_lags",
           "banded_right_multiply", "whiten_lags"]


def toeplitz_cholesky(lags: dict[int, np.ndarray], positions: int,
                      *, jitter: float = 1e-10) -> np.ndarray:
    """Factor the block-Toeplitz covariance once, for repeated sampling."""
    dimension = lags[0].shape[0]
    size = positions * dimension
    covariance = np.zeros((size, size))
    for i in range(positions):
        for j in range(positions):
            block = lags.get(j - i)
            if block is not None:
                covariance[i * dimension:(i + 1) * dimension,
                           j * dimension:(j + 1) * dimension] = block
    covariance = 0.5 * (covariance + covariance.T)
    values, basis = np.linalg.eigh(covariance)
    return (basis * np.sqrt(np.maximum(values, jitter))).astype(np.float32)


def sample_sequences(factor: np.ndarray, positions: int, sequences: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Draw sequences with the factored covariance."""
    dimension = factor.shape[0] // positions
    noise = rng.standard_normal((sequences, factor.shape[0])).astype(np.float32)
    return (noise @ factor.T).reshape(sequences, positions, dimension)


def pooled_lags(features: np.ndarray, radius: int) -> dict[int, np.ndarray]:
    """Apply the project's pooling estimator, 1/(nL) taper, to sampled features."""
    sequences, positions, _ = features.shape
    lags: dict[int, np.ndarray] = {}
    for lag in range(radius + 1):
        block = np.einsum("sid,sie->de", features[:, : positions - lag], features[:, lag:],
                          optimize=True).astype(np.float64)
        block /= sequences * positions
        lags[lag], lags[-lag] = block, block.T
    return lags


def whiten_lags(lags: dict[int, np.ndarray], whitener: np.ndarray,
                radius: int) -> dict[int, np.ndarray]:
    """Whiten each lag block; block-diagonal whitening preserves the banding."""
    return {d: whitener @ lags[d] @ whitener for d in range(-radius, radius + 1)}


def banded_right_multiply(left: np.ndarray, lags: dict[int, np.ndarray],
                          positions: int, radius: int) -> np.ndarray:
    """Return ``left @ Sigma`` for a block-Toeplitz ``Sigma`` with banded lags.

    ``Sigma_{ij} = C(j - i)``, so column block ``j`` collects
    ``sum_i left[:, i] C(j - i)`` over the ``2R+1`` in-band separations.
    """
    dimension = lags[0].shape[0]
    blocks = left.reshape(left.shape[0], positions, dimension)
    out = np.zeros_like(blocks)
    for delta in range(-radius, radius + 1):
        block = lags[delta]
        lo, hi = max(0, -delta), min(positions, positions - delta)
        if hi > lo:
            out[:, lo + delta:hi + delta, :] += blocks[:, lo:hi, :] @ block
    return out.reshape(left.shape)
