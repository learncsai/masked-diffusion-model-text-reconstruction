"""Finite-sample fluctuation theory of the pooled stationary covariance estimator.

The Wiener denoiser in :mod:`wiener_bridge` does not use an unstructured
sample covariance.  It pools embedding pairs at equal relative offset,

    Chat(delta) = (1 / (n L)) sum_{s} sum_{i=0}^{L-1-delta} y_{s,i} y_{s,i+delta}^T,

and assembles ``Sigma_{ij} = C(j - i)``.  The Wishart law describes a
different ensemble and overestimates the resulting split disagreement by
almost two orders of magnitude, so the correct object is the covariance of
this pooled estimator.

For a zero-mean stationary Gaussian process, Isserlis gives for the summands

    Cov[ y_i^a y_{i+d}^b , y_j^c y_{j+d'}^{d'} ]
        = C(h)_{ac} C(h + d' - d)_{bd} + C(h + d')_{ad} C(h - d)_{bc},

with ``h = j - i``.  Sequences are independent, so only same-sequence pairs
contribute and the count of ordered position pairs at separation ``h`` enters
through :func:`pair_overlap_counts`.  Hence

    Cov[Chat(d)_{ab}, Chat(d')_{cd}]
        = (1 / (n L^2)) sum_h M(h; d, d')
              [ C(h)_{ac} C(h+d'-d)_{bd} + C(h+d')_{ad} C(h-d)_{bc} ].

Three nested approximations are exposed so their contributions can be
separated: ``independent`` keeps only ``h = 0`` and ``d = d'`` (every pooled
pair treated as an independent observation), ``same_offset`` keeps all ``h``
but only ``d = d'`` (within-sequence correlation, no cross-offset coupling),
and ``full`` keeps everything.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pair_overlap_counts", "lag_matrix", "structured_covariance",
           "sample_stationary_sequences"]


def pair_overlap_counts(positions: int, delta: int, other: int) -> dict[int, int]:
    """Count ordered position pairs ``(i, j)`` with ``j - i = h`` valid for both offsets.

    ``i`` must satisfy ``0 <= i <= L-1-delta`` and ``j = i + h`` must satisfy
    ``0 <= j <= L-1-other``.  Returns only the separations with a positive
    count, which is what makes the sum over ``h`` cheap.
    """
    counts: dict[int, int] = {}
    low_i, high_i = 0, positions - 1 - delta
    low_j, high_j = 0, positions - 1 - other
    if high_i < low_i or high_j < low_j:
        return counts
    for h in range(low_j - high_i, high_j - low_i + 1):
        lo = max(low_i, low_j - h)
        hi = min(high_i, high_j - h)
        if hi >= lo:
            counts[h] = hi - lo + 1
    return counts


def lag_matrix(lags: dict[int, np.ndarray], h: int, dimension: int) -> np.ndarray:
    """Return ``C(h)``, or zero outside the estimated range."""
    block = lags.get(h)
    return np.zeros((dimension, dimension)) if block is None else block


def structured_covariance(
    lags: dict[int, np.ndarray],
    positions: int,
    offsets: list[int],
    sequences: int,
    *,
    level: str = "full",
) -> np.ndarray:
    """Return ``Cov[Chat(d)_{ab}, Chat(d')_{cd}]`` as a ``(O, k, k, O, k, k)`` array.

    ``offsets`` lists the non-negative lags that are modelled; ``sequences`` is
    the number of training sequences ``n``.  ``level`` selects the nesting
    described in the module docstring.  The result is the covariance of a
    single split, so a two-split difference carries an extra factor of two.
    """
    if level not in ("independent", "same_offset", "full"):
        raise ValueError("level must be 'independent', 'same_offset' or 'full'")
    dimension = lags[0].shape[0]
    width = len(offsets)
    out = np.zeros((width, dimension, dimension, width, dimension, dimension))
    scale = 1.0 / (sequences * positions**2)
    for p, delta in enumerate(offsets):
        for r, other in enumerate(offsets):
            if level != "full" and delta != other:
                continue
            counts = pair_overlap_counts(positions, delta, other)
            if level == "independent":
                counts = {0: counts.get(0, 0)}
            for h, count in counts.items():
                first = np.einsum(
                    "ac,bd->abcd",
                    lag_matrix(lags, h, dimension),
                    lag_matrix(lags, h + other - delta, dimension),
                )
                second = np.einsum(
                    "ad,bc->abcd",
                    lag_matrix(lags, h + other, dimension),
                    lag_matrix(lags, h - delta, dimension),
                )
                out[p, :, :, r] += scale * count * (first + second)
    return out


def sample_stationary_sequences(
    lags: dict[int, np.ndarray],
    positions: int,
    sequences: int,
    rng: np.random.Generator,
    *,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Draw Gaussian sequences whose exact lag covariance is ``C``.

    The full ``(L k) x (L k)`` block-Toeplitz covariance is factorised once;
    this is the surrogate against which the analytic law is validated, and the
    sampler used when only the analytic covariance operator is needed rather
    than a closed form.
    """
    dimension = lags[0].shape[0]
    size = positions * dimension
    covariance = np.zeros((size, size))
    for i in range(positions):
        for j in range(positions):
            covariance[
                i * dimension:(i + 1) * dimension, j * dimension:(j + 1) * dimension
            ] = lag_matrix(lags, j - i, dimension)
    covariance = 0.5 * (covariance + covariance.T)
    values, basis = np.linalg.eigh(covariance)
    factor = basis * np.sqrt(np.maximum(values, jitter))
    draw = rng.standard_normal((sequences, size)) @ factor.T
    return draw.reshape(sequences, positions, dimension)
