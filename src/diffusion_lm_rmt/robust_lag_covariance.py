"""Empirical sandwich laws for pooled lag-matrix estimators.

The Gaussian law in :mod:`stationary_covariance_theory` accounts exactly for
overlapping products *within* one finite sequence.  Text data add two possible
departures: non-Gaussian fourth moments and dependence between multiple
chunks cut from the same source document.  This module exposes the two
corresponding empirical covariance estimators without ever materialising the
usually enormous covariance matrix.

For a sequence ``s`` and lag ``d`` define

    T_s(d) = w_d / L sum_i y_{s,i} y_{s,i+d}^T.

The lag estimator is the mean of ``T_s``.  Gaussian multiplier draws from the
centred sequence contributions estimate the all-fourth-moment sandwich law.
Aggregating centred contributions by document before applying the multiplier
gives its document-cluster-robust counterpart.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["per_sequence_lag_terms", "multiplier_difference_draws",
           "same_lag_multiplier_difference_draws"]


def per_sequence_lag_terms(
    features: np.ndarray,
    offsets: Sequence[int],
    *,
    weights: Sequence[float] | None = None,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Return flattened per-sequence contributions to tapered lag estimates.

    Dividing every lag by the common sequence length matches the Bartlett
    normalisation used by :func:`stationary_autocovariance`.
    """
    values = np.asarray(features)
    if values.ndim != 3:
        raise ValueError("features must have shape (sequences, positions, dimension)")
    sequences, positions, dimension = values.shape
    offsets = [int(offset) for offset in offsets]
    if any(offset < 0 or offset >= positions for offset in offsets):
        raise ValueError("offsets must lie in [0, positions)")
    lag_weights = np.ones(len(offsets)) if weights is None else np.asarray(weights, dtype=float)
    if lag_weights.shape != (len(offsets),):
        raise ValueError("weights must have one entry per offset")

    out = np.empty((sequences, len(offsets) * dimension * dimension), dtype=dtype)
    block_width = dimension * dimension
    for index, (offset, weight) in enumerate(zip(offsets, lag_weights, strict=True)):
        terms = np.einsum(
            "sid,sie->sde",
            values[:, : positions - offset],
            values[:, offset:],
            optimize=True,
        )
        terms *= float(weight) / positions
        out[:, index * block_width:(index + 1) * block_width] = terms.reshape(
            sequences, block_width
        )
    return out


def multiplier_difference_draws(
    terms: np.ndarray,
    clusters: Sequence[object],
    *,
    target_sequences: int,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Draw a two-corpus lag-estimator difference from a cluster sandwich law.

    ``terms`` is a pooled reference sample, commonly the concatenation of the
    two observed corpora.  Its cluster sums estimate the covariance at that
    pooled sample size.  The multiplier is rescaled to ``target_sequences``
    and includes the factor two for two independent corpora of equal size.
    Unique cluster labels recover the ordinary sequence-level sandwich.
    """
    terms = np.asarray(terms)
    if terms.ndim != 2:
        raise ValueError("terms must be a two-dimensional matrix")
    if terms.shape[0] != len(clusters):
        raise ValueError("clusters must have one label per term row")
    if target_sequences <= 0 or draws <= 0:
        raise ValueError("target_sequences and draws must be positive")

    labels: dict[object, int] = {}
    inverse = np.empty(len(clusters), dtype=np.int64)
    for row, label in enumerate(clusters):
        if label not in labels:
            labels[label] = len(labels)
        inverse[row] = labels[label]
    cluster_count = len(labels)
    if cluster_count < 2:
        raise ValueError("at least two clusters are required")

    centred = terms.astype(np.float64, copy=False) - terms.mean(axis=0, dtype=np.float64)
    sums = np.zeros((cluster_count, terms.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, centred)
    pooled_sequences = terms.shape[0]
    scale = np.sqrt(
        2.0 * cluster_count
        / ((cluster_count - 1) * pooled_sequences * target_sequences)
    )
    rng = np.random.default_rng(seed)
    multipliers = rng.standard_normal((draws, cluster_count))
    sampled = (scale * (multipliers @ sums)).astype(np.float64, copy=False)
    counts = np.bincount(inverse)
    diagnostics: dict[str, float | int] = {
        "clusters": cluster_count,
        "pooled_sequences": pooled_sequences,
        "mean_cluster_size": float(counts.mean()),
        "max_cluster_size": int(counts.max()),
    }
    return sampled, diagnostics


def same_lag_multiplier_difference_draws(
    terms: np.ndarray,
    clusters: Sequence[object],
    *,
    offsets: int,
    dimension: int,
    target_sequences: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Keep each lag's empirical covariance but delete *cross-lag* blocks.

    Independent multiplier streams are applied to successive whole-lag
    slices.  Coordinates within a lag retain their exact empirical/cluster
    covariance.  No approximation to the marginal lag variances is made.
    """
    values = np.asarray(terms)
    block_width = dimension * dimension
    if values.ndim != 2 or values.shape[1] != offsets * block_width:
        raise ValueError("terms width must equal offsets * dimension**2")
    blocks = []
    for offset in range(offsets):
        sampled, _ = multiplier_difference_draws(
            values[:, offset * block_width:(offset + 1) * block_width],
            clusters,
            target_sequences=target_sequences,
            draws=draws,
            seed=seed + 104729 * offset,
        )
        blocks.append(sampled)
    return np.concatenate(blocks, axis=1)
