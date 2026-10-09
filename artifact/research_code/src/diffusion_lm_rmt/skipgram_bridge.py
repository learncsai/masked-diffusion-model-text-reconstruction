"""Offset-tied skip-gram linear denoiser with mask-pattern-specific stacking.

The probe-space estimators (global ``W_q`` and the mask-pattern conditional
mean) code each token as a random Gaussian scalar and fit a position x
position covariance.  Two structural defects follow.  First, averaging over
probe draws leaves only the coincidence part of the cross-position dependence,
``sum_a [P(x_i=a, x_j=a) - P_i(a) P_j(a)]``, which is a few percent of the
total; the estimator is blind to the rest by construction.  Second, the
off-diagonal covariance entries sit barely above their finite-sample noise
floor, so the conditional-mean map is built mostly from spurious correlation.

This module keeps the denoiser strictly linear in the input one-hots but
changes the parameterisation.  Prediction at a masked position is a weighted
sum of pairwise conditional distributions,

    q_i = m + sum_{j in O} w_{i-j} (P^{(i-j)}(. | x_j) - m),

with ``m`` the unigram marginal.  Offsets are tied so each ``P^{(d)}`` pools
``n (L - |d|)`` token pairs instead of ``n`` sequences, and the weights solve
the least-squares problem restricted to the offsets this sequence actually
exposes.  Vocabulary-sized tables are never materialised: the readout is
linear, so only the probe projections ``P^{(d)} p`` are needed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["offsets_within", "fit_skipgram_experts", "stacked_normal_equations",
           "skipgram_linear_denoise"]


def offsets_within(radius: int) -> list[int]:
    """Return the non-zero relative offsets a denoiser of this radius uses."""
    if radius < 1:
        raise ValueError("radius must be positive")
    return [offset for offset in range(-radius, radius + 1) if offset != 0]


def fit_skipgram_experts(
    token_ids: np.ndarray,
    probes: np.ndarray,
    offsets: list[int],
    *,
    smoothing: float = 5.0,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Return probe-projected pairwise conditionals and the unigram readout.

    ``token_ids`` has shape ``(sequences, positions)``.  For each offset ``d``
    the expert is the add-``smoothing`` estimate of ``P(x_{i} = a | x_{i-d})``
    backed off to the unigram marginal.  Only its probe projection is stored,
    centred on the marginal, as a ``(probes, vocabulary)`` array indexed by the
    conditioning token.  That is the entire vocabulary-sized cost: the readout
    ``p . q`` is linear, so the tables themselves are never needed.
    """
    token_ids = np.asarray(token_ids)
    probes = np.asarray(probes, dtype=np.float64)
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (sequences, positions)")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    vocabulary = probes.shape[1]
    marginal = probes[:, token_ids.reshape(-1)].mean(axis=1)
    experts: dict[int, np.ndarray] = {}
    for offset in offsets:
        if offset > 0:
            context = token_ids[:, :-offset].reshape(-1)
            target = token_ids[:, offset:].reshape(-1)
        else:
            context = token_ids[:, -offset:].reshape(-1)
            target = token_ids[:, :offset].reshape(-1)
        counts = np.bincount(context, minlength=vocabulary).astype(np.float64)
        projected = np.stack([
            np.bincount(context, weights=probes[probe][target], minlength=vocabulary)
            for probe in range(probes.shape[0])
        ])
        experts[offset] = (
            (projected + smoothing * marginal[:, None]) / (counts + smoothing)[None, :]
            - marginal[:, None]
        ).astype(np.float32)
    return experts, marginal


def stacked_normal_equations(
    token_ids: np.ndarray,
    fitting_probes: np.ndarray,
    offsets: list[int],
    *,
    smoothing: float = 5.0,
    folds: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Gram matrix and cross-term for the offset stacking weights.

    Experts are refit on ``folds - 1`` folds and evaluated on the held-out
    fold, so the weights see out-of-sample expert behaviour and do not inherit
    the experts' own overfitting.  Because ``E_p[(p . u)(p . v)] = u . v`` for
    standard Gaussian probes, accumulating the design matrix in probe space
    gives an unbiased estimate of the one-hot normal equations without ever
    forming a vocabulary-sized outer product.  The fitting probes must be
    drawn independently of the probes used for reporting.
    """
    token_ids = np.asarray(token_ids)
    fitting_probes = np.asarray(fitting_probes, dtype=np.float64)
    if folds < 2:
        raise ValueError("folds must be at least 2")

    radius = max(abs(offset) for offset in offsets)
    positions = np.arange(radius, token_ids.shape[1] - radius)
    if positions.size == 0:
        raise ValueError("sequence is too short for this radius")

    width, count = len(offsets), fitting_probes.shape[0]
    gram = np.zeros((width, width), dtype=np.float64)
    cross = np.zeros(width, dtype=np.float64)
    partition = np.array_split(np.arange(token_ids.shape[0]), folds)
    for fold in range(folds):
        held = token_ids[partition[fold]]
        fitted = token_ids[np.concatenate([
            partition[other] for other in range(folds) if other != fold
        ])]
        experts, marginal = fit_skipgram_experts(
            fitted, fitting_probes, offsets, smoothing=smoothing,
        )
        design = np.zeros((held.shape[0], positions.size, width, count), dtype=np.float32)
        for index, offset in enumerate(offsets):
            design[:, :, index, :] = experts[offset][:, held[:, positions - offset]].transpose(1, 2, 0)
        response = (
            fitting_probes[:, held[:, positions]].transpose(1, 2, 0) - marginal
        ).astype(np.float32)
        flat = design.reshape(-1, width, count).transpose(0, 2, 1).reshape(-1, width)
        gram += flat.T @ flat
        cross += flat.T @ response.reshape(-1)
    return gram, cross


def skipgram_linear_denoise(
    token_ids: np.ndarray,
    target_mask: np.ndarray,
    attention_mask: np.ndarray,
    experts: dict[int, np.ndarray],
    marginal: np.ndarray,
    gram: np.ndarray,
    cross: np.ndarray,
    offsets: list[int],
    *,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Reconstruct masked positions, solving per realized offset pattern.

    Returns a ``(sequences, positions, probes)`` array.  For every masked
    position the available offsets are those whose source position is both
    visible and inside the sequence; the stacking weights are the ridge
    solution of the normal equations restricted to that subset, so redundancy
    between overlapping context tokens is corrected rather than averaged over.
    Solutions are cached by offset subset, of which there are few.  Positions
    with no available offset fall back to the unigram marginal.
    """
    token_ids = np.asarray(token_ids)
    target_mask = np.asarray(target_mask, dtype=bool)
    attention_mask = np.asarray(attention_mask, dtype=bool)
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    sequences, positions = token_ids.shape
    prediction = np.tile(marginal[None, None, :], (sequences, positions, 1)).astype(np.float64)
    cache: dict[tuple[int, ...], np.ndarray] = {}
    for row in range(sequences):
        visible = set(np.flatnonzero((~target_mask[row]) & attention_mask[row]).tolist())
        for position in np.flatnonzero(target_mask[row] & attention_mask[row]):
            available = tuple(
                index for index, offset in enumerate(offsets)
                if (position - offset) in visible
            )
            if not available:
                continue
            if available not in cache:
                block = gram[np.ix_(available, available)]
                penalty = ridge * np.trace(block) / len(available)
                cache[available] = np.linalg.solve(
                    block + penalty * np.eye(len(available)), cross[list(available)],
                )
            contributions = np.stack([
                experts[offsets[index]][:, token_ids[row, position - offsets[index]]]
                for index in available
            ])
            prediction[row, position] = marginal + cache[available] @ contributions
    return prediction
