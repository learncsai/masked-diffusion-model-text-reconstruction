"""Analytic split-to-split covariance of the skip-gram linear denoiser.

The skip-gram denoiser is a linear operator built from empirical pairwise
conditionals.  Its cross-split fluctuation is therefore driven by the counting
noise in those tables, not by the resolvent that sets the stacking weights: an
offset-space Gram is 16-dimensional and estimated from n(L-|d|) pairs, whereas
each conditional row P(.|b) is estimated from only N_b pairs, and the rare
context tokens dominate.

Conditional on the context count N_b, the count row is multinomial, so with
add-alpha smoothing

    Cov(Phat(.|b)) = N_b / (N_b + alpha)^2 * [diag(p_b) - p_b p_b^T].

Rows with different conditioning tokens are independent, and offsets draw on
largely disjoint pair populations, so the leading prediction is diagonal in
(offset, context token).  Projecting on a probe and averaging over probe draws
uses E_p[sum_a p_a^2 P(a|b)] = 1 and E_p[(sum_a p_a P(a|b))^2] = sum_a P(a|b)^2,
which turns the probe-space variance into 1 - R_b for the collision
probability R_b.  Nothing vocabulary-sized is ever formed: both N_b and R_b
come from bincounts over the training pairs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["conditional_row_statistics", "predicted_split_disagreement"]


def conditional_row_statistics(
    token_ids: np.ndarray,
    offsets: list[int],
    vocabulary: int,
    *,
    smoothing: float = 5.0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return per-offset context counts and collision probabilities.

    For every offset ``d`` and conditioning token ``b`` this returns ``N_b``,
    the number of training pairs with that context, and ``R_b = sum_a
    Phat(a|b)^2`` under the same add-``smoothing`` estimate the denoiser uses.
    ``R_b`` is the Simpson index of the conditional: it is near one for a
    context that always takes the same continuation and near zero for a
    diffuse one, and it is what controls how much probe-space variance the
    multinomial noise produces.
    """
    token_ids = np.asarray(token_ids)
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (sequences, positions)")

    counts: dict[int, np.ndarray] = {}
    collisions: dict[int, np.ndarray] = {}
    for offset in offsets:
        if offset > 0:
            context = token_ids[:, :-offset].reshape(-1)
            target = token_ids[:, offset:].reshape(-1)
        else:
            context = token_ids[:, -offset:].reshape(-1)
            target = token_ids[:, :offset].reshape(-1)
        pair_key = context.astype(np.int64) * vocabulary + target.astype(np.int64)
        unique, pair_counts = np.unique(pair_key, return_counts=True)
        context_of_pair = (unique // vocabulary).astype(np.int64)
        total = np.bincount(context, minlength=vocabulary).astype(np.float64)
        squared = np.bincount(
            context_of_pair, weights=pair_counts.astype(np.float64) ** 2, minlength=vocabulary,
        )
        denominator = np.maximum(total + smoothing, 1.0) ** 2
        counts[offset] = total
        collisions[offset] = np.clip(squared / denominator, 0.0, 1.0)
    return counts, collisions


def predicted_split_disagreement(
    token_ids: np.ndarray,
    target_mask: np.ndarray,
    attention_mask: np.ndarray,
    weights_for: dict[tuple[int, ...], np.ndarray],
    counts: dict[int, np.ndarray],
    collisions: dict[int, np.ndarray],
    offsets: list[int],
    *,
    smoothing: float = 5.0,
) -> np.ndarray:
    """Return the predicted per-sequence cross-split disagreement.

    ``weights_for`` maps a realized offset subset to the stacking weights the
    denoiser uses for it, so the prediction is conditioned on the same mask
    pattern the estimator saw.  Two independent splits each contribute the
    multinomial variance of their own tables, hence the factor of two.  The
    return value is the mean over masked coordinates of the expected squared
    A-B difference, directly comparable to the measured disagreement.
    """
    token_ids = np.asarray(token_ids)
    target_mask = np.asarray(target_mask, dtype=bool)
    attention_mask = np.asarray(attention_mask, dtype=bool)

    sequences = token_ids.shape[0]
    predicted = np.zeros(sequences, dtype=np.float64)
    for row in range(sequences):
        visible = set(np.flatnonzero((~target_mask[row]) & attention_mask[row]).tolist())
        targets = np.flatnonzero(target_mask[row] & attention_mask[row])
        if targets.size == 0:
            continue
        total = 0.0
        for position in targets:
            available = tuple(
                index for index, offset in enumerate(offsets)
                if (position - offset) in visible
            )
            if not available:
                continue
            weights = weights_for[available]
            for slot, index in enumerate(available):
                offset = offsets[index]
                context = token_ids[row, position - offset]
                count = counts[offset][context]
                effective = count / max(count + smoothing, 1.0) ** 2
                total += weights[slot] ** 2 * 2.0 * effective * (
                    1.0 - collisions[offset][context]
                )
        predicted[row] = total / targets.size
    return predicted
