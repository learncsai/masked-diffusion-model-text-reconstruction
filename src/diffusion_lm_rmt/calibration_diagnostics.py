"""Controlled sparse diagnostics; this module never reads A/B outcomes.

An exact and a linearized reconstructor see the SAME resampled counts.
Their variance decomposition is algebraic, not a fitted explanation.
Unequal-size draws use independent RNG streams. The squared-mean term
subtracts Monte Carlo mean-estimation variance rather than clipping it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from . import sparse_categorical_lag as backend


def add(left, right, right_scale=1.0):
    return backend.Prediction(
        (left.residual + right_scale * right.residual).tocsr(),
        left.prior_weight + right_scale * right.prior_weight, left.prior,
    )


def scale(value, factor):
    return backend.Prediction(value.residual * factor,
                              value.prior_weight * factor, value.prior)


def score_weights(mask):
    """Each chunk has equal weight, then its hidden positions share it."""
    rows = np.nonzero(mask)[0]
    hidden = np.maximum(mask.sum(axis=1), 1)
    return 1.0 / (len(mask) * hidden[rows])


class Linearization:
    """First-order conditional rows at fixed expected lag counts."""

    def __init__(self, baseline_counts, prior, smoothing):
        self.tables = backend.conditional_tables(baseline_counts, prior, smoothing)
        self.totals = {d: np.asarray(c.sum(axis=1)).ravel()
                       for d, c in baseline_counts.items()}
        self.inverse = {d: 1.0 / (v + smoothing) for d, v in self.totals.items()}

    def tables_at(self, counts):
        result = {}
        for d, c in counts.items():
            delta_total = np.asarray(c.sum(axis=1)).ravel() - self.totals[d]
            relative_delta = delta_total * self.inverse[d]
            base, beta = self.tables[d]
            # P(B) + J_B(C-B) = C/d_B - (delta_t/d_B) B/d_B
            #                   + alpha p/d_B (1-delta_t/d_B).
            result[d] = (
                (c.multiply(self.inverse[d][:, None])
                 - base.multiply(relative_delta[:, None])).tocsr(),
                beta * (1.0 - relative_delta),
            )
        return result


@dataclass
class Moments:
    total: backend.Prediction
    sum_norm: np.ndarray
    count: int = 0

    @classmethod
    def empty(cls, points, prior):
        return cls(backend.Prediction(sparse.csr_matrix((points, len(prior))),
                                     np.zeros(points), prior), np.zeros(points))

    def update(self, output):
        self.total = add(self.total, output)
        self.sum_norm += output.norm2()
        self.count += 1

    def mean(self):
        if self.count == 0:
            raise ValueError("empty Monte Carlo sample")
        return scale(self.total, 1.0 / self.count)

    def variance(self):
        if self.count < 2:
            raise ValueError("at least two draws required")
        centered = self.sum_norm - self.total.norm2() / self.count
        if centered.min(initial=0) < -1e-9 * max(1., self.sum_norm.max(initial=0)):
            raise ArithmeticError("negative Monte Carlo variance")
        return np.maximum(centered, 0) / (self.count - 1)


class PairedMoments:
    def __init__(self, points, prior):
        self.exact = Moments.empty(points, prior)
        self.linear = Moments.empty(points, prior)
        self.remainder = Moments.empty(points, prior)
        self.sum_cross = np.zeros(points)

    def update(self, exact, linear):
        remainder = add(exact, linear, -1)
        self.exact.update(exact)
        self.linear.update(linear)
        self.remainder.update(remainder)
        self.sum_cross += backend.row_inner(linear, remainder)

    def components(self):
        count = self.exact.count
        ve = self.exact.variance()
        vl = self.linear.variance()
        vr = self.remainder.variance()
        covariance = (self.sum_cross - backend.row_inner(
            self.linear.total, self.remainder.total) / count) / (count - 1)
        np.testing.assert_allclose(ve, vl + vr + 2 * covariance,
                                   rtol=1e-8, atol=1e-11)
        return dict(exact_variance=ve, linear_variance=vl,
                    remainder_variance=vr, twice_covariance=2 * covariance)


def unequal_components(left: Moments, right: Moments):
    """Unbiased terms for independent Monte Carlo ensembles (pointwise).

The corrected squared-mean estimate can be negative from Monte Carlo
noise. Do not clip it, which would introduce an upward bias.
"""
    va, vb = left.variance(), right.variance()
    raw = backend.squared_difference(left.mean(), right.mean())
    correction = va / left.count + vb / right.count
    mean_term = raw - correction
    return dict(variance_sum=va + vb, mean_square_raw=raw,
                mean_mc_correction=correction, mean_square_unbiased=mean_term,
                total_disagreement=va + vb + mean_term)
