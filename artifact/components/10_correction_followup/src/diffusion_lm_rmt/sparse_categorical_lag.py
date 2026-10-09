"""Exact full-vocabulary lag reconstruction without dense K-by-K tables.

Every row vector is stored as a sparse part plus a multiple of the shared
unigram prior. This is algebra, not truncation or an approximate embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .categorical_lag import draw_reference_chunks, pattern_weights, per_sequence_squared_error


@dataclass
class Prediction:
    residual: sparse.csr_matrix
    prior_weight: np.ndarray
    prior: np.ndarray

    def norm2(self) -> np.ndarray:
        result = np.asarray(self.residual.multiply(self.residual).sum(axis=1)).ravel()
        result += 2 * self.prior_weight * (self.residual @ self.prior)
        result += self.prior_weight**2 * np.dot(self.prior, self.prior)
        return result

    def target_values(self, targets: np.ndarray) -> np.ndarray:
        if not len(targets):
            return np.empty(0, dtype=np.float64)
        return np.asarray(self.residual[np.arange(len(targets)), targets]).ravel() + (
            self.prior_weight * self.prior[targets]
        )

    def loss(self, targets: np.ndarray) -> np.ndarray:
        return 1 - 2 * self.target_values(targets) + self.norm2()

    def toarray(self) -> np.ndarray:
        """For small-vocabulary equivalence tests only."""
        return self.residual.toarray() + self.prior_weight[:, None] * self.prior


def row_inner(left: Prediction, right: Prediction) -> np.ndarray:
    result = np.asarray(left.residual.multiply(right.residual).sum(axis=1)).ravel()
    result += left.prior_weight * (right.residual @ left.prior)
    result += right.prior_weight * (left.residual @ left.prior)
    result += left.prior_weight * right.prior_weight * np.dot(left.prior, left.prior)
    return result


def squared_difference(left: Prediction, right: Prediction) -> np.ndarray:
    difference = Prediction(
        left.residual - right.residual,
        left.prior_weight - right.prior_weight,
        left.prior,
    )
    return difference.norm2()


def pair_counts(ids: np.ndarray, offsets: tuple[int, ...], classes: int) -> dict:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 2 or np.any((ids < 0) | (ids >= classes)):
        raise ValueError("ids must be a matrix of valid vocabulary IDs")
    result = {}
    for d in offsets:
        if not 0 < abs(d) < ids.shape[1]:
            raise ValueError("lag must be nonzero and shorter than a sequence")
        if d > 0:
            context, target = ids[:, :-d], ids[:, d:]
        else:
            context, target = ids[:, -d:], ids[:, :d]
        counts = sparse.coo_matrix(
            (np.ones(context.size), (context.ravel(), target.ravel())),
            shape=(classes, classes),
        ).tocsr()
        counts.sum_duplicates()
        counts.sort_indices()
        result[d] = counts
    return result


def conditional_tables(counts: dict, prior: np.ndarray, smoothing: float) -> dict:
    if smoothing <= 0 or np.any(prior < 0) or not np.isclose(prior.sum(), 1):
        raise ValueError("positive smoothing and a probability prior are required")
    result = {}
    for d, counts_d in counts.items():
        totals = np.asarray(counts_d.sum(axis=1)).ravel()
        inverse = 1 / (totals + smoothing)
        result[d] = (
            counts_d.multiply(inverse[:, None]).tocsr(),
            smoothing * inverse,
        )
    return result


def context_design(ids: np.ndarray, mask: np.ndarray, offsets: tuple[int, ...]):
    if ids.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError("matching ID and boolean mask arrays required")
    rows, positions = np.nonzero(mask)
    contexts, visibility = [], []
    patterns = np.zeros(len(rows), dtype=np.int64)
    for j, d in enumerate(offsets):
        source = positions - d
        inside = (source >= 0) & (source < ids.shape[1])
        visible = np.zeros(len(rows), dtype=bool)
        visible[inside] = ~mask[rows[inside], source[inside]]
        context = np.zeros(len(rows), dtype=np.int64)
        context[visible] = ids[rows[visible], source[visible]]
        contexts.append(context)
        visibility.append(visible)
        patterns |= visible.astype(np.int64) << j
    return rows, positions, contexts, visibility, patterns


def masked_prediction(
    ids, mask, tables, prior, offsets, gram, cross, ridge, weight_cache=None,
):
    rows, positions, contexts, visibility, patterns = context_design(ids, mask, offsets)
    if weight_cache is None:
        weight_cache = {}
    weights = np.zeros((len(rows), len(offsets)))
    for code in np.unique(patterns):
        if not code:
            continue
        available = tuple(j for j in range(len(offsets)) if code & (1 << j))
        if int(code) not in weight_cache:
            weight_cache[int(code)] = pattern_weights(gram, cross, available, ridge)
        weights[np.ix_(np.flatnonzero(patterns == code), available)] = weight_cache[int(code)]
    residual = sparse.csr_matrix((len(rows), len(prior)), dtype=np.float64)
    beta = np.ones(len(rows))
    for j, d in enumerate(offsets):
        selected = np.flatnonzero(visibility[j])
        if not len(selected):
            continue
        context = contexts[j][selected]
        table, prior_fraction = tables[d]
        selection = sparse.csr_matrix(
            (weights[selected, j], (selected, context)),
            shape=(len(rows), len(prior)),
        )
        residual += selection @ table
        beta[selected] += weights[selected, j] * (prior_fraction[context] - 1)
    residual.eliminate_zeros()
    return rows, ids[rows, positions], Prediction(residual, beta, prior)


def fitting_equations(
    fitting_ids, validation_ids, prior, offsets, smoothing, mask_rate, rng,
    *, batch_chunks=16,
):
    """Same equations and random mask as the dense implementation, in batches."""
    tables = conditional_tables(pair_counts(fitting_ids, offsets, len(prior)), prior, smoothing)
    mask = rng.random(validation_ids.shape) < mask_rate
    gram = np.zeros((len(offsets), len(offsets)))
    cross = np.zeros(len(offsets))
    for start in range(0, len(validation_ids), batch_chunks):
        ids = validation_ids[start:start + batch_chunks]
        block_mask = mask[start:start + batch_chunks]
        rows, positions, contexts, visibility, _ = context_design(ids, block_mask, offsets)
        targets = ids[rows, positions]
        designs = []
        for j, d in enumerate(offsets):
            selected = np.flatnonzero(visibility[j])
            select = sparse.csr_matrix(
                (np.ones(len(selected)), (selected, contexts[j][selected])),
                shape=(len(rows), len(prior)),
            )
            table, prior_fraction = tables[d]
            coefficient = np.zeros(len(rows))
            coefficient[selected] = prior_fraction[contexts[j][selected]] - 1
            design = Prediction(select @ table, coefficient, prior)
            designs.append(design)
            cross[j] += np.sum(design.target_values(targets) - (
                design.residual @ prior + coefficient * np.dot(prior, prior)
            ))
        for j in range(len(offsets)):
            for k in range(j + 1):
                value = float(np.sum(row_inner(designs[j], designs[k])))
                gram[j, k] += value
                if j != k:
                    gram[k, j] += value
    return gram, cross


def bootstrap_disagreement(
    reference_ids, groups, evaluation_ids, mask, prior, offsets, gram, cross,
    *, n, smoothing, ridge, draws, law, rng,
):
    """Twice sample output variance, retaining sparse-plus-prior outputs."""
    if draws < 2:
        raise ValueError("at least two bootstrap draws are needed")
    if law not in {"source_cluster", "independent_chunks", "same_lag"}:
        raise ValueError(f"unknown law {law}")
    rows = np.nonzero(mask)[0]
    total = sparse.csr_matrix((len(rows), len(prior)), dtype=np.float64)
    beta_sum = np.zeros(len(rows))
    norm_sum = np.zeros(len(rows))
    cache = {}
    for _ in range(draws):
        if law == "same_lag":
            counts = {
                d: pair_counts(
                    draw_reference_chunks(reference_ids, groups, n, rng),
                    (d,), len(prior),
                )[d] for d in offsets
            }
        else:
            sample = draw_reference_chunks(
                reference_ids, groups, n, rng,
                independent_chunks=(law == "independent_chunks"),
            )
            counts = pair_counts(sample, offsets, len(prior))
        tables = conditional_tables(counts, prior, smoothing)
        rows, _, output = masked_prediction(
            evaluation_ids, mask, tables, prior, offsets, gram, cross, ridge,
            weight_cache=cache,
        )
        total += output.residual
        beta_sum += output.prior_weight
        norm_sum += output.norm2()
    total_norm = Prediction(total, beta_sum, prior).norm2()
    centered = norm_sum - total_norm / draws
    if np.min(centered, initial=0) < -1e-10 * max(1, float(norm_sum.max(initial=0))):
        raise ArithmeticError("negative sample variance beyond numerical tolerance")
    point = 2 * np.maximum(centered, 0) / (draws - 1)
    return per_sequence_squared_error(rows, point, len(evaluation_ids)), np.full(draws, n)
