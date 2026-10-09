"""Exact sparse evaluation of the source-influence law in the manuscript."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .categorical_lag import pattern_weights, per_sequence_squared_error
from .sparse_categorical_lag import (
    Prediction, conditional_tables, context_design, pair_counts, row_inner,
)


def sparse_row_inner(left: sparse.csr_matrix, right: Prediction) -> np.ndarray:
    """Visit only the small source's nonzeros, not the reference row support."""
    coo = left.tocoo()
    result = np.zeros(left.shape[0])
    if coo.nnz:
        values = np.asarray(right.residual[coo.row, coo.col]).ravel()
        result += np.bincount(coo.row, weights=coo.data * values, minlength=left.shape[0])
    return result + (left @ right.prior) * right.prior_weight


def source_influence_disagreement(
    reference_ids, groups, evaluation_ids, mask, prior, offsets, gram, cross,
    *, n, smoothing, ridge, same_lag=False,
):
    documents, reference_chunks = len(groups), len(reference_ids)
    if documents < 2:
        raise ValueError("at least two reference sources are needed")
    classes = len(prior)
    counts = pair_counts(reference_ids, offsets, classes)
    baseline = {d: c * (n / reference_chunks) for d, c in counts.items()}
    tables = conditional_tables(baseline, prior, smoothing)
    denominators = {
        d: np.asarray(baseline[d].sum(axis=1)).ravel() + smoothing for d in offsets
    }
    rows, _, contexts, visible, patterns = context_design(evaluation_ids, mask, offsets)
    points = len(rows)
    weights = np.zeros((points, len(offsets)))
    for code in np.unique(patterns):
        if code:
            active = tuple(j for j in range(len(offsets)) if code & (1 << j))
            weights[np.ix_(np.flatnonzero(patterns == code), active)] = pattern_weights(
                gram, cross, active, ridge,
            )
    selectors, conditional = [], []
    for j, d in enumerate(offsets):
        selected = np.flatnonzero(visible[j])
        context = contexts[j][selected]
        select = sparse.csr_matrix(
            (np.ones(len(selected)), (selected, context)), shape=(points, classes),
        )
        conditional.append(Prediction(select @ tables[d][0], select @ tables[d][1], prior))
        selectors.append(sparse.csr_matrix(
            (weights[selected, j] / denominators[d][context], (selected, context)),
            shape=(points, classes),
        ))
    # u_g = sum_j (a_gj - t_gj P_j). The P_j inner products are shared by
    # every source, so do not expand their dense support for each source.
    kernels = {(j, k): row_inner(conditional[j], conditional[k])
               for j in range(len(offsets)) for k in range(j + 1)}
    def source_components(source_counts):
        pieces = [selector @ source_counts[d] for selector, d in zip(selectors, offsets)]
        totals = [np.asarray(a.sum(axis=1)).ravel() for a in pieces]
        return pieces, totals

    ref_parts, ref_totals = source_components(counts)
    ref_outputs = []
    for j, (part, total) in enumerate(zip(ref_parts, ref_totals)):
        ref_outputs.append(Prediction(
            (part - conditional[j].residual.multiply(total[:, None])).tocsr(),
            -total * conditional[j].prior_weight, prior,
        ))
    if same_lag:
        reference_norm = sum((v.norm2() for v in ref_outputs), np.zeros(points))
        ref_kernels = [row_inner(conditional[j], ref_outputs[j]) for j in range(len(offsets))]
    else:
        reference = Prediction(
            sum((v.residual for v in ref_outputs), sparse.csr_matrix((points, classes))),
            sum((v.prior_weight for v in ref_outputs), np.zeros(points)), prior,
        )
        reference_norm = reference.norm2()
        ref_kernels = [row_inner(p, reference) for p in conditional]

    norm_sum = np.zeros(points)
    weighted_cross = np.zeros(points)
    squared_chunks = 0
    for group in groups:
        pieces, totals = source_components(pair_counts(reference_ids[group], offsets, classes))
        chunks = len(group)
        squared_chunks += chunks**2
        if same_lag:
            for j, (part, total) in enumerate(zip(pieces, totals)):
                norm_sum += np.asarray(part.multiply(part).sum(axis=1)).ravel()
                norm_sum -= 2 * total * sparse_row_inner(part, conditional[j])
                norm_sum += total**2 * kernels[j, j]
                weighted_cross += chunks * (
                    sparse_row_inner(part, ref_outputs[j]) - total * ref_kernels[j]
                )
        else:
            total_part = sum(pieces, sparse.csr_matrix((points, classes)))
            norm_sum += np.asarray(total_part.multiply(total_part).sum(axis=1)).ravel()
            cross_value = sparse_row_inner(total_part, reference)
            for j, total in enumerate(totals):
                norm_sum -= 2 * total * sparse_row_inner(total_part, conditional[j])
                cross_value -= total * ref_kernels[j]
                for k in range(j + 1):
                    norm_sum += (1 if j == k else 2) * total * totals[k] * kernels[j, k]
            weighted_cross += chunks * cross_value
    centered = (norm_sum - 2 * weighted_cross / reference_chunks
                + squared_chunks * reference_norm / reference_chunks**2)
    if np.min(centered, initial=0) < -1e-10:
        raise ArithmeticError("negative source-influence variance")
    target_documents = n * documents / reference_chunks
    point = 2 * target_documents * np.maximum(centered, 0) / (documents - 1)
    return per_sequence_squared_error(rows, point, len(evaluation_ids))
