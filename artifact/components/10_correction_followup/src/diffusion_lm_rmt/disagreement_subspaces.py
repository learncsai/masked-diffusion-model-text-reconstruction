"""Matrix extensions of the existing lag disagreement laws and energy capture."""
from __future__ import annotations
import numpy as np
from . import sparse_categorical_lag as lag
from .categorical_lag import pattern_weights


def vocabulary_readout(classes, dimension=128, seed=20261001):
    z = np.random.default_rng(seed).standard_normal((classes, dimension))
    z -= z.mean(axis=0)
    q, r = np.linalg.qr(z, mode='reduced')
    return q * np.where(np.diag(r) < 0, -1., 1.)


def source_folds(source_ids, seed=20261002):
    names, counts = np.unique(source_ids, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(names))
    order = sorted(order, key=lambda j: -counts[j])  # seeded tie breaking
    totals, mapping = [0, 0], {}
    for j in order:
        fold = int(totals[1] < totals[0])
        mapping[names[j]] = fold
        totals[fold] += int(counts[j])
    if not all(totals):
        raise ValueError('Two nonempty source groups required')
    return np.array([mapping[s] for s in source_ids]), np.asarray(totals)


def point_weights(mask, folds):
    rows = np.nonzero(mask)[0]
    count = mask.sum(1)
    return rows, np.array([(folds[rows] == f) / count[rows] for f in (0, 1)])


def second_moments(delta, mask, folds):
    _, weights = point_weights(mask, folds)
    return np.array([delta.T @ (w[:, None] * delta) for w in weights])


def projected_prediction(ids, mask, train, prior, offsets, gram, cross, smoothing, ridge, readout):
    tables = lag.conditional_tables(lag.pair_counts(train, offsets, len(prior)), prior, smoothing)
    _, _, pred = lag.masked_prediction(ids, mask, tables, prior, offsets, gram, cross, ridge)
    return pred.residual @ readout + pred.prior_weight[:, None] * (prior @ readout)


def design(ids, mask, offsets, gram, cross, ridge):
    rows, _, contexts, visible, patterns = lag.context_design(ids, mask, offsets)
    weights = np.zeros((len(rows), len(offsets)))
    for code in np.unique(patterns):
        if code:
            active = tuple(j for j in range(len(offsets)) if code & (1 << j))
            weights[np.ix_(np.flatnonzero(patterns == code), active)] = pattern_weights(gram, cross, active, ridge)
    return rows, contexts, visible, weights


def forecast_matrices(reference, groups, ids, mask, folds, prior, offsets, gram, cross,
                      *, n, smoothing, ridge, readout, progress=None):
    """Return fold SUMS of chunk-mean covariance matrices (not fold means).

    Source matrices retain ratio centering and source-size centering. The
    multinomial matrix uses probabilities smoothed at reference size.
    Independent sources are the sampling units, not tokens or chunks.
    """
    classes, d = readout.shape
    counts = lag.pair_counts(reference, offsets, classes)
    scale = n / len(reference)
    rows, contexts, visible, weights = design(ids, mask, offsets, gram, cross, ridge)
    _, averaging = point_weights(mask, folds)
    m, points = len(groups), len(rows)
    prior_z = prior @ readout
    prior_second = readout.T @ (prior[:, None] * readout)
    info, ref_parts = [], []
    multi = np.zeros((2, d, d))
    for j, offset in enumerate(offsets):
        selected = np.flatnonzero(visible[j])
        unique, inverse = np.unique(contexts[j][selected], return_inverse=True)
        small = counts[offset][unique]
        totals = np.asarray(small.sum(1)).ravel()
        denom = scale * totals + smoothing
        raw_z = small @ readout
        prob_z = (scale * raw_z + smoothing * prior_z) / denom[:, None]
        probability = (raw_z + smoothing * prior_z) / (totals + smoothing)[:, None]
        a = weights[selected, j] / denom[inverse]
        ref = np.zeros((points, d))
        ref[selected] = a[:, None] * (raw_z[inverse] - totals[inverse, None] * prob_z[inverse])
        ref_parts.append(ref)
        info.append((offset, selected, unique, inverse, a, prob_z))
        # Aggregate diagonal categorical second moments before subtracting means.
        factor = 2 * scale * totals / denom**2
        for f in (0, 1):
            coeff = np.bincount(inverse, weights=averaging[f, selected] * weights[selected, j]**2,
                                minlength=len(unique)) * factor
            token_mass = np.asarray(small.T @ (coeff / (totals + smoothing))).ravel()
            diagonal = readout.T @ (token_mass[:, None] * readout)
            diagonal += np.sum(coeff * smoothing / (totals + smoothing)) * prior_second
            multi[f] += diagonal - probability.T @ (coeff[:, None] * probability)
    full = np.zeros((2, d, d))
    same = np.zeros_like(full)
    weighted = [np.zeros((points, d)) for _ in offsets]
    squared_chunks = 0
    for g, group in enumerate(groups):
        source = lag.pair_counts(reference[group], offsets, classes)
        parts = []
        b = len(group)
        squared_chunks += b*b
        for j, (offset, selected, unique, inverse, a, prob_z) in enumerate(info):
            small = source[offset][unique]
            totals = np.asarray(small.sum(1)).ravel()
            centered = small @ readout - totals[:, None] * prob_z
            part = np.zeros((points, d))
            part[selected] = a[:, None] * centered[inverse]
            weighted[j] += b * part
            parts.append(part)
            active = np.flatnonzero(totals[inverse] > 0)
            for f in (0, 1):
                w = averaging[f, selected[active]]
                z = part[selected[active]]
                same[f] += z.T @ (w[:, None] * z)
        summed = sum(parts)
        active = np.flatnonzero(np.any(summed != 0, axis=1))
        z = summed[active]
        for f in (0, 1):
            full[f] += z.T @ (averaging[f, active, None] * z)
        if progress is not None and (g+1) % 128 == 0:
            progress(g+1, m)
    # Algebraically expand sum_g (u_g - b_g*u_ref/n_ref)(...)^T.
    def center(matrix, ref, weighted_sum):
        for f in (0, 1):
            w = averaging[f, :, None]
            cross_term = ref.T @ (w * weighted_sum) / len(reference)
            matrix[f] += -cross_term - cross_term.T
            matrix[f] += squared_chunks / len(reference)**2 * (ref.T @ (w * ref))
    center(full, sum(ref_parts), sum(weighted))
    for ref, weighted_sum in zip(ref_parts, weighted):
        center(same, ref, weighted_sum)
    factor = 2 * (n*m/len(reference)) / (m-1)
    result = dict(source_full=full*factor, source_same_lag=same*factor, multinomial=multi)
    for value in result.values():
        value[:] = (value + value.transpose(0, 2, 1)) / 2
        if np.min(np.linalg.eigvalsh(value)) < -1e-10:
            raise ArithmeticError('Forecast covariance is not positive semidefinite')
    return result


def leading_basis(matrix, k):
    values, vectors = np.linalg.eigh((matrix + matrix.T)/2)
    if values[-k] < 1e-12 * max(values[-1], 1e-30):
        raise ValueError('Requested subspace includes a numerically null direction')
    return vectors[:, -k:]


def capture(method, neural, k, crossfit=True):
    denominator = float(np.trace(neural.sum(0)))
    if crossfit:
        numerator = sum(float(np.trace((u := leading_basis(method[1-f], k)).T @ neural[f] @ u))
                        for f in (0, 1))
    else:
        u = leading_basis(method.sum(0), k)
        numerator = float(np.trace(u.T @ neural.sum(0) @ u))
    return numerator/denominator


def random_capture(neural, ks=(1,2,4,8,16), draws=500, seed=20261003, crossfit=True):
    rng = np.random.default_rng(seed)
    d = neural.shape[-1]
    denominator = np.trace(neural.sum(0))
    if not crossfit:
        neural = neural.sum(0, keepdims=True)
    values = np.zeros((draws, len(ks)))
    for trial in range(draws):
        # Independent random spaces for each held-out fold, like cross-fitting.
        for f in range(len(neural)):
            u, _ = np.linalg.qr(rng.standard_normal((d, max(ks))))
            energy = np.cumsum(np.sum(u * (neural[f] @ u), axis=0)) / denominator
            values[trial] += energy[np.array(ks)-1]
    return values
