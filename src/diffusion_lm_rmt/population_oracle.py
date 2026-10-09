"""Exact contracted source moments for the finite-topic reset/shift control.

No reference or A/B samples enter these moments. A transition over d steps
is r**d times a cyclic shift plus a reset row. Expanding at the (at most
four) queried times gives independent blocks with deterministic shifts
inside each block. This computes the covariance contraction without a
K**4 count-covariance tensor.
"""
from itertools import product
import numpy as np


def topic_means(reset, repeat, length, offsets):
    k = len(reset)
    marginals = [reset]
    for _ in range(1, length):
        marginals.append(repeat*np.roll(marginals[-1], 1)+(1-repeat)*reset)
    result = {}
    for d in sorted({abs(d) for d in offsets}):
        u = np.sum(marginals[:length-d], axis=0)
        w = sum(((1-repeat)*repeat**j*np.roll(reset, j) for j in range(d)),
                np.zeros(k))
        value = np.outer(u, w)
        value[np.arange(k), (np.arange(k)+d) % k] += repeat**d*u
        if d in offsets:
            result[d] = value
        if -d in offsets:
            result[-d] = value.T
    return result


def event_inner(reset, repeat, length, times, contexts, centers):
    """E[I(X_a=u) I(X_c=v) (e_Xb-p) dot (e_Xd-q)].

    Batch axis indexes evaluation queries (and optionally target sizes).
    All repeated times and conflicting categorical constraints are retained.
    """
    k = len(reset)
    a, b, c, d = times
    u, v = contexts
    p, q = centers
    n = len(u)
    row = np.arange(n)
    ordered = sorted(set(times))
    marginal = reset.copy()
    for _ in range(ordered[0]):
        marginal = repeat*np.roll(marginal, 1)+(1-repeat)*reset
    pq = np.einsum('ij,ij->i', p, q)
    result = np.zeros(n)
    gaps = np.diff(ordered)
    reset_rows = {int(g): sum(((1-repeat)*repeat**j*np.roll(reset, j)
                              for j in range(g)), np.zeros(k)) for g in gaps}
    for branches in product((0, 1), repeat=len(gaps)):
        # 0: deterministic shift. 1: reset, starting an independent block.
        coefficient = 1.
        distributions = [marginal]
        starts = [ordered[0]]
        node = {ordered[0]: (0, 0)}
        for time, gap, branch in zip(ordered[1:], gaps, branches):
            if branch:
                distributions.append(reset_rows[int(gap)])
                starts.append(time)
            else:
                coefficient *= repeat**int(gap)
            node[time] = (len(starts)-1, time-starts[-1])
        if coefficient == 0:
            continue
        fixed = {}
        valid = np.ones(n, dtype=bool)
        for time, value in ((a, u), (c, v)):
            block, shift = node[time]
            anchor = (value-shift) % k
            if block in fixed:
                valid &= fixed[block] == anchor
            else:
                fixed[block] = anchor
        masses = [dist[fixed[h]] if h in fixed else np.full(n, dist.sum())
                  for h, dist in enumerate(distributions)]
        def outside(*excluded):
            value = coefficient*valid.astype(float)
            for h, mass in enumerate(masses):
                if h not in excluded:
                    value = value*mass
            return value
        mass = outside()
        def weighted(time, center):
            block, shift = node[time]
            if block in fixed:
                return mass*center[row, (fixed[block]+shift) % k]
            return outside(block)*(center @ np.roll(distributions[block], shift))
        hb, sb = node[b]
        hd, sd = node[d]
        if hb == hd:
            equal = mass if (sb-sd) % k == 0 else np.zeros(n)
        elif hb in fixed and hd in fixed:
            equal = mass*((fixed[hb]+sb-fixed[hd]-sd) % k == 0)
        elif hb in fixed:
            equal = outside(hd)*distributions[hd][(fixed[hb]+sb-sd) % k]
        elif hd in fixed:
            equal = outside(hb)*distributions[hb][(fixed[hd]+sd-sb) % k]
        else:
            equal = outside(hb, hd)*np.dot(np.roll(distributions[hb], sb),
                                          np.roll(distributions[hd], sd))
        result += equal-weighted(b, q)-weighted(d, p)+mass*pq
    return result


def exact_forecasts(population, contexts, weights, prior, offsets, sizes,
                    smoothing, chunks_per_source, progress=None):
    """Exact population source and multinomial laws for fixed evaluation queries.

    Each source contains c chunks, independent conditional on a shared topic.
    For one-chunk influence Y, source variance per chunk is
    E||Y||^2 + (c-1)E||E[Y|H]||^2 - c||E Y||^2.
    Multiplication by 2*n gives independent A/B linearized disagreement.
    """
    contexts = np.asarray(contexts, dtype=int)
    weights = np.asarray(weights, dtype=float)
    points, lags = weights.shape
    assert contexts.shape == weights.shape and lags == len(offsets)
    k, length = population.classes, population.length
    means = {d: x.toarray() for d, x in population.mean_lag_counts(offsets).items()}
    centers, coefficients = [], []
    multi = []
    for n in sizes:
        center, coeff = [], []
        multinomial = np.zeros(points)
        for j, delta in enumerate(offsets):
            m = means[delta][contexts[:, j]]
            rho = m.sum(axis=1)
            denominator = n*rho+smoothing
            center.append((n*m+smoothing*prior)/denominator[:, None])
            coeff.append(weights[:, j]/denominator)
            conditional = np.divide(m, rho[:, None], out=np.zeros_like(m),
                                    where=rho[:, None] > 0)
            multinomial += 2*n*rho/denominator**2*weights[:, j]**2*(
                1-np.einsum('ij,ij->i', conditional, conditional))
        centers.append(center)
        coefficients.append(np.asarray(coeff).T)
        multi.append(multinomial)
    centers = np.asarray(centers)  # sizes, lags, points, K
    coefficients = np.asarray(coefficients)  # sizes, points, lags
    second = np.zeros((len(sizes), points))
    mean = np.zeros((len(sizes), points, k))
    topic_mean_norm = np.zeros_like(second)
    resets = population.reset_distributions()
    for h, reset in enumerate(resets):
        topic = topic_means(reset, population.repeat, length, offsets)
        mu = np.zeros_like(mean)
        for j, delta in enumerate(offsets):
            m = topic[delta][contexts[:, j]]
            mu += coefficients[:, :, j, None]*(m[None]-m.sum(axis=1)[None, :, None]*centers[:, j])
        mean += mu/len(resets)
        topic_mean_norm += np.einsum('spk,spk->sp', mu, mu)/len(resets)
        for j, dj in enumerate(offsets):
            for l in range(j+1):
                dl = offsets[l]
                selected = np.flatnonzero((weights[:, j] != 0) & (weights[:, l] != 0))
                if not len(selected):
                    continue
                u = np.tile(contexts[selected, j], len(sizes))
                v = np.tile(contexts[selected, l], len(sizes))
                p = centers[:, j, selected].reshape(-1, k)
                q = centers[:, l, selected].reshape(-1, k)
                inner = np.zeros(len(u))
                for a in range(max(0, -dj), min(length, length-dj)):
                    for c in range(max(0, -dl), min(length, length-dl)):
                        inner += event_inner(reset, population.repeat, length,
                                             (a, a+dj, c, c+dl), (u, v), (p, q))
                second[:, selected] += (1 if j == l else 2)*inner.reshape(len(sizes), -1)*(
                    coefficients[:, selected, j]*coefficients[:, selected, l])/len(resets)
        if progress:
            progress(h+1, len(resets))
    c = chunks_per_source
    variance = second+(c-1)*topic_mean_norm-c*np.einsum('spk,spk->sp', mean, mean)
    if variance.min(initial=0) < -1e-10:
        raise ArithmeticError('Negative exact source variance')
    source = 2*np.asarray(sizes)[:, None]*np.maximum(variance, 0)
    return dict(source=source, multinomial=np.asarray(multi), chunk_second=second,
                topic_mean_norm=topic_mean_norm, mean_norm=np.einsum('spk,spk->sp', mean, mean))
