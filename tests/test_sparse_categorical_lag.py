"""Dense equivalence and full-vocabulary edge cases for the sparse backend."""

import numpy as np
import pytest

from diffusion_lm_rmt import categorical_lag as dense
from diffusion_lm_rmt import sparse_categorical_lag as sparse


@pytest.mark.parametrize("q", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_sparse_predictions_and_weight_equations_equal_dense(q):
    rng = np.random.default_rng(16)
    ids = rng.integers(0, 6, size=(25, 8))
    evaluation = rng.integers(0, 7, size=(9, 8))
    prior = np.array([.2, .1, .25, .15, .1, .2, 0.])
    offsets = dense.offset_list(2)
    dg, db = dense.fitting_equations(ids[:12], ids[12:], prior, offsets, 5, q,
                                     np.random.default_rng(41))
    sg, sb = sparse.fitting_equations(ids[:12], ids[12:], prior, offsets, 5, q,
                                      np.random.default_rng(41), batch_chunks=3)
    np.testing.assert_allclose(sg, dg, atol=1e-12)
    np.testing.assert_allclose(sb, db, atol=1e-12)
    dc = dense.pair_counts(ids, offsets, len(prior))
    sc = sparse.pair_counts(ids, offsets, len(prior))
    for d in offsets:
        np.testing.assert_array_equal(sc[d].toarray(), dc[d])
    mask = rng.random(evaluation.shape) < q
    dr, dt, dp = dense.masked_prediction(
        evaluation, mask, dense.conditional_tables(dc, prior, 5),
        prior, offsets, dg, db, .01)
    sr, st, sp = sparse.masked_prediction(
        evaluation, mask, sparse.conditional_tables(sc, prior, 5),
        prior, offsets, sg, sb, .01)
    np.testing.assert_array_equal(sr, dr)
    np.testing.assert_array_equal(st, dt)
    np.testing.assert_allclose(sp.toarray(), dp, atol=1e-12)
    expected = 1 - 2 * dp[np.arange(len(dt)), dt] + (dp**2).sum(axis=1)
    np.testing.assert_allclose(sp.loss(st), expected, atol=1e-12)
    np.testing.assert_allclose(sparse.squared_difference(sp, sp), 0, atol=1e-12)


@pytest.mark.parametrize("law", ["source_cluster", "independent_chunks", "same_lag"])
def test_sparse_bootstrap_same_draws_equal_dense(law):
    rng = np.random.default_rng(5)
    reference = rng.integers(0, 7, size=(24, 8))
    groups = [np.arange(i, i + 3) for i in range(0, 24, 3)]
    evaluation = rng.integers(0, 8, size=(6, 8))
    mask = rng.random(evaluation.shape) < .5
    prior = np.full(8, 1 / 8)
    offsets = dense.offset_list(2)
    args = (reference, groups, evaluation, mask, prior, offsets, np.eye(4), np.ones(4))
    kwargs = dict(n=13, smoothing=5, ridge=.01, draws=12, law=law)
    expected, _ = dense.bootstrap_disagreement(*args, **kwargs, rng=np.random.default_rng(8))
    actual, _ = sparse.bootstrap_disagreement(*args, **kwargs, rng=np.random.default_rng(8))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)


def test_full_vocabulary_unseen_rows_and_targets_are_not_dropped():
    classes = 50257
    ids = np.array([[0, 1, 2, 50256], [1, 2, 1, 0]])
    prior = np.zeros(classes)
    prior[:3] = [.5, .25, .25]
    offsets = dense.offset_list(1)
    counts = sparse.pair_counts(ids, offsets, classes)
    assert all(c.shape == (classes, classes) and c.nnz <= 6 for c in counts.values())
    evaluation = np.array([[10000, 40000, 10000]])
    mask = np.array([[False, True, False]])
    _, target, predicted = sparse.masked_prediction(
        evaluation, mask, sparse.conditional_tables(counts, prior, 5),
        prior, offsets, np.eye(2), np.ones(2), .01)
    assert target[0] == 40000
    assert predicted.residual.nnz == 0
    np.testing.assert_allclose(predicted.loss(target), 1 + np.dot(prior, prior))


@pytest.mark.parametrize("same_lag", [False, True])
@pytest.mark.parametrize("n", [12, 48])
def test_sparse_source_influence_equals_dense(same_lag, n):
    from diffusion_lm_rmt.sparse_categorical_influence import source_influence_disagreement
    rng = np.random.default_rng(102)
    reference = rng.integers(0, 7, size=(20, 8))
    groups = [np.arange(0, 2), np.arange(2, 7), np.arange(7, 11), np.arange(11, 20)]
    evaluation = rng.integers(0, 8, size=(5, 8))
    mask = rng.random(evaluation.shape) < .5
    prior = np.full(8, 1 / 8)
    offsets = dense.offset_list(2)
    args = (reference, groups, evaluation, mask, prior, offsets,
            np.eye(4), np.array([.1, .2, .3, .4]))
    kwargs = dict(n=n, smoothing=5, ridge=.01, same_lag=same_lag)
    expected = dense.source_influence_disagreement(*args, **kwargs)
    actual = source_influence_disagreement(*args, **kwargs)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("q", [.2, .8])
@pytest.mark.parametrize("n", [9, 120])
def test_source_influence_matches_independent_finite_difference(q, n):
    """Check the derivative against perturbed reconstructions, not another formula."""
    from diffusion_lm_rmt.sparse_categorical_influence import source_influence_disagreement
    rng = np.random.default_rng(921)
    reference = rng.integers(0, 5, size=(18, 8))
    groups = [np.arange(0, 2), np.arange(2, 6), np.arange(6, 11), np.arange(11, 18)]
    evaluation = rng.integers(0, 6, size=(7, 8))
    mask = rng.random(evaluation.shape) < q
    prior = np.full(6, 1 / 6)
    offsets = dense.offset_list(2)
    gram, cross = np.eye(4), np.array([.1, .2, .3, .4])
    counts = dense.pair_counts(reference, offsets, 6)
    baseline = {d: c * n / len(reference) for d, c in counts.items()}
    squared = np.zeros(mask.sum())
    epsilon = 1e-4
    for group in groups:
        source = dense.pair_counts(reference[group], offsets, 6)
        centered = {d: source[d] - len(group) / len(reference) * counts[d] for d in offsets}
        predictions = []
        for sign in (-1, 1):
            perturbed = {d: baseline[d] + sign * epsilon * centered[d] for d in offsets}
            rows, _, prediction = dense.masked_prediction(
                evaluation, mask, dense.conditional_tables(perturbed, prior, 5),
                prior, offsets, gram, cross, .01)
            predictions.append(prediction)
        derivative = (predictions[1] - predictions[0]) / (2 * epsilon)
        squared += np.sum(derivative**2, axis=1)
    expected = dense.per_sequence_squared_error(rows, squared, len(evaluation))
    expected *= 2 * n * len(groups) / len(reference) / (len(groups) - 1)
    actual = source_influence_disagreement(
        reference, groups, evaluation, mask, prior, offsets, gram, cross,
        n=n, smoothing=5, ridge=.01)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-12)
