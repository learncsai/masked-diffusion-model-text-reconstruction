"""Independent checks of the multinomial baseline and dependence comparison."""
import math
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_lag_disagreement import multinomial_row_trace, multinomial_forecast, summarize
from diffusion_lm_rmt.sparse_categorical_influence import source_influence_disagreement


def test_multinomial_variance_matches_exact_enumeration_with_smoothing():
    counts = sparse.csr_matrix([[2., 4.], [0., 0.]])
    prior = np.array([.3, .7])
    alpha = 5.
    result = multinomial_row_trace(counts, prior, alpha, scale=.5)
    theta = (np.array([2., 4.]) + alpha * prior) / (6 + alpha)
    target_n = 3
    outcomes = np.array([(np.array([k, target_n-k]) + alpha*prior)/(target_n+alpha)
                         for k in range(target_n+1)])
    probabilities = np.array([math.comb(target_n, k)*theta[0]**k*theta[1]**(target_n-k)
                              for k in range(target_n+1)])
    expected = sum(probabilities[i]*probabilities[j]*np.sum((outcomes[i]-outcomes[j])**2)
                   for i in range(target_n+1) for j in range(target_n+1))
    np.testing.assert_allclose(result, [expected, 0], atol=1e-15)
    # The old omission of smoothing contributions must not pass this test.
    omitted_collision = (2**2 + 4**2)/(6+alpha)**2
    omitted = 2*target_n/(target_n+alpha)**2*(1-omitted_collision)
    assert abs(result[0]-omitted) > .01


def test_multinomial_forecast_matches_direct_pattern_calculation():
    reference = np.array([[0, 1, 0, 2], [1, 0, 1, 2], [0, 2, 1, 0]])
    evaluation = np.array([[0, 1, 2, 0], [2, 0, 1, 2]])
    mask = np.array([[False, True, False, True], [True, True, True, True]])
    offsets = (-1, 1)
    prior = np.array([.2, .5, .3])
    gram = np.array([[2., .2], [.2, 3.]])
    cross = np.array([.5, -.3])
    alpha, ridge, n = 2., .01, 6
    actual = multinomial_forecast(reference, evaluation, mask, prior, offsets, gram, cross,
                                  n=n, smoothing=alpha, ridge=ridge)
    tables = {}
    for d in offsets:
        counts = np.zeros((3, 3))
        for row in reference:
            for j in range(len(row)):
                if 0 <= j+d < len(row):
                    counts[row[j], row[j+d]] += 1
        tables[d] = counts
    point = []
    for i in (1, 3):
        active = [j for j, d in enumerate(offsets) if 0 <= i-d < 4 and not mask[0, i-d]]
        block = gram[np.ix_(active, active)]
        weights = np.linalg.solve(block + ridge*np.trace(block)/len(active)*np.eye(len(active)), cross[active])
        value = 0.
        for j, w in zip(active, weights):
            counts = tables[offsets[j]][evaluation[0, i-offsets[j]]]
            theta = (counts + alpha*prior)/(counts.sum()+alpha)
            target_n = n/len(reference)*counts.sum()
            value += w*w*2*target_n/(target_n+alpha)**2*(1-np.sum(theta**2))
        point.append(value)
    np.testing.assert_allclose(actual, [np.mean(point), 0], atol=1e-14)


def test_cross_lag_covariance_can_cancel_disagreement():
    reference = np.array([[0, 1, 0], [0, 2, 0], [0, 1, 0]])
    groups = [np.array([j]) for j in range(3)]
    evaluation = np.array([[0, 2, 0]])
    mask = np.array([[False, True, False]])
    args = (reference, groups, evaluation, mask, np.ones(3)/3, (-1, 1), np.eye(2), np.array([1., -1.]))
    same = source_influence_disagreement(*args, n=3, smoothing=2., ridge=.01, same_lag=True)
    full = source_influence_disagreement(*args, n=3, smoothing=2., ridge=.01, same_lag=False)
    assert same[0] > .01
    np.testing.assert_allclose(full, 0, atol=1e-12)


def test_summary_keeps_pair_log_errors_and_correlations_separate():
    forecasts = np.array([[1., 2., 3.], [2., 4., 6.], [1., 2., 3.]])
    observed = np.array([[1., 2., 3.], [3., 2., 1.], [2., 4., 6.]])
    pairs, rows = summarize('test', 512, forecasts, observed)
    assert len(pairs) == 9
    np.testing.assert_allclose(rows[0]['prediction_observation_ratio'], .75)
    np.testing.assert_allclose(rows[0]['mean_absolute_log_error'], np.log(2)/3)
    np.testing.assert_allclose(rows[0]['mean_pair_spearman'], 1/3)
