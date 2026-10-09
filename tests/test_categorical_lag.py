"""Checks for the categorical lag estimator and its source-level derivative."""

import numpy as np

from diffusion_lm_rmt.categorical_lag import (
    conditional_directional,
    conditional_tables,
    masked_prediction,
    offset_list,
    pair_counts,
    per_sequence_squared_error,
    source_influence_disagreement,
)


def test_conditional_count_directional_matches_finite_difference() -> None:
    rng = np.random.default_rng(8)
    prior = np.array([0.2, 0.3, 0.5])
    counts = {1: rng.uniform(0.5, 10, size=(3, 3))}
    direction = {1: rng.normal(size=(3, 3))}
    epsilon = 1e-6
    numerical = (
        conditional_tables({1: counts[1] + epsilon * direction[1]}, prior, 2.0)[1]
        - conditional_tables(counts, prior, 2.0)[1]
    ) / epsilon
    derivative = conditional_directional(counts, direction, prior, 2.0)[1]
    np.testing.assert_allclose(numerical, derivative, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(derivative.sum(axis=1), 0, atol=1e-12)


def test_masked_output_and_sequence_normalization() -> None:
    ids = np.array([[0, 1, 0], [1, 1, 0]])
    mask = np.array([[True, False, True], [False, True, False]])
    prior = np.array([0.5, 0.5])
    offsets = offset_list(1)
    tables = conditional_tables(pair_counts(ids, offsets, 2), prior, 1.0)
    rows, targets, prediction = masked_prediction(
        ids, mask, tables, prior, offsets,
        np.eye(2), np.ones(2), 0.1,
    )
    assert len(prediction) == 3
    np.testing.assert_array_equal(targets, np.array([0, 0, 1]))
    np.testing.assert_allclose(prediction.sum(axis=1), 1)
    np.testing.assert_allclose(
        per_sequence_squared_error(rows, np.array([1.0, 3.0, 2.0]), 2),
        np.array([2.0, 2.0]),
    )


def test_document_clustering_increases_variance_for_shared_topic() -> None:
    rng = np.random.default_rng(42)
    topics = rng.integers(0, 2, size=80)
    reference = np.repeat(topics, 4 * 6).reshape(320, 6)
    document_groups = [np.arange(i * 4, (i + 1) * 4) for i in range(80)]
    chunk_groups = [np.array([i]) for i in range(320)]
    eval_ids = np.tile(np.array([0, 0, 1, 1, 0, 1]), (12, 1))
    mask = np.tile(np.array([False, True, False, True, False, True]), (12, 1))
    offsets = offset_list(1)
    prior = np.array([0.5, 0.5])
    gram = np.eye(2)
    cross = np.array([0.5, 0.5])
    document = source_influence_disagreement(
        reference, document_groups, eval_ids, mask, prior, offsets, gram, cross,
        n=80, smoothing=1.0, ridge=0.1,
    )
    chunk = source_influence_disagreement(
        reference, chunk_groups, eval_ids, mask, prior, offsets, gram, cross,
        n=80, smoothing=1.0, ridge=0.1,
    )
    assert np.mean(document) > 2 * np.mean(chunk) > 0


def test_global_masked_coefficient_mismatch() -> None:
    correlation = 0.5
    mask_rate = 0.2
    global_coefficient = (
        mask_rate * correlation
        / (1 - (1 - mask_rate) ** 2 * correlation**2)
    )
    np.testing.assert_allclose(global_coefficient, 0.119047619047619)
    assert global_coefficient < correlation


def test_equal_second_moments_do_not_fix_lag_zero_sampling_variance() -> None:
    rng = np.random.default_rng(101)
    gaussian = rng.normal(size=(400, 512))
    signs = rng.choice(np.array([-1.0, 1.0]), size=gaussian.shape)
    # Both have population variance one; their fourth moments differ.
    gaussian_lag_zero = np.mean(gaussian**2, axis=1)
    sign_lag_zero = np.mean(signs**2, axis=1)
    assert 0.002 < np.var(gaussian_lag_zero, ddof=1) < 0.006
    np.testing.assert_allclose(np.var(sign_lag_zero), 0)
