import itertools

import numpy as np

from diffusion_lm_rmt.population_calibration import SparsePopulation
from diffusion_lm_rmt.categorical_lag import offset_list
from diffusion_lm_rmt.sparse_categorical_lag import pair_counts


def make_population(repeat=.25):
    return SparsePopulation(dict(classes=3, chunk_length=4, repeat_rate=repeat,
        topic_weight=.55, topic_size=2, topics=2, zipf_exponent=1.05), np.random.default_rng(31))


def test_population_lag_means_match_exhaustive_enumeration():
    population = make_population()
    offsets = offset_list(2)
    expected = {d: np.zeros((3, 3)) for d in offsets}
    for v in population.reset_distributions():
        transition = .75*np.repeat(v[None, :], 3, axis=0)
        transition[np.arange(3), (np.arange(3)+1) % 3] += .25
        for sequence in itertools.product(range(3), repeat=4):
            probability = v[sequence[0]] / len(population.topics)
            for a, b in zip(sequence[:-1], sequence[1:]):
                probability *= transition[a, b]
            counts = pair_counts(np.asarray([sequence]), offsets, 3)
            for d in offsets:
                expected[d] += probability*counts[d].toarray()
    for d, matrix in population.mean_lag_counts(offsets).items():
        np.testing.assert_allclose(matrix.toarray(), expected[d], rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(matrix.sum(), 4-abs(d))


def test_population_sampling_matches_finite_position_means():
    population = make_population(.8)
    offsets = offset_list(2)
    counts = pair_counts(population.sample(100000, np.random.default_rng(44)), offsets, 3)
    for d, matrix in population.mean_lag_counts(offsets).items():
        np.testing.assert_allclose(counts[d].toarray()/100000, matrix.toarray(), atol=.009, rtol=0)


def test_no_successor_limit_is_topic_conditionally_iid():
    population = make_population(0)
    distributions = population.reset_distributions()
    second_moment = distributions.T @ distributions / len(distributions)
    for d, matrix in population.mean_lag_counts(offset_list(2)).items():
        np.testing.assert_allclose(matrix.toarray(), (4-abs(d))*second_moment)
