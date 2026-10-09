import itertools
import numpy as np
from diffusion_lm_rmt.population_calibration import SparsePopulation
from diffusion_lm_rmt.population_oracle import exact_forecasts, event_inner
from diffusion_lm_rmt.sparse_categorical_lag import pair_counts


def enumerate_topic(reset, repeat, length):
    k = len(reset)
    transition = (1-repeat)*np.repeat(reset[None], k, axis=0)
    transition[np.arange(k), (np.arange(k)+1) % k] += repeat
    for sequence in itertools.product(range(k), repeat=length):
        probability = reset[sequence[0]]
        for a, b in zip(sequence[:-1], sequence[1:]):
            probability *= transition[a, b]
        yield np.asarray(sequence), probability


def test_four_time_contraction_matches_enumeration_with_repeated_times():
    rng = np.random.default_rng(38)
    reset = np.array([.5, .3, .2])
    u, v = np.array([0, 1, 2, 1]), np.array([2, 0, 1, 1])
    p, q = rng.dirichlet(np.ones(3), 4), rng.dirichlet(np.ones(3), 4)
    for times in [(0, 2, 1, 3), (2, 0, 3, 1), (1, 2, 1, 0),
                  (0, 1, 2, 1), (0, 3, 3, 0), (0, 1, 0, 1)]:
        expected = np.zeros(4)
        a, b, c, d = times
        for seq, probability in enumerate_topic(reset, .37, 4):
            dot = np.sum((np.eye(3)[seq[b]]-p)*(np.eye(3)[seq[d]]-q), axis=1)
            expected += probability*(seq[a] == u)*(seq[c] == v)*dot
        np.testing.assert_allclose(event_inner(reset, .37, 4, times, (u, v), (p, q)),
                                   expected, rtol=1e-12, atol=1e-14)


def test_exact_source_law_matches_enumerated_chunk_and_cluster_moments():
    population = SparsePopulation(dict(classes=3, chunk_length=4, repeat_rate=.4,
        topic_weight=.6, topic_size=2, topics=2, zipf_exponent=1.1), np.random.default_rng(4))
    offsets = (-2, -1, 1, 2)
    contexts = np.array([[0, 1, 2, 0], [2, 0, 1, 1], [1, 1, 0, 2]])
    weights = np.array([[.2, .3, -.1, .4], [0, .3, .2, 0], [0, 0, 0, 0]])
    prior = np.array([.2, .3, .5])
    sizes, alpha = [8, 20], 2.3
    means = {d: x.toarray() for d, x in population.mean_lag_counts(offsets).items()}
    for chunks_per_source in (1, 4):
        answer = exact_forecasts(population, contexts, weights, prior, offsets, sizes,
                                 alpha, chunks_per_source)
        expected = []
        for n in sizes:
            topic_mu, topic_second = [], []
            for reset in population.reset_distributions():
                mu, second = np.zeros((3, 3)), np.zeros(3)
                for seq, probability in enumerate_topic(reset, population.repeat, 4):
                    counts = pair_counts(seq[None], offsets, 3)
                    reward = np.zeros((3, 3))
                    for j, d in enumerate(offsets):
                        rows = means[d][contexts[:, j]]
                        denominator = n*rows.sum(axis=1)+alpha
                        center = (n*rows+alpha*prior)/denominator[:, None]
                        sample = counts[d].toarray()[contexts[:, j]]
                        reward += weights[:, j, None]*(sample-center*sample.sum(axis=1)[:, None])/denominator[:, None]
                    mu += probability*reward
                    second += probability*np.sum(reward**2, axis=1)
                topic_mu.append(mu)
                topic_second.append(second)
            c = chunks_per_source
            expected.append(2*n*(np.mean(topic_second, axis=0)
                +(c-1)*np.mean(np.sum(np.array(topic_mu)**2, axis=2), axis=0)
                -c*np.sum(np.mean(topic_mu, axis=0)**2, axis=1)))
        np.testing.assert_allclose(answer['source'], expected, rtol=1e-11, atol=1e-13)
