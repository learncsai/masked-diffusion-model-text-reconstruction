"""A tractable sparse population with exact mean lag counts.

This control is not a language model. A chunk is one independent source.
Its latent topic fixes a reset distribution v. At each step the token is
advanced modulo K with probability r and reset to v otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


class SparsePopulation:
    def __init__(self, spec, rng):
        self.classes = int(spec["classes"])
        self.length = int(spec["chunk_length"])
        self.repeat = float(spec["repeat_rate"])
        self.topic_weight = float(spec["topic_weight"])
        self.topic_size = int(spec["topic_size"])
        probabilities = np.arange(1, self.classes+1, dtype=float) ** -float(spec["zipf_exponent"])
        self.unigram = probabilities / probabilities.sum()
        self.topics = np.asarray([rng.choice(self.classes, self.topic_size, replace=False,
            p=self.unigram) for _ in range(int(spec["topics"]))])

    def reset_distributions(self):
        v = np.repeat(((1-self.topic_weight) * self.unigram)[None, :], len(self.topics), axis=0)
        for row, topic in zip(v, self.topics):
            row[topic] += self.topic_weight / self.topic_size
        return v

    def sample(self, chunks, rng):
        topic = self.topics[rng.integers(len(self.topics), size=chunks)]
        positions = rng.integers(self.topic_size, size=(chunks, self.length))
        local = np.take_along_axis(topic, positions, axis=1)
        global_ids = rng.choice(self.classes, size=local.shape, p=self.unigram)
        ids = np.where(rng.random(local.shape) < self.topic_weight, local, global_ids)
        advance = rng.random(local.shape) < self.repeat
        for position in range(1, self.length):
            ids[:, position] = np.where(advance[:, position],
                (ids[:, position-1]+1) % self.classes, ids[:, position])
        return ids

    def mean_lag_counts(self, offsets):
        """Expected raw lag counts per chunk, with finite-position boundaries.

        For a topic, P^d = r^d S^d + 1 w_d^T, where S advances a token
        and w_d=(1-r) sum_{j=0}^{d-1} r^j v S^j. Summing the position
        marginals gives diag(u_d) P^d. No stationary approximation is used.
        """
        positive = sorted({abs(d) for d in offsets})
        result = {d: np.zeros((self.classes, self.classes)) for d in positive}
        for reset in self.reset_distributions():
            marginals = [reset]
            for _ in range(1, self.length):
                marginals.append(self.repeat * np.roll(marginals[-1], 1)
                                 + (1-self.repeat) * reset)
            for delta in positive:
                u = np.sum(marginals[:self.length-delta], axis=0)
                w = sum(((1-self.repeat) * self.repeat**j * np.roll(reset, j)
                         for j in range(delta)), np.zeros(self.classes))
                result[delta] += np.outer(u, w)
                row = np.arange(self.classes)
                result[delta][row, (row+delta) % self.classes] += self.repeat**delta * u
        result = {d: matrix / len(self.topics) for d, matrix in result.items()}
        return {d: sparse.csr_matrix(result[d] if d > 0 else result[-d].T) for d in offsets}
