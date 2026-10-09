"""Source-split jackknife diagnostics for the finite-reference forecast.

This correction uses reference data only. It retains complete sources and
all lag terms in each component forecast. It does not impute unseen rows.
"""
from __future__ import annotations

import numpy as np

from .reference_ratio import reference_forecasts


def source_split_jackknife(reference, groups, evaluation, mask, prior, offsets,
                           gram, cross, sizes, *, smoothing, ridge, seed):
    """Full-reference forecasts and a two-group jackknife, before clipping.

    If E[F_S] = F_infinity + b/S + o(1/S), then
    2 F_S - sum_h (S_h/S) F_{S_h} removes the leading b/S term.
    S is the number of independent sources, not the number of chunks.
    This is a regular-regime argument, not a sparse-row guarantee.
    """
    if len(groups) < 4:
        raise ValueError("Each source half needs at least two sources")
    options = dict(smoothing=smoothing, ridge=ridge)
    full = reference_forecasts(reference, groups, evaluation, mask, prior,
        offsets, gram, cross, sizes, **options)
    order = np.random.default_rng(seed).permutation(len(groups))
    halves = []
    fractions = []
    for chosen in np.array_split(order, 2):
        indices = np.concatenate([groups[i] for i in chosen])
        lengths = [len(groups[i]) for i in chosen]
        boundaries = np.cumsum([0, *lengths])
        local_groups = [np.arange(a, b) for a, b in zip(boundaries[:-1], boundaries[1:])]
        halves.append(reference_forecasts(reference[indices], local_groups,
            evaluation, mask, prior, offsets, gram, cross, sizes, **options))
        fractions.append(len(chosen)/len(groups))
    result = {}
    for n in sizes:
        value = dict(full[n])
        half = sum(weight * h[n]['point_forecasts'][2]
                   for weight, h in zip(fractions, halves))
        corrected = 2*full[n]['point_forecasts'][2] - half
        value['jackknife_raw'] = corrected
        # Positivity is imposed only on the final scalar variance forecast.
        # Raw point estimates are retained for diagnostics and may be negative.
        value['jackknife_mean_raw'] = float(corrected @ value['point_weight'])
        value['jackknife_mean'] = max(value['jackknife_mean_raw'], 0.)
        value['negative_point_fraction'] = float(np.mean(corrected < 0))
        value['half_source_fractions'] = np.asarray(fractions)
        result[n] = value
    return result


def sample_source_chunks(population, chunks, chunks_per_source, rng):
    """IID sources with a shared topic across their non-overlapping chunks."""
    if chunks % chunks_per_source:
        raise ValueError("Use a whole number of sources")
    topics = population.topics[rng.integers(len(population.topics),
                                           size=chunks//chunks_per_source)]
    topics = np.repeat(topics, chunks_per_source, axis=0)
    shape = (chunks, population.length)
    local = np.take_along_axis(topics, rng.integers(population.topic_size, size=shape), axis=1)
    global_ids = rng.choice(population.classes, size=shape, p=population.unigram)
    ids = np.where(rng.random(shape) < population.topic_weight, local, global_ids)
    advance = rng.random(shape) < population.repeat
    for position in range(1, population.length):
        ids[:, position] = np.where(advance[:, position],
            (ids[:, position-1]+1) % population.classes, ids[:, position])
    groups = [np.arange(i, i+chunks_per_source) for i in range(0, chunks, chunks_per_source)]
    return ids, groups


def crossed_interval(forecasts, observations, *, seed, draws=5000):
    """Intervals resampling banks and independent A/B pairs separately.

    forecasts is banks x methods. observations contains one score per A/B pair.
    The same bank indices and pair indices are used for every method.
    Evaluation and auxiliary data are held fixed.
    """
    forecasts, observations = np.asarray(forecasts), np.asarray(observations)
    if forecasts.ndim != 2 or len(forecasts) < 2 or len(observations) < 2:
        raise ValueError("Need multiple reference banks and independent pairs")
    if np.any(forecasts <= 0) or np.any(observations <= 0):
        raise ValueError("Log-error inference requires positive forecasts and observations")
    rng = np.random.default_rng(seed)
    banks = rng.integers(len(forecasts), size=(draws, len(forecasts)))
    pairs = rng.integers(len(observations), size=(draws, len(observations)))
    denominator = observations[pairs].mean(axis=1)
    numerator = forecasts[banks].mean(axis=1)
    ratios = numerator / denominator[:, None]
    # Calibration accuracy of each reference bank against mean population
    # disagreement. Banks are not treated as independent B*P observations.
    errors = np.abs(np.log(forecasts[banks] / denominator[:, None, None])).mean(axis=1)
    actual_errors = np.abs(np.log(forecasts/observations.mean())).mean(axis=0)
    return dict(ratio=forecasts.mean(axis=0)/observations.mean(),
        ratio_ci=np.quantile(ratios, [.025, .975], axis=0).T,
        mean_bank_absolute_log_error=actual_errors,
        error_ci=np.quantile(errors, [.025, .975], axis=0).T,
        paired_error_difference=actual_errors-actual_errors[0],
        paired_error_difference_ci=np.quantile(errors-errors[:, :1], [.025, .975], axis=0).T)
