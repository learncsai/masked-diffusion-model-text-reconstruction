"""Categorical offset-tied reconstruction and prospective resampling tools."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse


def offset_list(radius: int) -> tuple[int, ...]:
    if radius < 1:
        raise ValueError("radius must be positive")
    return tuple(d for d in range(-radius, radius + 1) if d)


def token_counts(ids: np.ndarray, classes: int) -> np.ndarray:
    return np.bincount(np.asarray(ids).ravel(), minlength=classes).astype(np.float64)


def pair_counts(ids: np.ndarray, offsets: tuple[int, ...], classes: int) -> dict[int, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 2 or np.any((ids < 0) | (ids >= classes)):
        raise ValueError("ids must be a two-dimensional array of valid class IDs")
    result = {}
    for d in offsets:
        if abs(d) >= ids.shape[1]:
            raise ValueError("lag exceeds sequence length")
        if d > 0:
            context, target = ids[:, :-d], ids[:, d:]
        else:
            context, target = ids[:, -d:], ids[:, :d]
        result[d] = np.bincount(
            (context * classes + target).ravel(), minlength=classes * classes,
        ).reshape(classes, classes).astype(np.float64)
    return result


def conditional_tables(
    counts: dict[int, np.ndarray], prior: np.ndarray, smoothing: float,
) -> dict[int, np.ndarray]:
    prior = np.asarray(prior, dtype=np.float64)
    if smoothing <= 0 or np.any(prior < 0) or not np.isclose(prior.sum(), 1):
        raise ValueError("positive smoothing and a probability prior are required")
    return {
        d: (c + smoothing * prior[None, :]) / (c.sum(axis=1)[:, None] + smoothing)
        for d, c in counts.items()
    }


def fitting_equations(
    fitting_ids: np.ndarray,
    validation_ids: np.ndarray,
    prior: np.ndarray,
    offsets: tuple[int, ...],
    smoothing: float,
    mask_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact one-hot least-squares equations on held-out auxiliary documents."""
    classes = len(prior)
    tables = conditional_tables(pair_counts(fitting_ids, offsets, classes), prior, smoothing)
    mask = rng.random(validation_ids.shape) < mask_rate
    rows, positions = np.nonzero(mask)
    gram = np.zeros((len(offsets), len(offsets)), dtype=np.float64)
    cross = np.zeros(len(offsets), dtype=np.float64)
    for row, position in zip(rows, positions):
        usable = [j for j, d in enumerate(offsets)
                  if 0 <= position - d < validation_ids.shape[1]
                  and not mask[row, position - d]]
        if not usable:
            continue
        design = np.stack([
            tables[offsets[j]][validation_ids[row, position - offsets[j]]] - prior
            for j in usable
        ])
        block = np.ix_(usable, usable)
        gram[block] += design @ design.T
        cross[usable] += design[:, validation_ids[row, position]] - design @ prior
    return gram, cross


def pattern_weights(
    gram: np.ndarray, cross: np.ndarray, available: tuple[int, ...], ridge: float,
) -> np.ndarray:
    if not available:
        return np.empty(0)
    block = gram[np.ix_(available, available)]
    penalty = ridge * max(float(np.trace(block)) / len(available), 1e-12)
    return np.linalg.solve(block + penalty * np.eye(len(available)), cross[list(available)])


def masked_prediction(
    ids: np.ndarray,
    mask: np.ndarray,
    tables: dict[int, np.ndarray],
    prior: np.ndarray,
    offsets: tuple[int, ...],
    gram: np.ndarray,
    cross: np.ndarray,
    ridge: float,
    weight_cache: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rows, targets, and class vectors at hidden positions only."""
    if ids.shape != mask.shape:
        raise ValueError("ids and mask shape mismatch")
    rows, positions = np.nonzero(mask)
    target = ids[rows, positions]
    prediction = np.tile(prior, (len(rows), 1))
    patterns = np.zeros(len(rows), dtype=np.int32)
    for j, d in enumerate(offsets):
        context_position = positions - d
        inside = (context_position >= 0) & (context_position < ids.shape[1])
        visible = np.zeros(len(rows), dtype=bool)
        visible[inside] = ~mask[rows[inside], context_position[inside]]
        patterns |= visible.astype(np.int32) << j
    if weight_cache is None:
        weight_cache = {}
    for pattern in np.unique(patterns):
        if pattern == 0:
            continue
        selection = np.flatnonzero(patterns == pattern)
        usable = tuple(j for j in range(len(offsets)) if pattern & (1 << j))
        code = int(pattern)
        if code not in weight_cache:
            weight_cache[code] = pattern_weights(gram, cross, usable, ridge)
        weights = weight_cache[code]
        for weight, j in zip(weights, usable):
            context = ids[rows[selection], positions[selection] - offsets[j]]
            prediction[selection] += weight * (tables[offsets[j]][context] - prior)
    return rows, target, prediction


def per_sequence_squared_error(
    rows: np.ndarray, values: np.ndarray, sequences: int,
) -> np.ndarray:
    totals = np.bincount(rows, weights=values, minlength=sequences)
    counts = np.bincount(rows, minlength=sequences)
    return totals / np.maximum(counts, 1)


def source_groups(records: list[dict]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row, record in enumerate(records):
        groups[record["document_id"]].append(row)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def draw_reference_chunks(
    ids: np.ndarray, groups: list[np.ndarray], n: int,
    rng: np.random.Generator, *, independent_chunks: bool = False,
) -> np.ndarray:
    if independent_chunks:
        return ids[rng.integers(len(ids), size=n)]
    picked: list[np.ndarray] = []
    length = 0
    while length < n:
        group = groups[int(rng.integers(len(groups)))]
        picked.append(group)
        length += len(group)
    return ids[np.concatenate(picked)[:n]]


def bootstrap_disagreement(
    reference_ids: np.ndarray,
    groups: list[np.ndarray],
    evaluation_ids: np.ndarray,
    mask: np.ndarray,
    prior: np.ndarray,
    offsets: tuple[int, ...],
    gram: np.ndarray,
    cross: np.ndarray,
    *, n: int,
    smoothing: float,
    ridge: float,
    draws: int,
    law: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict two-fit disagreement from an independent reference sample.

    For each draw, recompute the joint lag tables and output. The mean
    pairwise squared difference is twice the sample variance of one output.
    This is a document bootstrap plug-in law, not a finite-sample identity.
    """
    if draws < 2:
        raise ValueError("at least two bootstrap draws are needed")
    sequences = len(evaluation_ids)
    mean = None
    m2 = None
    inputs = []
    weight_cache: dict[int, np.ndarray] = {}
    for draw in range(draws):
        if law == "same_lag":
            counts = {
                d: pair_counts(
                    draw_reference_chunks(reference_ids, groups, n, rng), (d,), len(prior),
                )[d]
                for d in offsets
            }
        elif law in {"source_cluster", "independent_chunks"}:
            sample = draw_reference_chunks(
                reference_ids, groups, n, rng,
                independent_chunks=(law == "independent_chunks"),
            )
            counts = pair_counts(sample, offsets, len(prior))
        else:
            raise ValueError(f"unknown law {law}")
        tables = conditional_tables(counts, prior, smoothing)
        rows, _, output = masked_prediction(
            evaluation_ids, mask, tables, prior, offsets, gram, cross, ridge,
            weight_cache=weight_cache,
        )
        if mean is None:
            mean = output.copy()
            m2 = np.zeros_like(output)
        else:
            delta = output - mean
            mean += delta / (draw + 1)
            m2 += delta * (output - mean)
        inputs.append(len(sample) if law != "same_lag" else n)
    assert mean is not None and m2 is not None
    point_variance = 2 * np.sum(m2 / (draws - 1), axis=1)
    per_sequence = per_sequence_squared_error(rows, point_variance, sequences)
    return per_sequence, np.asarray(inputs)


def conditional_directional(
    counts: dict[int, np.ndarray], direction: dict[int, np.ndarray],
    prior: np.ndarray, smoothing: float,
) -> dict[int, np.ndarray]:
    """Derivative of smoothed conditional rows for a fixed auxiliary prior."""
    tables = conditional_tables(counts, prior, smoothing)
    return {
        d: (direction[d] - tables[d] * direction[d].sum(axis=1)[:, None])
        / (counts[d].sum(axis=1)[:, None] + smoothing)
        for d in counts
    }


def source_influence_disagreement(
    reference_ids: np.ndarray,
    groups: list[np.ndarray],
    evaluation_ids: np.ndarray,
    mask: np.ndarray,
    prior: np.ndarray,
    offsets: tuple[int, ...],
    gram: np.ndarray,
    cross: np.ndarray,
    *, n: int, smoothing: float, ridge: float, same_lag: bool = False,
) -> np.ndarray:
    """First-order source-cluster trace for two future equal-size samples.

    The empirical reference source covariance uses ratio-estimator influences
    ``S_g - T_g S_ref/T_ref``. Here T_g is the number of chunks from source g.
    The target has n chunks and approximately n / mean(T_g) source draws.
    Auxiliary prior and stacking equations are conditioned on and fixed.
    """
    classes = len(prior)
    documents = len(groups)
    if documents < 2:
        raise ValueError("at least two reference sources are needed")
    reference_chunks = len(reference_ids)
    reference_counts = pair_counts(reference_ids, offsets, classes)
    scale = n / reference_chunks
    baseline_counts = {d: scale * c for d, c in reference_counts.items()}
    baseline_tables = conditional_tables(baseline_counts, prior, smoothing)
    denominators = {
        d: baseline_counts[d].sum(axis=1) + smoothing for d in offsets
    }

    rows, positions = np.nonzero(mask)
    locations = []
    columns = []
    values = []
    pattern = np.zeros(len(rows), dtype=np.int32)
    for j, d in enumerate(offsets):
        source = positions - d
        inside = (source >= 0) & (source < evaluation_ids.shape[1])
        visible = np.zeros(len(rows), dtype=bool)
        visible[inside] = ~mask[rows[inside], source[inside]]
        pattern |= visible.astype(np.int32) << j
    for code in np.unique(pattern):
        if code == 0:
            continue
        selected = np.flatnonzero(pattern == code)
        active = tuple(j for j in range(len(offsets)) if code & (1 << j))
        weights = pattern_weights(gram, cross, active, ridge)
        for weight, j in zip(weights, active):
            source = positions[selected] - offsets[j]
            context = evaluation_ids[rows[selected], source]
            locations.extend(selected.tolist())
            columns.extend((j * classes + context).tolist())
            values.extend(np.full(len(selected), weight).tolist())
    design = sparse.csr_matrix(
        (values, (locations, columns)), shape=(len(rows), len(offsets) * classes),
    )

    squared_sum = np.zeros(len(rows), dtype=np.float64)
    for group in groups:
        group_counts = pair_counts(reference_ids[group], offsets, classes)
        chunks = len(group)
        influence = np.empty((len(offsets) * classes, classes), dtype=np.float64)
        for j, d in enumerate(offsets):
            centered = group_counts[d] - (chunks / reference_chunks) * reference_counts[d]
            influence[j * classes:(j + 1) * classes] = (
                centered - baseline_tables[d] * centered.sum(axis=1)[:, None]
            ) / denominators[d][:, None]
        if same_lag:
            for j in range(len(offsets)):
                output = design[:, j * classes:(j + 1) * classes] @ (
                    influence[j * classes:(j + 1) * classes]
                )
                squared_sum += np.einsum("ij,ij->i", output, output)
        else:
            output = design @ influence
            squared_sum += np.einsum("ij,ij->i", output, output)
    target_documents = n * documents / reference_chunks
    point = 2 * target_documents * squared_sum / (documents - 1)
    return per_sequence_squared_error(rows, point, len(evaluation_ids))
