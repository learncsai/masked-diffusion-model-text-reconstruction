"""Fit ``D(n) = a*n^-alpha (+ b)`` to the neural dataset-size sweep.

This adapts ``rmt.fitting.fit_scaling_law`` to a setting where the classical
assumption behind it — many independently trained replicates per n — does
not hold: each n in this experiment affords exactly one trained cross-split
pair (retraining replicates is not affordable at this budget). Reusing
``fit_scaling_law`` unmodified would silently misuse its replicate-level
bootstrap as if it quantified across-training-run variance, when it would
really only reflect held-out-sequence sampling noise within the single
trained pair. This module makes that substitution explicit instead of
hiding it: it bootstraps over evaluation sequences (weighted by masked
tokens, matching ``evaluation.aggregate_with_bootstrap``) and labels the
result accordingly everywhere it is reported.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DatasetSizeFit:
    model: str
    metric: str
    amplitude: float
    exponent: float
    offset: float
    exponent_ci_low: float
    exponent_ci_high: float
    r_squared: float
    n_values: list[int]
    point_estimates: list[float]
    uncertainty_source: str = "sequence-level bootstrap within one trained pair per n; NOT independent-training-replicate variance"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _with_offset(n: FloatArray, amplitude: float, exponent: float, offset: float) -> FloatArray:
    return amplitude * n ** (-exponent) + offset


def _no_offset(n: FloatArray, amplitude: float, exponent: float) -> FloatArray:
    return amplitude * n ** (-exponent)


def _fit_means(n: FloatArray, y: FloatArray, with_offset: bool) -> tuple[np.ndarray, float]:
    if with_offset:
        initial = [max(float(y[0] * n[0]), 1e-12), 1.0, max(float(y[-1] * 0.05), 0.0)]
        parameters, _ = curve_fit(_with_offset, n, y, p0=initial, bounds=([0.0, 0.0, 0.0], [np.inf, 5.0, np.inf]), maxfev=20_000)
        predicted = _with_offset(n, *parameters)
    else:
        parameters, _ = curve_fit(_no_offset, n, y, p0=[max(float(y[0] * n[0]), 1e-12), 1.0], bounds=([0.0, 0.0], [np.inf, 5.0]), maxfev=20_000)
        predicted = _no_offset(n, *parameters)
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return parameters, 1.0 - residual / total if total > 0 else 1.0


def load_per_sequence_pools(per_sequence_csv: Path, metric: str, pair_prefix: str) -> dict[int, tuple[FloatArray, FloatArray]]:
    """Group one pair's per-sequence rows into (values, masked_token_weights) by n.

    Expects pair names of the form ``f"{pair_prefix}{n}"`` as written by
    ``dataset_size_evaluation.run_dataset_size_evaluation``.
    """
    rows = list(csv.DictReader(per_sequence_csv.open()))
    pools: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        pair = row["pair"]
        if not pair.startswith(pair_prefix):
            continue
        n = int(pair[len(pair_prefix):])
        pools[n].append((float(row[metric]), float(row["masked_tokens"])))
    return {n: (np.asarray([v for v, _ in pool]), np.asarray([w for _, w in pool])) for n, pool in pools.items()}


def fit_dataset_size_curve(
    pools: dict[int, tuple[FloatArray, FloatArray]],
    *,
    metric: str,
    with_offset: bool,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> DatasetSizeFit:
    """Fit the power law with weighted-sequence bootstrap exponent uncertainty."""
    n_values = sorted(pools)
    if len(n_values) < 3:
        raise ValueError("at least three n values are required for a power-law fit")
    n = np.asarray(n_values, dtype=np.float64)
    point_estimates = np.asarray([np.average(pools[value][0], weights=pools[value][1]) for value in n_values])
    parameters, r_squared = _fit_means(n, point_estimates, with_offset)

    rng = np.random.default_rng(seed)
    exponents: list[float] = []
    for _ in range(bootstrap_samples):
        boot_means = np.empty(n.size, dtype=np.float64)
        for row, value in enumerate(n_values):
            values, weights = pools[value]
            draw = rng.integers(0, values.size, values.size)
            boot_means[row] = np.average(values[draw], weights=weights[draw])
        try:
            boot_parameters, _ = _fit_means(n, boot_means, with_offset)
            exponents.append(float(boot_parameters[1]))
        except (RuntimeError, ValueError):
            continue
    low, high = np.quantile(exponents, [0.025, 0.975]) if exponents else (float("nan"), float("nan"))
    offset = float(parameters[2]) if with_offset else 0.0
    return DatasetSizeFit(
        model="a*n^-alpha+b" if with_offset else "a*n^-alpha", metric=metric,
        amplitude=float(parameters[0]), exponent=float(parameters[1]), offset=offset,
        exponent_ci_low=float(low), exponent_ci_high=float(high), r_squared=float(r_squared),
        n_values=n_values, point_estimates=[float(v) for v in point_estimates],
    )


def hidden_disagreement_totals(
    spectral_bridge_csv: Path,
    *,
    state_variant: str = "per_token_layer_norm",
    comparison_type: str = "propagated_hidden",
) -> dict[tuple[int, int], float]:
    """Sum per-mode hidden disagreement into one D_op-analog scalar per (n, layer).

    No bootstrap is available for this series (the underlying per-token
    values are not persisted), so it is reported as a point-estimate-only
    secondary series -- explicitly not power-law fit with a quantified CI.
    """
    rows = list(csv.DictReader(spectral_bridge_csv.open()))
    totals: dict[tuple[int, int], float] = defaultdict(float)
    for row in rows:
        if row.get("state_variant", "raw") != state_variant:
            continue
        if row.get("comparison_type", "propagated_hidden") != comparison_type:
            continue
        key = (int(row["training_chunks"]), int(row["layer"]))
        totals[key] += float(row["hidden_disagreement"])
    return dict(totals)


def fit_equal_mask_grid_curve(
    aggregate_csv: Path,
    *,
    metric: str = "js_divergence",
    pair_prefix: str = "cross_split_same_init_n",
    valid_n: tuple[int, ...] = (1024, 2048, 4096),
) -> dict[str, object]:
    """Fit a no-offset curve to equal-weight mask-fraction means.

    This estimand gives each configured mask probability one vote.  It is
    distinct from pooling per-sequence rows with masked-token weights, which
    implicitly gives high-mask conditions more influence.  With one trained
    pair per n, the returned exponent is descriptive and has no training-run
    confidence interval.
    """
    grouped: dict[int, list[float]] = defaultdict(list)
    fractions: dict[int, set[float]] = defaultdict(set)
    for row in csv.DictReader(aggregate_csv.open()):
        pair = row["pair"]
        if row["metric"] != metric or not pair.startswith(pair_prefix):
            continue
        n = int(pair[len(pair_prefix):])
        grouped[n].append(float(row["mean"]))
        fractions[n].add(float(row["mask_fraction"]))
    if any(n not in grouped for n in valid_n):
        missing = [n for n in valid_n if n not in grouped]
        raise ValueError(f"missing aggregate mask-grid rows for n={missing}")
    grid_sizes = {len(fractions[n]) for n in grouped}
    if len(grid_sizes) != 1:
        raise ValueError("all n values must use the same number of mask fractions")
    point_estimates = {n: float(np.mean(grouped[n])) for n in sorted(grouped)}
    n_array = np.asarray(valid_n, dtype=np.float64)
    y = np.asarray([point_estimates[n] for n in valid_n], dtype=np.float64)
    parameters, r_squared = _fit_means(n_array, y, with_offset=False)
    return {
        "metric": metric,
        "aggregation": "equal weight over mask fractions; each mask-fraction mean is token-weighted within that fraction",
        "mask_fractions_per_n": grid_sizes.pop(),
        "point_estimates": {str(n): value for n, value in point_estimates.items()},
        "valid_n": list(valid_n),
        "model": "a*n^-alpha",
        "amplitude": float(parameters[0]),
        "exponent": float(parameters[1]),
        "r_squared": float(r_squared),
        "uncertainty": "descriptive point estimate only; one trained pair per n",
    }
