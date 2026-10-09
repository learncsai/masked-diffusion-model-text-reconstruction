"""Quantitative comparison between the Phase I Gaussian/Fréchet prediction and
observed neural hidden-state cross-split disagreement.

This is the piece Phase I and the neural milestones stopped short of: instead
of only checking that hidden-state disagreement is *correlated* with the
reference-covariance eigenvalue (which any spectrum with a dominant top mode
would show), fit the actual closed-form resolvent-mask prediction from
``rmt.core.perturbative_split_prediction`` to the observed per-mode
disagreement and report observed/predicted ratios, the way Phase I's
``experiment_records.csv`` does for the synthetic study.

Approximations, stated explicitly because they are not derived from the
network architecture:

* The network is not a linear Gaussian denoiser, so there is no
  architecture-derived operator ``W`` and no reason for observed and
  predicted disagreement to share physical units. A single free positive
  scale ``kappa`` is fit (observed ~= kappa * predicted); everything above
  ratio/shape comparisons should be read through that lens.
* ``gamma = c(1-q)/q`` reuses the existing masking-bridge analogy already
  documented in ``docs/diffusion_llm_methodology.md`` (mask fraction 0.5,
  c = mean reference eigenvalue), not a derived Transformer quantity.
* ``n_a, n_b`` are each split's training-chunk count, matching this
  project's definition of ``n``, not literally independent Gaussian draws.
* Three reductions are deliberately kept separate: diagonal operator-entry
  variance, row-summed operator variance, and input-covariance-weighted
  prediction variance.  None is silently designated as the neural observable.
  The weighted form is the closest match to disagreement on actual shared
  inputs, but only after its input covariance and linearization assumptions
  have been measured.

Two tests are run, and they answer different questions:

1. **Within-spectrum shape fit** (``fit_group``): at one fixed n and layer,
   does the *curved* resolvent shape fit the observed per-mode disagreement
   better than the naive "disagreement is proportional to the raw
   eigenvalue" hypothesis? A free scale ``kappa`` is fit per group here,
   which means this test says nothing about whether the theory's
   ``(1/n_A+1/n_B)`` magnitude prefactor is correct across n -- kappa can
   absorb that prefactor completely if refit at every n.
2. **Cross-condition magnitude prediction, no recalibration**
   (``check_shared_calibration``): calibrate kappa on *one* reference
   condition only, then apply that same kappa to every other n with zero
   further fitting, and report how well it holds. This is the decisive test
   of whether the theory predicts the *amount* by which disagreement
   changes with n, not just its shape at each n taken separately.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .rmt.core import RMTObservable, perturbative_split_prediction, reduce_perturbative_prediction

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BridgeFit:
    layer: int
    state_variant: str
    comparison_type: str
    observable: RMTObservable
    modes_used: int
    gamma: float
    n_a: int
    n_b: int
    c_mean_eigenvalue: float
    kappa_fixed_slope: float
    r_squared_fixed_slope: float
    slope_free: float
    slope_free_ci_low: float
    slope_free_ci_high: float
    r_squared_free_slope: float
    log_correlation_resolvent_prediction: float
    log_correlation_raw_eigenvalue: float
    ratio_median: float
    ratio_p05: float
    ratio_p95: float
    uncertainty_unit: str
    heldout_mode_log_rmse: float
    raw_eigenvalue_heldout_mode_log_rmse: float
    linearization_r2: float
    minimum_linearization_r2: float
    linearization_gate_passed: bool

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _log_log_fit(x: FloatArray, y: FloatArray, *, fixed_slope: float | None) -> tuple[float, float, float]:
    """Fit ``log(y) = intercept + slope*log(x)``; return (slope, intercept, r_squared)."""
    log_x, log_y = np.log(x), np.log(y)
    if fixed_slope is not None:
        intercept = float(np.mean(log_y - fixed_slope * log_x))
        predicted = fixed_slope * log_x + intercept
        slope = fixed_slope
    else:
        slope, intercept = np.polyfit(log_x, log_y, 1)
        predicted = slope * log_x + intercept
    residual = log_y - predicted
    total = log_y - log_y.mean()
    r_squared = 1.0 - float(np.sum(residual**2)) / float(np.sum(total**2)) if np.sum(total**2) > 0 else float("nan")
    return float(slope), float(intercept), r_squared


def _twofold_fixed_shape_log_rmse(x: FloatArray, y: FloatArray) -> float:
    """Calibrate a scale on alternating modes and score the held-out modes."""
    residuals: list[FloatArray] = []
    indices = np.arange(x.size)
    for fold in (0, 1):
        train = indices % 2 != fold
        test = ~train
        if train.sum() == 0 or test.sum() == 0:
            continue
        intercept = float(np.mean(np.log(y[train]) - np.log(x[train])))
        residuals.append(np.log(y[test]) - (np.log(x[test]) + intercept))
    if not residuals:
        return float("nan")
    joined = np.concatenate(residuals)
    return float(np.sqrt(np.mean(joined**2)))


def _bootstrap_sequence_slope(
    x: FloatArray,
    sequence_disagreement: FloatArray,
    *,
    sequence_weights: FloatArray | None,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap held-out sequences, never ordered/correlated eigenmodes."""
    if sequence_disagreement.ndim != 2 or sequence_disagreement.shape[1] != x.size:
        raise ValueError("sequence disagreement must have shape (sequences, modes)")
    weights = (
        np.ones(sequence_disagreement.shape[0], dtype=np.float64)
        if sequence_weights is None else np.asarray(sequence_weights, dtype=np.float64)
    )
    if weights.shape != (sequence_disagreement.shape[0],) or (weights <= 0).any():
        raise ValueError("sequence weights must be positive with one value per sequence")
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(samples):
        draw = rng.integers(0, sequence_disagreement.shape[0], sequence_disagreement.shape[0])
        y = np.average(sequence_disagreement[draw], axis=0, weights=weights[draw])
        positive = y > 0
        if positive.sum() < 4:
            continue
        slope, _, _ = _log_log_fit(x[positive], y[positive], fixed_slope=None)
        slopes.append(slope)
    if not slopes:
        return float("nan"), float("nan")
    return float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))


def predicted_and_observed(
    eigenvalues: FloatArray,
    hidden_disagreement: FloatArray,
    *,
    n_a: int,
    n_b: int,
    mask_fraction: float,
    observable: RMTObservable = "rowsum_operator",
    input_variances_in_eigenbasis: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Compute one declared prediction and matching observed positive modes.

    Returns ``(eigenvalues, predicted, observed, gamma)`` restricted to modes
    with positive eigenvalue and positive observed disagreement.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    hidden_disagreement = np.asarray(hidden_disagreement, dtype=np.float64)
    if eigenvalues.ndim != 1 or hidden_disagreement.shape != eigenvalues.shape:
        raise ValueError("eigenvalues and hidden disagreement must be matching one-dimensional arrays")
    if not np.isfinite(eigenvalues).all() or (eigenvalues < 0).any():
        raise ValueError("eigenvalues must be finite and nonnegative")
    if not np.isfinite(hidden_disagreement).all() or (hidden_disagreement < 0).any():
        raise ValueError("hidden disagreement must be finite and nonnegative")
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must lie strictly between 0 and 1")
    q = 1.0 - mask_fraction
    c = float(eigenvalues.mean())
    if c <= 0:
        raise ValueError("mean eigenvalue must be positive")
    gamma = c * (1.0 - q) / q
    prediction = perturbative_split_prediction(eigenvalues, gamma, n_a, n_b)
    reduced = reduce_perturbative_prediction(
        prediction,
        observable=observable,
        input_variances_in_eigenbasis=input_variances_in_eigenbasis,
    )
    predicted_full = reduced.per_output_mode
    positive = (eigenvalues > 0) & (hidden_disagreement > 0) & (predicted_full > 0)
    if positive.sum() < 4:
        raise ValueError("fewer than 4 usable positive modes")
    return eigenvalues[positive], predicted_full[positive], hidden_disagreement[positive], gamma


def fit_group(
    eigenvalues: FloatArray,
    hidden_disagreement: FloatArray,
    *,
    layer: int,
    state_variant: str,
    n_a: int,
    n_b: int,
    mask_fraction: float,
    observable: RMTObservable = "rowsum_operator",
    input_variances_in_eigenbasis: FloatArray | None = None,
    comparison_type: str = "propagated_hidden",
    sequence_disagreement: FloatArray | None = None,
    sequence_weights: FloatArray | None = None,
    linearization_r2: float = float("nan"),
    minimum_linearization_r2: float = 0.5,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> tuple[BridgeFit, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Fit the Phase I closed-form prediction to one (layer, variant) group.

    Returns the fit summary plus the per-mode eigenvalue, predicted,
    observed, and observed/predicted ratio arrays (positive modes only), so
    a caller can also save a per-mode table.
    """
    try:
        values, predicted, observed, gamma = predicted_and_observed(
            eigenvalues, hidden_disagreement, n_a=n_a, n_b=n_b, mask_fraction=mask_fraction,
            observable=observable, input_variances_in_eigenbasis=input_variances_in_eigenbasis,
        )
    except ValueError as error:
        raise ValueError(f"layer {layer} ({state_variant}): {error}") from error
    c = float(np.asarray(eigenvalues, dtype=np.float64).mean())

    _, intercept_fixed, r_squared_fixed = _log_log_fit(predicted, observed, fixed_slope=1.0)
    kappa = float(np.exp(intercept_fixed))
    slope_free, _, r_squared_free = _log_log_fit(predicted, observed, fixed_slope=None)
    if sequence_disagreement is not None and bootstrap_samples > 0:
        full_theory = reduce_perturbative_prediction(
            perturbative_split_prediction(np.asarray(eigenvalues), gamma, n_a, n_b),
            observable=observable,
            input_variances_in_eigenbasis=(
                input_variances_in_eigenbasis if observable == "prediction_covariance_weighted" else None
            ),
        ).per_output_mode
        usable = (
            (np.asarray(eigenvalues) > 0)
            & (np.asarray(hidden_disagreement) > 0)
            & (full_theory > 0)
        )
        sequence_selected = np.asarray(sequence_disagreement, dtype=np.float64)[:, usable]
        ci_low, ci_high = _bootstrap_sequence_slope(
            predicted, sequence_selected, samples=bootstrap_samples, seed=seed,
            sequence_weights=sequence_weights,
        )
        uncertainty_unit = "held-out sequence bootstrap"
    else:
        ci_low = ci_high = float("nan")
        uncertainty_unit = "none; eigenmodes are not resampled"

    log_correlation_resolvent = float(np.corrcoef(np.log(predicted), np.log(observed))[0, 1])
    log_correlation_raw = float(np.corrcoef(np.log(values), np.log(observed))[0, 1])

    ratio = observed / (kappa * predicted)
    heldout_mode_log_rmse = _twofold_fixed_shape_log_rmse(predicted, observed)
    raw_heldout_mode_log_rmse = _twofold_fixed_shape_log_rmse(values, observed)
    fit = BridgeFit(
        layer=layer, state_variant=state_variant, comparison_type=comparison_type,
        observable=observable, modes_used=int(values.size), gamma=gamma,
        n_a=n_a, n_b=n_b, c_mean_eigenvalue=c, kappa_fixed_slope=kappa, r_squared_fixed_slope=r_squared_fixed,
        slope_free=slope_free, slope_free_ci_low=ci_low, slope_free_ci_high=ci_high, r_squared_free_slope=r_squared_free,
        log_correlation_resolvent_prediction=log_correlation_resolvent, log_correlation_raw_eigenvalue=log_correlation_raw,
        ratio_median=float(np.median(ratio)), ratio_p05=float(np.quantile(ratio, 0.05)), ratio_p95=float(np.quantile(ratio, 0.95)),
        uncertainty_unit=uncertainty_unit,
        heldout_mode_log_rmse=heldout_mode_log_rmse,
        raw_eigenvalue_heldout_mode_log_rmse=raw_heldout_mode_log_rmse,
        linearization_r2=float(linearization_r2),
        minimum_linearization_r2=float(minimum_linearization_r2),
        linearization_gate_passed=bool(np.isfinite(linearization_r2) and linearization_r2 >= minimum_linearization_r2),
    )
    return fit, values, predicted, observed, ratio


@dataclass(frozen=True)
class CrossConditionCheck:
    """One (reference -> checked) prediction with zero refitting at ``n_checked``."""

    layer: int
    state_variant: str
    comparison_type: str
    observable: RMTObservable
    reference_n: int
    kappa_from_reference: float
    n_checked: int
    is_reference: bool
    r_squared_no_recalibration: float
    ratio_median: float
    ratio_p05: float
    ratio_p95: float
    log_rmse_no_recalibration: float
    aggregate_observed_over_predicted: float
    calibration_method: str

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


def check_shared_calibration(
    predicted_by_n: dict[int, FloatArray],
    observed_by_n: dict[int, FloatArray],
    *,
    reference_n: int,
    layer: int,
    state_variant: str,
    observable: RMTObservable = "rowsum_operator",
    comparison_type: str = "propagated_hidden",
) -> list[CrossConditionCheck]:
    """Calibrate kappa on ``reference_n`` only, then apply it everywhere else unchanged.

    This is the test ``fit_group`` cannot do: since ``fit_group`` refits
    kappa independently at every n, a high within-n R^2 says nothing about
    whether the theory's ``(1/n_A+1/n_B)`` prefactor correctly predicts how
    the magnitude should change across n. Here kappa is estimated exactly
    once (least-squares log-intercept at fixed slope 1, on the reference
    condition) and then held fixed while checking every other n -- so the
    ``kappa`` is the reference aggregate observed sum divided by predicted
    sum, so the reference aggregate ratio is exactly one.  The only way to
    score well away from the reference is for the theory's own built-in
    n-dependence to do the work rather than a per-condition refit.
    """
    if reference_n not in predicted_by_n:
        raise ValueError(f"reference_n={reference_n} not present in predicted_by_n")
    if set(predicted_by_n) != set(observed_by_n):
        raise ValueError("predicted_by_n and observed_by_n must have identical n keys")
    ref_predicted, ref_observed = predicted_by_n[reference_n], observed_by_n[reference_n]
    if ref_predicted.shape != ref_observed.shape:
        raise ValueError("reference predicted and observed arrays must have matching shapes")
    # This test targets cross-condition magnitude.  Calibrating the aggregate
    # makes the reference aggregate ratio exactly one; within-spectrum shape
    # remains the job of fit_group's log-intercept and held-out-mode metrics.
    kappa = float(ref_observed.sum() / ref_predicted.sum())

    results: list[CrossConditionCheck] = []
    for n in sorted(predicted_by_n):
        predicted = kappa * predicted_by_n[n]
        observed = observed_by_n[n]
        if predicted.shape != observed.shape:
            raise ValueError(f"n={n}: predicted and observed arrays must have matching shapes")
        if (predicted <= 0).any() or (observed <= 0).any():
            raise ValueError(f"n={n}: predicted and observed values must be positive")
        log_residual = np.log(observed) - np.log(predicted)
        total = np.log(observed) - np.log(observed).mean()
        r_squared = 1.0 - float(np.sum(log_residual**2)) / float(np.sum(total**2)) if np.sum(total**2) > 0 else float("nan")
        ratio = observed / predicted
        results.append(CrossConditionCheck(
            layer=layer, state_variant=state_variant, comparison_type=comparison_type, observable=observable,
            reference_n=reference_n, kappa_from_reference=kappa,
            n_checked=n, is_reference=(n == reference_n), r_squared_no_recalibration=r_squared,
            ratio_median=float(np.median(ratio)), ratio_p05=float(np.quantile(ratio, 0.05)), ratio_p95=float(np.quantile(ratio, 0.95)),
            log_rmse_no_recalibration=float(np.sqrt(np.mean(log_residual**2))),
            aggregate_observed_over_predicted=float(observed.sum() / predicted.sum()),
            calibration_method="reference aggregate observed sum / predicted sum",
        ))
    return results


def _sequence_matrix(rows: list[dict[str, str]], modes: int) -> tuple[FloatArray, FloatArray] | None:
    """Return a sequence-by-mode matrix from persisted per-sequence rows."""
    if not rows:
        return None
    sequence_ids = sorted({int(row["sequence_index"]) for row in rows})
    lookup = {(int(row["sequence_index"]), int(row["mode"])): float(row["hidden_disagreement"]) for row in rows}
    token_lookup = {(int(row["sequence_index"]), int(row["mode"])): int(row.get("tokens", 1)) for row in rows}
    matrix = np.empty((len(sequence_ids), modes), dtype=np.float64)
    weights = np.empty(len(sequence_ids), dtype=np.float64)
    for sequence_row, sequence_id in enumerate(sequence_ids):
        sequence_token_counts: set[int] = set()
        for mode in range(modes):
            key = (sequence_id, mode)
            if key not in lookup:
                raise ValueError(f"missing per-sequence disagreement for sequence={sequence_id}, mode={mode}")
            matrix[sequence_row, mode] = lookup[key]
            sequence_token_counts.add(token_lookup[key])
        if len(sequence_token_counts) != 1:
            raise ValueError(f"inconsistent token counts for sequence={sequence_id}")
        weights[sequence_row] = float(sequence_token_counts.pop())
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("per-sequence disagreement must be finite and nonnegative")
    return matrix, weights


def run_bridge(
    spectral_bridge_csv: Path,
    *,
    n_a: int,
    n_b: int,
    mask_fraction: float = 0.5,
    output_dir: Path,
    observables: Sequence[RMTObservable] = ("rowsum_operator",),
    per_sequence_csv: Path | None = None,
    minimum_linearization_r2: float = 0.5,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Load a saved ``hidden_spectral_bridge.csv`` and fit every group."""
    rows = list(csv.DictReader(spectral_bridge_csv.open()))
    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), row["state_variant"], row.get("comparison_type", "propagated_hidden"))].append(row)

    sequence_groups: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    if per_sequence_csv is not None:
        for row in csv.DictReader(per_sequence_csv.open()):
            sequence_groups[(int(row["layer"]), row["state_variant"], row.get("comparison_type", "propagated_hidden"))].append(row)

    fits: list[BridgeFit] = []
    per_mode_rows: list[dict[str, Any]] = []
    for (layer, variant, comparison_type), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(row["mode"]))
        eigenvalues = np.asarray([float(row["eigenvalue"]) for row in ordered], dtype=np.float64)
        disagreement = np.asarray([float(row["hidden_disagreement"]) for row in ordered], dtype=np.float64)
        input_variances = None
        if all(row.get("input_variance_in_reference_basis", "") != "" for row in ordered):
            input_variances = np.asarray([float(row["input_variance_in_reference_basis"]) for row in ordered])
        sequence_payload = _sequence_matrix(sequence_groups.get((layer, variant, comparison_type), []), len(ordered))
        sequence_matrix = sequence_payload[0] if sequence_payload is not None else None
        sequence_weights = sequence_payload[1] if sequence_payload is not None else None
        linearization_r2 = float(ordered[0].get("linearization_r2", "nan"))
        for observable in observables:
            if observable == "prediction_covariance_weighted" and input_variances is None:
                raise ValueError(
                    f"layer {layer} ({variant}, {comparison_type}): weighted observable requires "
                    "input_variance_in_reference_basis in the spectral CSV"
                )
            fit, values, predicted, observed, ratio = fit_group(
                eigenvalues, disagreement, layer=layer, state_variant=variant, comparison_type=comparison_type,
                observable=observable,
                input_variances_in_eigenbasis=(
                    input_variances if observable == "prediction_covariance_weighted" else None
                ),
                sequence_disagreement=sequence_matrix, n_a=n_a, n_b=n_b,
                sequence_weights=sequence_weights,
                linearization_r2=linearization_r2, minimum_linearization_r2=minimum_linearization_r2,
                mask_fraction=mask_fraction, bootstrap_samples=bootstrap_samples, seed=seed,
            )
            fits.append(fit)
            full_prediction = reduce_perturbative_prediction(
                perturbative_split_prediction(eigenvalues, fit.gamma, n_a, n_b),
                observable=observable,
                input_variances_in_eigenbasis=input_variances if observable == "prediction_covariance_weighted" else None,
            ).per_output_mode
            usable = (eigenvalues > 0) & (disagreement > 0) & (full_prediction > 0)
            original_modes = np.flatnonzero(usable)
            for mode_index, value, pred, observed_value, ratio_value in zip(original_modes, values, predicted, observed, ratio):
                per_mode_rows.append({
                    "layer": layer, "state_variant": variant, "comparison_type": comparison_type,
                    "observable": observable, "mode": int(mode_index), "eigenvalue": float(value),
                    "predicted_disagreement_raw": float(pred), "predicted_disagreement_scaled": float(pred * fit.kappa_fixed_slope),
                    "observed_disagreement": float(observed_value), "observed_over_scaled_predicted": float(ratio_value),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    if not fits or not per_mode_rows:
        raise ValueError("no usable bridge groups were found")
    fit_rows = [fit.to_dict() for fit in fits]
    with (output_dir / "rmt_neural_bridge_fits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(fit_rows[0]))
        writer.writeheader()
        writer.writerows(fit_rows)
    with (output_dir / "rmt_neural_bridge_per_mode.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(per_mode_rows[0]))
        writer.writeheader()
        writer.writerows(per_mode_rows)
    summary = {
        "interpretation": (
            "Compares explicitly declared diagonal-operator, row-summed-operator, and/or input-covariance-"
            "weighted prediction observables. Per-output-mode predictions are not divided by width; only their "
            "aggregate uses 1/d. A free kappa is fit within each group, so these fits test shape, not cross-n "
            "magnitude. Slope uncertainty is reported only when per-sequence mode data are supplied; eigenmodes "
            "are never bootstrapped as independent observations."
        ),
        "source_csv": str(spectral_bridge_csv),
        "n_a": n_a, "n_b": n_b, "mask_fraction": mask_fraction, "observables": list(observables),
        "minimum_linearization_r2": minimum_linearization_r2,
        "per_sequence_csv": str(per_sequence_csv) if per_sequence_csv is not None else None,
        "fits": fit_rows,
    }
    (output_dir / "rmt_neural_bridge_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_cross_n_check(
    multi_n_spectral_bridge_csv: Path,
    *,
    reference_n: int,
    mask_fraction: float = 0.5,
    output_dir: Path,
    observables: Sequence[RMTObservable] = ("rowsum_operator",),
) -> dict[str, Any]:
    """Run ``check_shared_calibration`` for every (layer, state_variant) group in a
    multi-n ``hidden_spectral_bridge.csv`` (one that has a ``training_chunks`` column,
    as produced by ``dataset_size_evaluation.run_dataset_size_evaluation``).
    """
    rows = list(csv.DictReader(multi_n_spectral_bridge_csv.open()))
    grouped: dict[tuple[int, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(
            int(row["layer"]), row["state_variant"], row.get("comparison_type", "propagated_hidden"),
            int(row["training_chunks"]),
        )].append(row)

    by_group: dict[tuple[int, str, str], dict[int, list[dict[str, str]]]] = defaultdict(dict)
    for (layer, variant, comparison_type, n), group_rows in grouped.items():
        by_group[(layer, variant, comparison_type)][n] = group_rows

    all_results: list[CrossConditionCheck] = []
    for (layer, variant, comparison_type), by_n in sorted(by_group.items()):
        for observable in observables:
            predicted_by_n: dict[int, FloatArray] = {}
            observed_by_n: dict[int, FloatArray] = {}
            for n, group_rows in by_n.items():
                ordered = sorted(group_rows, key=lambda row: int(row["mode"]))
                eigenvalues = np.asarray([float(row["eigenvalue"]) for row in ordered], dtype=np.float64)
                disagreement = np.asarray([float(row["hidden_disagreement"]) for row in ordered], dtype=np.float64)
                input_variances = None
                if all(row.get("input_variance_in_reference_basis", "") != "" for row in ordered):
                    input_variances = np.asarray([float(row["input_variance_in_reference_basis"]) for row in ordered])
                if observable == "prediction_covariance_weighted" and input_variances is None:
                    continue
                try:
                    _, predicted, observed, _ = predicted_and_observed(
                        eigenvalues, disagreement, n_a=n, n_b=n, mask_fraction=mask_fraction,
                        observable=observable,
                        input_variances_in_eigenbasis=(
                            input_variances if observable == "prediction_covariance_weighted" else None
                        ),
                    )
                except ValueError:
                    continue
                predicted_by_n[n] = predicted
                observed_by_n[n] = observed
            if reference_n not in predicted_by_n or len(predicted_by_n) < 2:
                continue
            all_results.extend(check_shared_calibration(
                predicted_by_n, observed_by_n, reference_n=reference_n, layer=layer,
                state_variant=variant, observable=observable, comparison_type=comparison_type,
            ))

    output_dir.mkdir(parents=True, exist_ok=True)
    if not all_results:
        raise ValueError("no cross-n groups contained the reference n and at least one checked n")
    result_rows = [result.to_dict() for result in all_results]
    with (output_dir / "cross_n_shared_calibration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    summary = {
        "interpretation": (
            "Kappa is calibrated exactly once, on reference_n only, then applied with NO further fitting to "
            "every other n. r_squared_no_recalibration and ratio_median at n != reference_n are the decisive "
            "test of whether the theory's (1/n_A+1/n_B) magnitude prefactor predicts how disagreement "
            "actually changes with n -- unlike fit_group's per-n R^2, kappa cannot absorb that prefactor "
            "here. Only the reference condition's global scale calibration is tautological; its spectral-shape "
            "R2 and ratio spread remain diagnostic."
        ),
        "reference_n": reference_n, "mask_fraction": mask_fraction, "observables": list(observables),
        "results": result_rows,
    }
    (output_dir / "cross_n_shared_calibration_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
