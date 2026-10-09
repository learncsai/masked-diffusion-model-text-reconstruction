"""Configuration-driven Phase I experiment and artifact generation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "diffusion_lm_rmt_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "diffusion_lm_rmt_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml

from .rmt.core import (
    estimate_known_mean_covariance,
    masked_corrupted_covariance,
    masked_linear_denoiser_operator,
    monte_carlo_masked_prediction_disagreement,
    operator_disagreement,
    perturbative_split_prediction,
    prediction_disagreement,
    resolvent_filter,
    sample_gaussian,
    spectral_disagreement,
)
from .rmt.covariance import CovarianceModel, generate_covariance
from .rmt.fitting import fit_scaling_law


def _seed(base: int, *parts: int) -> int:
    return int(np.random.SeedSequence([base, *parts]).generate_state(1, dtype=np.uint32)[0])


def _covariance(config: dict[str, Any], family: str, d: int) -> CovarianceModel:
    options = config["covariance"]
    return generate_covariance(
        family, d, seed=int(config["covariance_seed"]), eigenvectors=config["eigenvectors"],
        spike_strengths=options["spike_strengths"], alpha=float(options["alpha"]),
        broad_condition_number=float(options["broad_condition_number"]),
    )


def _split_operators(model: CovarianceModel, n: int, gamma: float, seed_a: int, seed_b: int):
    samples_a = sample_gaussian(model.matrix, n, seed=seed_a)
    samples_b = sample_gaussian(model.matrix, n, seed=seed_b)
    covariance_a = estimate_known_mean_covariance(samples_a)
    covariance_b = estimate_known_mean_covariance(samples_b)
    return covariance_a, covariance_b, resolvent_filter(covariance_a, gamma), resolvent_filter(covariance_b, gamma)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    keys = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _mean_by(records: Iterable[dict[str, Any]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in records:
        groups[float(row[x_key])].append(float(row[y_key]))
    x = np.asarray(sorted(groups), dtype=np.float64)
    means = np.asarray([np.mean(groups[value]) for value in x], dtype=np.float64)
    standard_errors = np.asarray([np.std(groups[value], ddof=1) / np.sqrt(len(groups[value])) if len(groups[value]) > 1 else 0 for value in x])
    return x, means, standard_errors


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 180, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2,
    })


def _save_line_figure(path: Path, series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], xlabel: str, ylabel: str, *, logx=False, logy=False) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    for label, x, mean, error in series:
        ax.errorbar(x, mean, yerr=error, marker="o", capsize=2, linewidth=1.3, label=label)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def _plot_artifacts(output: Path, records: list[dict[str, Any]], matrices: dict[str, np.ndarray]) -> None:
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    _style()
    families = sorted({row["family"] for row in records if row["experiment"] == "classical"})
    series = []
    for family in families:
        selected = [row for row in records if row["experiment"] == "classical" and row["family"] == family]
        series.append((family, *_mean_by(selected, "n", "d_op")))
    _save_line_figure(figures / "disagreement_vs_n.png", series, "samples per split n", r"$D_{op}$", logx=True, logy=True)

    series = []
    high_rows = [row for row in records if row["experiment"] == "high_dimensional"]
    for family in sorted({row["family"] for row in high_rows}):
        for d in sorted({int(row["d"]) for row in high_rows if row["family"] == family}):
            selected = [row for row in high_rows if row["family"] == family and int(row["d"]) == d]
            series.append((f"{family}, d={d}", *_mean_by(selected, "aspect_ratio", "d_op")))
    _save_line_figure(figures / "disagreement_vs_aspect_ratio.png", series, r"aspect ratio $d/n$", r"$D_{op}$")

    selected = [row for row in records if row["experiment"] == "gamma_sweep"]
    _save_line_figure(figures / "disagreement_vs_gamma.png", [("power law", *_mean_by(selected, "gamma", "d_op"))], r"resolvent noise $\gamma$", r"$D_{op}$", logx=True, logy=True)

    series = []
    for family in ["isotropic", "power_law"]:
        for method in ["exact_mask", "resolvent_mask"]:
            selected = [row for row in records if row["experiment"] == "mask_sweep" and row["family"] == family and row["method"] == method]
            series.append((f"{family}: {method}", *_mean_by(selected, "mask_fraction", "d_op")))
    _save_line_figure(figures / "disagreement_vs_mask_fraction.png", series, "mask fraction 1-q", r"$D_{op}$", logy=True)

    selected = [row for row in records if row["experiment"] == "classical"]
    observed = np.asarray([row["d_op"] for row in selected])
    predicted = np.asarray([row["perturbative_d_op"] for row in selected])
    fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    for family in families:
        chosen = np.asarray([row["family"] == family for row in selected])
        ax.scatter(predicted[chosen], observed[chosen], s=13, alpha=0.65, label=family)
    bounds = [max(min(observed.min(), predicted.min()), 1e-12), max(observed.max(), predicted.max())]
    ax.plot(bounds, bounds, color="black", linestyle="--", linewidth=1, label="identity")
    ax.set(xscale="log", yscale="log", xlabel="perturbative prediction", ylabel="observed $D_{op}$")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(figures / "observed_vs_predicted.png")
    plt.close(fig)

    candidates = [key for key in matrices if key.startswith("observed_power_law")]
    if candidates:
        observed_key = max(candidates, key=lambda key: int(key.rsplit("_n", 1)[1]))
        predicted_key = observed_key.replace("observed_", "predicted_")
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), constrained_layout=True)
        positive = np.concatenate([matrices[observed_key].ravel(), matrices[predicted_key].ravel()])
        positive = positive[positive > 0]
        floor, ceiling = np.quantile(positive, [0.03, 0.97])
        from matplotlib.colors import LogNorm
        for ax, key, title in [(axes[0], observed_key, "observed"), (axes[1], predicted_key, "first-order prediction")]:
            image = ax.imshow(np.maximum(matrices[key], floor), origin="lower", norm=LogNorm(vmin=floor, vmax=ceiling), cmap="magma")
            ax.set(title=title, xlabel="population mode j", ylabel="population mode i")
        fig.colorbar(image, ax=axes, label=r"$\mathbb{E}[\Delta W_{ij}^2]$")
        fig.savefig(figures / "eigenmode_disagreement.png")
        plt.close(fig)


def _manifest(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip() or None
    except OSError:
        git_commit = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_file": str(config_path.resolve()), "configuration": config,
        "seeds": {"base": config["seed"], "covariance": config["covariance_seed"]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__},
        "system": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "git_commit": git_commit,
    }


def run(config_path: Path, output_override: Path | None = None) -> Path:
    """Execute all Phase I sweeps and return the output directory."""
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = output_override or Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    mode_sums: dict[str, np.ndarray] = {}
    mode_counts: dict[str, int] = defaultdict(int)
    per_mode_rows: list[dict[str, Any]] = []
    base_seed = int(config["seed"])
    replicates = int(config["replicates"])

    classical = config["classical"]
    for family_index, family in enumerate(config["covariance_families"]):
        model = _covariance(config, family, int(classical["d"]))
        for n_index, n in enumerate(classical["n_values"]):
            prediction = perturbative_split_prediction(model.eigenvalues, float(classical["gamma"]), int(n), int(n))
            key = f"{family}_n{n}"
            mode_sums[f"observed_{key}"] = np.zeros((model.matrix.shape[0], model.matrix.shape[0]))
            mode_sums[f"predicted_{key}"] = prediction.entrywise.copy()
            for replicate in range(replicates):
                _, _, operator_a, operator_b = _split_operators(
                    model, int(n), float(classical["gamma"]),
                    _seed(base_seed, 1, family_index, n_index, replicate, 0),
                    _seed(base_seed, 1, family_index, n_index, replicate, 1),
                )
                d_op, d_rms = operator_disagreement(operator_a, operator_b)
                spectral = spectral_disagreement(operator_a, operator_b, model.eigenvectors)
                mode_sums[f"observed_{key}"] += spectral.entrywise
                mode_counts[f"observed_{key}"] += 1
                records.append({
                    "experiment": "classical", "method": "resolvent", "family": family,
                    "d": model.matrix.shape[0], "n": int(n), "aspect_ratio": model.matrix.shape[0] / int(n),
                    "gamma": float(classical["gamma"]), "q": "", "mask_fraction": "", "replicate": replicate,
                    "d_op": d_op, "d_rms": d_rms, "perturbative_d_op": prediction.aggregate,
                    "prediction_ratio": d_op / prediction.aggregate, "relative_error": (d_op - prediction.aggregate) / prediction.aggregate,
                    "d_pred_analytic": "", "d_pred_mc": "", "mask_operator_approx_error_a": "", "mask_operator_approx_error_b": "",
                })
            observed_mean = mode_sums[f"observed_{key}"] / mode_counts[f"observed_{key}"]
            for mode in range(model.matrix.shape[0]):
                per_mode_rows.append({
                    "experiment": "classical", "family": family, "d": model.matrix.shape[0], "n": int(n),
                    "gamma": float(classical["gamma"]), "mode": mode, "eigenvalue": model.eigenvalues[mode],
                    "observed_contribution": observed_mean[mode].sum() / model.matrix.shape[0],
                    "predicted_contribution": prediction.entrywise[mode].sum() / model.matrix.shape[0],
                })

    high = config["high_dimensional"]
    for family_index, family in enumerate(high["families"]):
        for d_index, d in enumerate(high["d_values"]):
            model = _covariance(config, family, int(d))
            for ratio_index, ratio in enumerate(high["aspect_ratios"]):
                n = max(2, int(round(model.matrix.shape[0] / float(ratio))))
                prediction = perturbative_split_prediction(model.eigenvalues, float(high["gamma"]), n, n)
                for replicate in range(replicates):
                    _, _, operator_a, operator_b = _split_operators(model, n, float(high["gamma"]), _seed(base_seed, 2, family_index, d_index, ratio_index, replicate, 0), _seed(base_seed, 2, family_index, d_index, ratio_index, replicate, 1))
                    d_op, d_rms = operator_disagreement(operator_a, operator_b)
                    records.append({"experiment": "high_dimensional", "method": "resolvent", "family": family, "d": model.matrix.shape[0], "n": n, "aspect_ratio": model.matrix.shape[0] / n, "gamma": float(high["gamma"]), "q": "", "mask_fraction": "", "replicate": replicate, "d_op": d_op, "d_rms": d_rms, "perturbative_d_op": prediction.aggregate, "prediction_ratio": d_op / prediction.aggregate, "relative_error": (d_op - prediction.aggregate) / prediction.aggregate, "d_pred_analytic": "", "d_pred_mc": "", "mask_operator_approx_error_a": "", "mask_operator_approx_error_b": ""})

    gamma_config = config["gamma_sweep"]
    model = _covariance(config, gamma_config["family"], int(gamma_config["d"]))
    for gamma_index, gamma in enumerate(gamma_config["values"]):
        prediction = perturbative_split_prediction(model.eigenvalues, float(gamma), int(gamma_config["n"]), int(gamma_config["n"]))
        for replicate in range(replicates):
            _, _, operator_a, operator_b = _split_operators(model, int(gamma_config["n"]), float(gamma), _seed(base_seed, 3, gamma_index, replicate, 0), _seed(base_seed, 3, gamma_index, replicate, 1))
            d_op, d_rms = operator_disagreement(operator_a, operator_b)
            records.append({"experiment": "gamma_sweep", "method": "resolvent", "family": gamma_config["family"], "d": model.matrix.shape[0], "n": int(gamma_config["n"]), "aspect_ratio": model.matrix.shape[0] / int(gamma_config["n"]), "gamma": float(gamma), "q": "", "mask_fraction": "", "replicate": replicate, "d_op": d_op, "d_rms": d_rms, "perturbative_d_op": prediction.aggregate, "prediction_ratio": d_op / prediction.aggregate, "relative_error": (d_op - prediction.aggregate) / prediction.aggregate, "d_pred_analytic": "", "d_pred_mc": "", "mask_operator_approx_error_a": "", "mask_operator_approx_error_b": ""})

    mask_config = config["mask_sweep"]
    for family_index, family in enumerate(mask_config["families"]):
        model = _covariance(config, family, int(mask_config["d"]))
        c = float(np.mean(np.diag(model.matrix)))
        for q_index, q in enumerate(mask_config["q_values"]):
            q = float(q)
            gamma = c * (1.0 - q) / q
            corrupted_population = masked_corrupted_covariance(model.matrix, q)
            for replicate in range(replicates):
                covariance_a, covariance_b, _, _ = _split_operators(model, int(mask_config["n"]), max(gamma, 1e-12), _seed(base_seed, 4, family_index, q_index, replicate, 0), _seed(base_seed, 4, family_index, q_index, replicate, 1))
                exact_a = masked_linear_denoiser_operator(covariance_a, q)
                exact_b = masked_linear_denoiser_operator(covariance_b, q)
                resolvent_a = resolvent_filter(covariance_a, gamma)
                resolvent_b = resolvent_filter(covariance_b, gamma)
                approx_a, _ = operator_disagreement(exact_a, resolvent_a)
                approx_b, _ = operator_disagreement(exact_b, resolvent_b)
                for method, operator_a, operator_b in [("exact_mask", exact_a, exact_b), ("resolvent_mask", resolvent_a, resolvent_b)]:
                    d_op, d_rms = operator_disagreement(operator_a, operator_b)
                    analytic = prediction_disagreement(operator_a, operator_b, corrupted_population)
                    mc = monte_carlo_masked_prediction_disagreement(operator_a, operator_b, model.matrix, q, draws=int(mask_config["monte_carlo_draws"]), seed=_seed(base_seed, 5, family_index, q_index, replicate))
                    perturbative = perturbative_split_prediction(model.eigenvalues, gamma, int(mask_config["n"]), int(mask_config["n"])).aggregate if method == "resolvent_mask" else ""
                    records.append({"experiment": "mask_sweep", "method": method, "family": family, "d": model.matrix.shape[0], "n": int(mask_config["n"]), "aspect_ratio": model.matrix.shape[0] / int(mask_config["n"]), "gamma": gamma, "q": q, "mask_fraction": 1.0 - q, "replicate": replicate, "d_op": d_op, "d_rms": d_rms, "perturbative_d_op": perturbative, "prediction_ratio": d_op / perturbative if perturbative != "" else "", "relative_error": (d_op - perturbative) / perturbative if perturbative != "" else "", "d_pred_analytic": analytic, "d_pred_mc": mc, "mask_operator_approx_error_a": approx_a, "mask_operator_approx_error_b": approx_b})

    averaged_matrices = {key: value / mode_counts[key] if key.startswith("observed_") else value for key, value in mode_sums.items()}
    _write_csv(output / "experiment_records.csv", records)
    _write_csv(output / "per_mode_contributions.csv", per_mode_rows)
    np.savez_compressed(output / "eigenmode_disagreement_matrices.npz", **averaged_matrices)

    fits = []
    n_values = np.asarray(classical["n_values"], dtype=np.int64)
    for family_index, family in enumerate(config["covariance_families"]):
        table = np.empty((n_values.size, replicates), dtype=np.float64)
        for n_index, n in enumerate(n_values):
            selected = [row for row in records if row["experiment"] == "classical" and row["family"] == family and row["n"] == int(n)]
            table[n_index] = [float(row["d_op"]) for row in selected]
        for with_offset in [True, False]:
            fit = fit_scaling_law(n_values, table, with_offset=with_offset, bootstrap_samples=int(config["bootstrap_samples"]), seed=_seed(base_seed, 6, family_index, int(with_offset)))
            fits.append({"family": family, **fit.to_dict()})
    with (output / "scaling_fits.json").open("w", encoding="utf-8") as handle:
        json.dump(fits, handle, indent=2)
    _write_csv(output / "scaling_fits.csv", fits)
    with (output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_manifest(config_path, config), handle, indent=2)
    _plot_artifacts(output, records, averaged_matrices)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = run(arguments.config, arguments.output)
    print(f"Artifacts written to {output.resolve()}")


if __name__ == "__main__":
    main()
