"""Two-point dataset-size check after the neural representation gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .evaluation import (
    _write_csv,
    aggregate_with_bootstrap,
    create_evaluation_corruptions,
    evaluate_pair,
    load_checkpoint_model,
    spectral_bridge_diagnostics,
    summarize_evaluation,
)
from .training import select_device


def _plot_size_curve(
    output: Path,
    aggregate: list[dict[str, Any]],
    spectral: list[dict[str, Any]],
    summaries: dict[int, dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    sizes = sorted(summaries)
    for size in sizes:
        pair = f"cross_split_same_init_n{size}"
        selected = sorted(
            (row for row in aggregate if row["pair"] == pair and row["metric"] == "js_divergence"),
            key=lambda row: row["mask_fraction"],
        )
        x = np.asarray([row["mask_fraction"] for row in selected])
        y = np.asarray([row["mean"] for row in selected])
        low = np.asarray([row["ci_low"] for row in selected])
        high = np.asarray([row["ci_high"] for row in selected])
        axes[0, 0].plot(x, y, marker="o", label=f"n={size}")
        axes[0, 0].fill_between(x, low, high, alpha=0.15)
    axes[0, 0].set(xlabel="mask probability", ylabel="token-level JS divergence", title="Frozen-corruption cross-split disagreement")
    axes[0, 0].legend(frameon=False)

    mean_js = [summaries[size]["pairs"][f"cross_split_same_init_n{size}"]["js_divergence"]["mask_grid_mean"] for size in sizes]
    axes[0, 1].plot(sizes, mean_js, marker="o")
    axes[0, 1].set(xscale="log", xlabel="training chunks n", ylabel="mask-grid mean JS", title="Dataset-size direction check")

    for size in sizes:
        chosen = sorted(
            (
                row for row in spectral
                if row["training_chunks"] == size and row["layer"] == 1
                and row["state_variant"] == "per_token_layer_norm"
                and row.get("comparison_type", "propagated_hidden") == "propagated_hidden"
            ),
            key=lambda row: row["mode"],
        )
        axes[1, 0].plot([row["mode"] + 1 for row in chosen], [row["eigenvalue"] for row in chosen], label=f"n={size}")
    axes[1, 0].set(xscale="log", yscale="log", xlabel="spectral mode", ylabel="normalized covariance eigenvalue", title="Block-1 reference spectrum")
    axes[1, 0].legend(frameon=False)

    ranks = []
    correlations = []
    for size in sizes:
        block = next(
            row for row in summaries[size]["spectral"]
            if row["layer"] == 1 and row["state_variant"] == "per_token_layer_norm"
            and row.get("comparison_type", "propagated_hidden") == "propagated_hidden"
        )
        ranks.append(block["effective_rank"])
        correlations.append(block["log_eigenvalue_log_disagreement_correlation"])
    axes[1, 1].plot(sizes, ranks, marker="o", color="tab:blue", label="effective rank")
    axes[1, 1].set(xscale="log", xlabel="training chunks n", ylabel="effective rank", title="Exploratory spectral diagnostics")
    twin = axes[1, 1].twinx()
    twin.plot(sizes, correlations, marker="s", color="tab:orange", label="mode correlation")
    twin.set(ylabel="log-eigenvalue/log-disagreement correlation", ylim=(-1.05, 1.05))
    lines = axes[1, 1].lines + twin.lines
    axes[1, 1].legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=7)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.savefig(output / "dataset_size_curve.png", dpi=180)
    plt.close(figure)


def run_dataset_size_evaluation(
    config_path: Path,
    pairs: dict[int, tuple[Path, Path]],
    output: Path,
) -> None:
    """Evaluate all dataset-size pairs on one materialized corruption bank."""
    config = yaml.safe_load(config_path.read_text())
    device = select_device(config.get("device", "auto"))
    output.mkdir(parents=True, exist_ok=True)
    corruptions = create_evaluation_corruptions(config, output / "evaluation_corruptions.pt")
    tokenizer = json.loads((Path(config["data"]["tokenizer_dir"]) / "tokenizer_metadata.json").read_text())
    excluded = (int(tokenizer["mask_token_id"]), int(tokenizer["pad_token_id"]))
    pair_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    spectral_per_sequence_rows: list[dict[str, Any]] = []
    summaries: dict[int, dict[str, Any]] = {}
    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["evaluation"]["seed"])
    for size, (run_a, run_b) in sorted(pairs.items()):
        model_a = load_checkpoint_model(run_a / "checkpoint.pt", device)
        model_b = load_checkpoint_model(run_b / "checkpoint.pt", device)
        pair_name = f"cross_split_same_init_n{size}"
        rows = evaluate_pair(model_a, model_b, corruptions, config, pair_name=pair_name, device=device, excluded_ids=excluded)
        pair_rows.extend(rows)
        aggregate = aggregate_with_bootstrap(rows, bootstrap_samples, seed + size)
        spectra, spectra_per_sequence = spectral_bridge_diagnostics(model_a, model_b, corruptions, config, device)
        for row in spectra:
            row["training_chunks"] = size
        for row in spectra_per_sequence:
            row["training_chunks"] = size
        spectral_rows.extend(spectra)
        spectral_per_sequence_rows.extend(spectra_per_sequence)
        summaries[size] = summarize_evaluation(aggregate, spectra)
        del model_a, model_b
    aggregate_rows = aggregate_with_bootstrap(pair_rows, bootstrap_samples, seed)
    _write_csv(output / "denoiser_metrics_per_sequence.csv", pair_rows)
    _write_csv(output / "denoiser_metrics_aggregate.csv", aggregate_rows)
    _write_csv(output / "hidden_spectral_bridge.csv", spectral_rows)
    _write_csv(output / "hidden_spectral_bridge_per_sequence.csv", spectral_per_sequence_rows)
    payload = {
        "interpretation": "exploratory dataset-size direction check; two points do not establish a scaling law or RMT prediction",
        "sizes": {str(size): summary for size, summary in summaries.items()},
    }
    (output / "dataset_size_summary.json").write_text(json.dumps(payload, indent=2))
    _plot_size_curve(output, aggregate_rows, spectral_rows, summaries)
