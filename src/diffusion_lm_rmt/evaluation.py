"""Controlled denoiser reproducibility metrics and exploratory spectral bridge."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from .diffusion.corruption import corrupt_tokens
from .diffusion.model import DenoiserConfig, MaskedDiffusionTransformer
from .neural_data import ChunkDataset
from .training import select_device


def load_checkpoint_model(path: Path, device: torch.device) -> MaskedDiffusionTransformer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = MaskedDiffusionTransformer(DenoiserConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def create_evaluation_corruptions(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    """Materialize one fixed corruption tensor per requested mask probability."""
    data_dir = Path(config["data"]["processed_dir"])
    tokenizer = json.loads((Path(config["data"]["tokenizer_dir"]) / "tokenizer_metadata.json").read_text())
    dataset = ChunkDataset(data_dir / "eval.pt")
    special = [tokenizer["mask_token_id"], tokenizer["pad_token_id"], tokenizer["eos_token_id"]]
    payload: dict[str, Any] = {"input_ids": dataset.input_ids, "attention_mask": dataset.attention_mask, "fractions": {}}
    base_seed = int(config["evaluation"]["seed"])
    for index, fraction in enumerate(config["evaluation"]["mask_fractions"]):
        times = torch.full((len(dataset),), float(fraction))
        corruption = corrupt_tokens(dataset.input_ids, dataset.attention_mask, times, mask_token_id=tokenizer["mask_token_id"], special_token_ids=special, generator=torch.Generator().manual_seed(base_seed + index), ensure_masked=True)
        payload["fractions"][str(fraction)] = {"corrupted_ids": corruption.corrupted_ids, "target_mask": corruption.target_mask, "times": times}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    (output_path.with_suffix(".sha256")).write_text(digest + "\n")
    return payload


def _token_metrics(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    true_ids: torch.Tensor,
    *,
    chunk_size: int,
    excluded_ids: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    """Compute float32 probability metrics while streaming vocabulary chunks."""
    a, b = logits_a.float().clone(), logits_b.float().clone()
    vocab = a.shape[-1]
    allowed = torch.ones(vocab, dtype=torch.bool)
    for token_id in excluded_ids:
        allowed[int(token_id)] = False
        a[:, int(token_id)] = -torch.inf; b[:, int(token_id)] = -torch.inf
    log_z_a, log_z_b = torch.logsumexp(a, dim=-1), torch.logsumexp(b, dim=-1)
    mean_a, mean_b = a[:, allowed].mean(dim=-1), b[:, allowed].mean(dim=-1)
    js = torch.zeros_like(log_z_a); skl = torch.zeros_like(log_z_a)
    prob_dot = torch.zeros_like(log_z_a); prob_a_sq = torch.zeros_like(log_z_a); prob_b_sq = torch.zeros_like(log_z_a)
    centered_dot = torch.zeros_like(log_z_a); centered_a_sq = torch.zeros_like(log_z_a); centered_b_sq = torch.zeros_like(log_z_a)
    true_a = torch.zeros_like(log_z_a); true_b = torch.zeros_like(log_z_a)
    for start in range(0, vocab, chunk_size):
        stop = min(start + chunk_size, vocab)
        valid = allowed[start:stop]
        chunk_a, chunk_b = a[:, start:stop][:, valid], b[:, start:stop][:, valid]
        log_a, log_b = chunk_a - log_z_a[:, None], chunk_b - log_z_b[:, None]
        prob_a, prob_b = log_a.exp(), log_b.exp()
        log_mix = torch.logaddexp(log_a, log_b) - math.log(2.0)
        js += 0.5 * ((prob_a * (log_a - log_mix)).sum(-1) + (prob_b * (log_b - log_mix)).sum(-1))
        skl += 0.5 * ((prob_a * (log_a - log_b)).sum(-1) + (prob_b * (log_b - log_a)).sum(-1))
        prob_dot += (prob_a * prob_b).sum(-1); prob_a_sq += (prob_a * prob_a).sum(-1); prob_b_sq += (prob_b * prob_b).sum(-1)
        centered_a, centered_b = chunk_a - mean_a[:, None], chunk_b - mean_b[:, None]
        centered_dot += (centered_a * centered_b).sum(-1); centered_a_sq += (centered_a * centered_a).sum(-1); centered_b_sq += (centered_b * centered_b).sum(-1)
        inside = (true_ids >= start) & (true_ids < stop)
        if inside.any():
            original_ids = torch.arange(start, stop)[valid]
            lookup = torch.full((stop - start,), -1, dtype=torch.long)
            lookup[original_ids - start] = torch.arange(original_ids.numel())
            local = lookup[true_ids[inside] - start]
            if (local < 0).any():
                raise ValueError("true target is an excluded generated token")
            true_a[inside] = prob_a[inside, local]; true_b[inside] = prob_b[inside, local]
    eps = 1e-30
    return {
        "js_divergence": js.clamp_min(0), "symmetrized_kl": skl.clamp_min(0),
        "probability_cosine": prob_dot / (prob_a_sq * prob_b_sq).sqrt().clamp_min(eps),
        "centered_logit_correlation": centered_dot / (centered_a_sq * centered_b_sq).sqrt().clamp_min(eps),
        "true_probability_abs_difference": (true_a - true_b).abs(),
        "top1_agreement": (a.argmax(-1) == b.argmax(-1)).float(),
        "accuracy_a": (a.argmax(-1) == true_ids).float(), "accuracy_b": (b.argmax(-1) == true_ids).float(),
        "nll_a": -torch.log(true_a.clamp_min(eps)), "nll_b": -torch.log(true_b.clamp_min(eps)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def evaluate_pair(
    model_a: MaskedDiffusionTransformer,
    model_b: MaskedDiffusionTransformer,
    corruptions: dict[str, Any],
    config: dict[str, Any],
    *,
    pair_name: str,
    device: torch.device,
    excluded_ids: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    """Return per-sequence aggregates on identical precomputed corruptions."""
    rows: list[dict[str, Any]] = []
    batch_size = int(config["evaluation"]["batch_size"])
    for fraction in config["evaluation"]["mask_fractions"]:
        data = corruptions["fractions"][str(fraction)]
        for start in range(0, corruptions["input_ids"].shape[0], batch_size):
            stop = min(start + batch_size, corruptions["input_ids"].shape[0])
            clean = corruptions["input_ids"][start:stop]
            attention = corruptions["attention_mask"][start:stop]
            corrupted = data["corrupted_ids"][start:stop]
            targets = data["target_mask"][start:stop]
            times = data["times"][start:stop]
            logits_a = model_a(corrupted.to(device), attention.to(device), times.to(device)).float().cpu()
            logits_b = model_b(corrupted.to(device), attention.to(device), times.to(device)).float().cpu()
            for row_index in range(stop - start):
                selected = targets[row_index]
                metrics = _token_metrics(logits_a[row_index, selected], logits_b[row_index, selected], clean[row_index, selected], chunk_size=int(config["evaluation"]["vocabulary_chunk_size"]), excluded_ids=excluded_ids)
                row = {"pair": pair_name, "mask_fraction": float(fraction), "sequence_index": start + row_index, "masked_tokens": int(selected.sum())}
                row.update({name: float(values.mean()) for name, values in metrics.items()})
                rows.append(row)
    return rows


def aggregate_with_bootstrap(rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    """Pool token means while bootstrapping whole held-out sequences.

    Each input metric is a within-sequence token mean.  Weighting by the number
    of masked tokens therefore recovers the requested token-level estimate,
    while resampling rows (rather than individual tokens) keeps the sequence as
    the uncertainty unit.
    """
    metrics = [key for key in rows[0] if key not in {"pair", "mask_fraction", "sequence_index", "masked_tokens"}]
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["pair"], row["mask_fraction"])].append(row)
    rng = np.random.default_rng(seed)
    output = []
    for (pair, fraction), group in grouped.items():
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            weights = np.asarray([row["masked_tokens"] for row in group], dtype=np.float64)
            indices = np.arange(values.size)
            draws = (rng.choice(indices, values.size, replace=True) for _ in range(samples))
            bootstrap = np.asarray([
                np.average(values[draw], weights=weights[draw]) for draw in draws
            ])
            output.append({"pair": pair, "mask_fraction": fraction, "metric": metric, "mean": np.average(values, weights=weights), "ci_low": np.quantile(bootstrap, 0.025), "ci_high": np.quantile(bootstrap, 0.975), "sequences": values.size, "masked_tokens": int(weights.sum()), "aggregation": "masked-token weighted; sequence bootstrap"})
    return output


def paired_js_contrasts(rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    """Paired sequence-bootstrap differences between controlled JS curves."""
    definitions = {
        "cross_split_minus_initialization": ("cross_split_same_init", "same_data_diff_init"),
        "fully_independent_minus_cross_split_same_init": ("cross_split_diff_init", "cross_split_same_init"),
    }
    lookup = {
        (row["pair"], float(row["mask_fraction"]), int(row["sequence_index"])): row
        for row in rows
    }
    fractions = sorted({float(row["mask_fraction"]) for row in rows})
    rng = np.random.default_rng(seed)
    output = []
    for name, (left_pair, right_pair) in definitions.items():
        for fraction in fractions:
            sequence_ids = sorted({
                int(row["sequence_index"]) for row in rows
                if row["pair"] == left_pair and float(row["mask_fraction"]) == fraction
                and (right_pair, fraction, int(row["sequence_index"])) in lookup
            })
            if not sequence_ids:
                continue
            left = np.asarray([lookup[(left_pair, fraction, index)]["js_divergence"] for index in sequence_ids])
            right = np.asarray([lookup[(right_pair, fraction, index)]["js_divergence"] for index in sequence_ids])
            weights = np.asarray([lookup[(left_pair, fraction, index)]["masked_tokens"] for index in sequence_ids], dtype=np.float64)
            differences = left - right
            indices = np.arange(len(sequence_ids))
            bootstrap = np.asarray([
                np.average(differences[draw], weights=weights[draw])
                for draw in (rng.choice(indices, len(indices), replace=True) for _ in range(samples))
            ])
            output.append({
                "contrast": name,
                "left_pair": left_pair,
                "right_pair": right_pair,
                "mask_fraction": fraction,
                "mean_js_difference": float(np.average(differences, weights=weights)),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
                "sequences": len(sequence_ids),
                "masked_tokens": int(weights.sum()),
            })
    return output


def _crossfit_linearization(
    inputs: torch.Tensor,
    differences: torch.Tensor,
    sequence_ids: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    ridge: float = 1e-3,
) -> tuple[float, torch.Tensor]:
    """Cross-fit ``difference ~= input @ operator`` with a sequence-level split."""
    unique = torch.unique(sequence_ids, sorted=True)
    if unique.numel() < 4:
        return float("nan"), torch.full((differences.shape[1],), float("nan"), dtype=torch.double)
    calibration_ids = unique[::2]
    validation_ids = unique[1::2]
    calibration = torch.isin(sequence_ids, calibration_ids)
    validation = torch.isin(sequence_ids, validation_ids)
    x_train, y_train = inputs[calibration], differences[calibration]
    x_test, y_test = inputs[validation], differences[validation]
    x_mean, y_mean = x_train.mean(0, keepdim=True), y_train.mean(0, keepdim=True)
    x_train_centered, y_train_centered = x_train - x_mean, y_train - y_mean
    scale = torch.trace(x_train_centered.T @ x_train_centered) / max(x_train_centered.shape[1], 1)
    penalty = ridge * scale.clamp_min(1e-12)
    operator = torch.linalg.solve(
        x_train_centered.T @ x_train_centered + penalty * torch.eye(x_train.shape[1], dtype=torch.double),
        x_train_centered.T @ y_train_centered,
    )
    predicted = (x_test - x_mean) @ operator + y_mean
    residual = y_test - predicted
    centered = y_test - y_test.mean(0, keepdim=True)
    total = centered.square().sum().clamp_min(1e-30)
    global_r2 = 1.0 - float(residual.square().sum() / total)
    projected_residual = residual @ eigenvectors
    projected_centered = centered @ eigenvectors
    mode_r2 = 1.0 - projected_residual.square().sum(0) / projected_centered.square().sum(0).clamp_min(1e-30)
    return global_r2, mode_r2


@torch.no_grad()
def spectral_bridge_diagnostics(
    model_a: MaskedDiffusionTransformer,
    model_b: MaskedDiffusionTransformer,
    corruptions: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    *,
    same_initialization: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure propagated and common-block-input disagreement with input statistics.

    The common-input comparison applies both learned blocks to exactly the same
    averaged upstream hidden sequence.  This removes disagreement propagated
    from earlier layers.  The recorded input covariance is token-level and is
    expressed in the output-reference eigenbasis; attention-mediated cross-token
    covariance remains an explicit approximation.

    ``same_initialization`` must be set to ``False`` when ``model_a`` and
    ``model_b`` were independently initialized: this function directly
    subtracts raw hidden coordinates with no Procrustes/CCA alignment step,
    which is only valid when both models share their initial weights (so
    their hidden bases are guaranteed to correspond coordinate-for-coordinate
    at initialization). Passing ``False`` records this in the saved
    ``alignment`` field instead of silently mislabeling every row as
    identically initialized; it does not change the (still unaligned, still
    not a validated comparison) computation itself.
    """
    data = corruptions["fractions"][str(0.5)]
    maximum = int(config["evaluation"]["hidden_state_max_tokens"])
    layers = [int(layer) for layer in config["evaluation"]["hidden_layers"]]
    states_a: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    states_b: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    shared_inputs: dict[int, list[torch.Tensor]] = defaultdict(list)
    sequence_parts: list[torch.Tensor] = []
    collected = 0
    for start in range(0, corruptions["input_ids"].shape[0], 2):
        stop = min(start + 2, corruptions["input_ids"].shape[0])
        ids, attention, times = data["corrupted_ids"][start:stop], corruptions["attention_mask"][start:stop], data["times"][start:stop]
        device_ids, device_attention, device_times = ids.to(device), attention.to(device), times.to(device)
        _, hidden_a, inputs_a = model_a(
            device_ids, device_attention, device_times, return_hidden_states=True, return_block_inputs=True,
        )
        _, hidden_b, inputs_b = model_b(
            device_ids, device_attention, device_times, return_hidden_states=True, return_block_inputs=True,
        )
        time_a, time_b = model_a.time_embedding(device_times), model_b.time_embedding(device_times)
        padding_mask = ~device_attention.bool()
        active = attention.bool().reshape(-1)
        sequence_grid = torch.arange(start, stop)[:, None].expand_as(attention).reshape(-1)[active]
        sequence_parts.append(sequence_grid)
        for layer in layers:
            common_input = 0.5 * (inputs_a[layer] + inputs_b[layer])
            controlled_a = model_a.blocks[layer](common_input, time_a, padding_mask)
            controlled_b = model_b.blocks[layer](common_input, time_b, padding_mask)
            for comparison_type, first, second in (
                ("propagated_hidden", hidden_a[layer], hidden_b[layer]),
                ("shared_input_block", controlled_a, controlled_b),
            ):
                states_a[(comparison_type, layer)].append(first.float().cpu().reshape(-1, first.shape[-1])[active])
                states_b[(comparison_type, layer)].append(second.float().cpu().reshape(-1, second.shape[-1])[active])
            shared_inputs[layer].append(common_input.float().cpu().reshape(-1, common_input.shape[-1])[active])
        collected += int(active.sum())
        if collected >= maximum:
            break
    sequence_ids = torch.cat(sequence_parts)[:maximum]
    rows: list[dict[str, Any]] = []
    per_sequence_rows: list[dict[str, Any]] = []
    for comparison_type in ("propagated_hidden", "shared_input_block"):
        for layer in layers:
            raw_a = torch.cat(states_a[(comparison_type, layer)])[:maximum].double()
            raw_b = torch.cat(states_b[(comparison_type, layer)])[:maximum].double()
            raw_input = torch.cat(shared_inputs[layer])[:maximum].double()
            variants = {
                "raw": (raw_a, raw_b, raw_input),
                "per_token_layer_norm": (
                    torch.nn.functional.layer_norm(raw_a, (raw_a.shape[-1],)),
                    torch.nn.functional.layer_norm(raw_b, (raw_b.shape[-1],)),
                    torch.nn.functional.layer_norm(raw_input, (raw_input.shape[-1],)),
                ),
            }
            for variant, (a, b, shared_input) in variants.items():
                reference = 0.5 * (a + b); reference -= reference.mean(dim=0, keepdim=True)
                covariance = reference.T @ reference / max(reference.shape[0] - 1, 1)
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                order = torch.argsort(eigenvalues, descending=True); eigenvalues, eigenvectors = eigenvalues[order].clamp_min(0), eigenvectors[:, order]
                delta = (a - b) - (a - b).mean(dim=0, keepdim=True)
                projected_delta_squared = (delta @ eigenvectors).square()
                disagreement = projected_delta_squared.mean(dim=0)
                centered_input = shared_input - shared_input.mean(dim=0, keepdim=True)
                input_variances = (centered_input @ eigenvectors).square().sum(dim=0) / max(centered_input.shape[0] - 1, 1)
                linearization_r2, linearization_mode_r2 = _crossfit_linearization(
                    shared_input, a - b, sequence_ids, eigenvectors,
                    ridge=float(config["evaluation"].get("linearization_ridge", 1e-3)),
                )
                total = eigenvalues.sum().clamp_min(1e-30); probabilities = eigenvalues / total
                disagreement_total = disagreement.sum().clamp_min(1e-30)
                participation_ratio = float(total.square() / eigenvalues.square().sum().clamp_min(1e-30))
                effective_rank = float(torch.exp(-(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()))
                covariance_cumulative = eigenvalues.cumsum(0) / total
                disagreement_cumulative = disagreement.cumsum(0) / disagreement_total
                for mode in range(eigenvalues.numel()):
                    rows.append({
                        "layer": layer, "state_variant": variant, "comparison_type": comparison_type,
                        "mode": mode, "eigenvalue": float(eigenvalues[mode]),
                        "hidden_disagreement": float(disagreement[mode]),
                        "input_variance_in_reference_basis": float(input_variances[mode]),
                        "linearization_r2": linearization_r2,
                        "linearization_mode_r2": float(linearization_mode_r2[mode]),
                        "cumulative_covariance_fraction": float(covariance_cumulative[mode]),
                        "cumulative_disagreement_fraction": float(disagreement_cumulative[mode]),
                        "participation_ratio": participation_ratio, "effective_rank": effective_rank,
                        "samples": reference.shape[0],
                        "alignment": (
                            "shared coordinates; identical initialization"
                            if same_initialization else
                            "UNALIGNED coordinates; independent initialization -- raw subtraction is not a "
                            "validated comparison without Procrustes/CCA alignment"
                        ),
                        "input_covariance_basis": "diagonal of shared upstream token covariance in output-reference eigenbasis",
                        "input_covariance_approximation": "token-level; cross-token attention covariance omitted",
                    })
                for sequence_id in torch.unique(sequence_ids, sorted=True):
                    selected = sequence_ids == sequence_id
                    sequence_disagreement = projected_delta_squared[selected].mean(dim=0)
                    for mode in range(eigenvalues.numel()):
                        per_sequence_rows.append({
                            "layer": layer, "state_variant": variant, "comparison_type": comparison_type,
                            "sequence_index": int(sequence_id), "mode": mode,
                            "hidden_disagreement": float(sequence_disagreement[mode]),
                            "tokens": int(selected.sum()),
                        })
    return rows, per_sequence_rows


@torch.no_grad()
def spectral_bridge(
    model_a: MaskedDiffusionTransformer,
    model_b: MaskedDiffusionTransformer,
    corruptions: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Backward-compatible aggregate-only wrapper."""
    rows, _ = spectral_bridge_diagnostics(model_a, model_b, corruptions, config, device)
    return rows


def _plot_milestone(output: Path, aggregate: list[dict[str, Any]], spectral: list[dict[str, Any]], rmt_records: Path, filename: str = "milestone_1.png") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    for ax, pair, title in [(axes[0, 0], "cross_split_same_init", "Cross-split / same initialization"), (axes[0, 1], "same_data_diff_init", "Same-data controls")]:
        selected = [row for row in aggregate if row["pair"] == pair and row["metric"] == "js_divergence"]
        x = np.array([row["mask_fraction"] for row in selected]); y = np.array([row["mean"] for row in selected])
        low = np.array([row["ci_low"] for row in selected]); high = np.array([row["ci_high"] for row in selected])
        label = "different initialization" if pair == "same_data_diff_init" else "cross split"
        ax.plot(x, y, marker="o", label=label); ax.fill_between(x, low, high, alpha=.2); ax.set(title=title, xlabel="mask probability", ylabel="token-level JS divergence")
        if pair == "cross_split_same_init":
            combined = [row for row in aggregate if row["pair"] == "cross_split_diff_init" and row["metric"] == "js_divergence"]
            if combined:
                cx = np.array([row["mask_fraction"] for row in combined]); cy = np.array([row["mean"] for row in combined])
                cl = np.array([row["ci_low"] for row in combined]); ch = np.array([row["ci_high"] for row in combined])
                ax.plot(cx, cy, marker="s", label="cross split + different init"); ax.fill_between(cx, cl, ch, alpha=.15)
                ax.legend(frameon=False, fontsize=7)
        if pair == "same_data_diff_init":
            control = [row for row in aggregate if row["pair"] == "same_data_same_init_diff_sgd" and row["metric"] == "js_divergence"]
            if control:
                cx = np.array([row["mask_fraction"] for row in control]); cy = np.array([row["mean"] for row in control])
                cl = np.array([row["ci_low"] for row in control]); ch = np.array([row["ci_high"] for row in control])
                ax.plot(cx, cy, marker="s", label="different SGD/corruption stream"); ax.fill_between(cx, cl, ch, alpha=.15)
            ax.legend(frameon=False, fontsize=7)
    with rmt_records.open() as handle:
        rmt = list(csv.DictReader(handle))
    selected = [row for row in rmt if row["experiment"] == "mask_sweep" and row["family"] == "power_law" and row["method"] == "exact_mask"]
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in selected: grouped[float(row["gamma"])].append(float(row["d_op"]))
    x = np.array(sorted(grouped)); y = np.array([np.mean(grouped[value]) for value in x])
    axes[1, 0].plot(x, y, marker="o"); axes[1, 0].set(xscale="log", yscale="log", xlabel=r"effective $\gamma=c(1-q)/q$", ylabel=r"Gaussian $D_{op}$", title="Linear mask-aware baseline")
    for layer in sorted({row["layer"] for row in spectral}):
        chosen = [
            row for row in spectral
            if row["layer"] == layer
            and row.get("state_variant", "raw") == "per_token_layer_norm"
            and row.get("comparison_type", "propagated_hidden") == "propagated_hidden"
        ]
        axes[1, 1].plot([row["mode"] + 1 for row in chosen], [row["eigenvalue"] for row in chosen], label=f"layer {layer}")
    axes[1, 1].set(xscale="log", yscale="log", xlabel="spectral mode", ylabel="hidden covariance eigenvalue", title="Empirical hidden-state spectrum"); axes[1, 1].legend(frameon=False)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def summarize_evaluation(aggregate: list[dict[str, Any]], spectral: list[dict[str, Any]], contrasts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Create compact, machine-readable headline and spectral diagnostics."""
    pairs = sorted({row["pair"] for row in aggregate})
    pair_summary: dict[str, Any] = {}
    for pair in pairs:
        pair_summary[pair] = {}
        for metric in sorted({row["metric"] for row in aggregate if row["pair"] == pair}):
            selected = sorted(
                (row for row in aggregate if row["pair"] == pair and row["metric"] == metric),
                key=lambda row: row["mask_fraction"],
            )
            fractions = np.asarray([row["mask_fraction"] for row in selected], dtype=np.float64)
            means = np.asarray([row["mean"] for row in selected], dtype=np.float64)
            pair_summary[pair][metric] = {
                "mask_grid_mean": float(means.mean()),
                "minimum": float(means.min()),
                "maximum": float(means.max()),
                "mask_fraction_pearson_correlation": float(np.corrcoef(fractions, means)[0, 1]) if means.size > 1 and means.std() > 0 else None,
            }
    cross_js = pair_summary.get("cross_split_same_init", {}).get("js_divergence", {}).get("mask_grid_mean")
    init_js = pair_summary.get("same_data_diff_init", {}).get("js_divergence", {}).get("mask_grid_mean")
    comparison = {
        "cross_split_to_initialization_js_ratio": cross_js / init_js
        if cross_js is not None and init_js not in {None, 0}
        else None
    }
    spectral_summary: list[dict[str, Any]] = []
    keys = sorted({(
        int(row["layer"]), row.get("state_variant", "raw"),
        row.get("comparison_type", "propagated_hidden"),
    ) for row in spectral})
    for layer, variant, comparison_type in keys:
        selected = sorted(
            (
                row for row in spectral
                if int(row["layer"]) == layer
                and row.get("state_variant", "raw") == variant
                and row.get("comparison_type", "propagated_hidden") == comparison_type
            ),
            key=lambda row: int(row["mode"]),
        )
        eigenvalues = np.asarray([row["eigenvalue"] for row in selected], dtype=np.float64)
        disagreement = np.asarray([row["hidden_disagreement"] for row in selected], dtype=np.float64)
        positive = (eigenvalues > 0) & (disagreement > 0)
        log_correlation = float(np.corrcoef(np.log(eigenvalues[positive]), np.log(disagreement[positive]))[0, 1]) if positive.sum() > 1 else None
        top_index = min(9, len(selected) - 1)
        spectral_summary.append({
            "layer": layer,
            "state_variant": variant,
            "comparison_type": comparison_type,
            "effective_rank": float(selected[0]["effective_rank"]),
            "participation_ratio": float(selected[0]["participation_ratio"]),
            "top_10_covariance_fraction": float(selected[top_index]["cumulative_covariance_fraction"]),
            "top_10_disagreement_fraction": float(selected[top_index]["cumulative_disagreement_fraction"]),
            "log_eigenvalue_log_disagreement_correlation": log_correlation,
        })
    contrast_summary = {}
    for name in sorted({row["contrast"] for row in contrasts or []}):
        selected = [row for row in contrasts or [] if row["contrast"] == name]
        contrast_summary[name] = {
            "mask_grid_mean_js_difference": float(np.mean([row["mean_js_difference"] for row in selected])),
            "all_mask_rate_intervals_above_zero": all(row["ci_low"] > 0 for row in selected),
            "all_mask_rate_intervals_below_zero": all(row["ci_high"] < 0 for row in selected),
        }
    return {"pairs": pair_summary, "comparisons": comparison, "paired_js_contrasts": contrast_summary, "spectral": spectral_summary}


def run_evaluation(config_path: Path, run_dirs: dict[str, Path], output: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    device = select_device(config.get("device", "auto")); output.mkdir(parents=True, exist_ok=True)
    corruption_path = output / "evaluation_corruptions.pt"
    corruptions = create_evaluation_corruptions(config, corruption_path)
    tokenizer = json.loads((Path(config["data"]["tokenizer_dir"]) / "tokenizer_metadata.json").read_text())
    excluded = (int(tokenizer["mask_token_id"]), int(tokenizer["pad_token_id"]))
    models = {name: load_checkpoint_model(path / "checkpoint.pt", device) for name, path in run_dirs.items()}
    pair_rows = []
    pair_rows += evaluate_pair(models["split_a_init0"], models["split_b_init0"], corruptions, config, pair_name="cross_split_same_init", device=device, excluded_ids=excluded)
    pair_rows += evaluate_pair(models["split_a_init0"], models["split_a_init1"], corruptions, config, pair_name="same_data_diff_init", device=device, excluded_ids=excluded)
    pair_rows += evaluate_pair(models["split_a_init1"], models["split_b_init0"], corruptions, config, pair_name="cross_split_diff_init", device=device, excluded_ids=excluded)
    if "split_a_sgd1" in models:
        pair_rows += evaluate_pair(models["split_a_init0"], models["split_a_sgd1"], corruptions, config, pair_name="same_data_same_init_diff_sgd", device=device, excluded_ids=excluded)
    self_rows = evaluate_pair(models["split_a_init0"], models["split_a_init0"], corruptions, config, pair_name="self_check", device=device, excluded_ids=excluded)
    if max(row["js_divergence"] for row in self_rows) > 1e-7 or min(row["top1_agreement"] for row in self_rows) < 1.0:
        raise RuntimeError("self-comparison invariants failed")
    pair_rows += self_rows
    per_sequence = output / "denoiser_metrics_per_sequence.csv"; _write_csv(per_sequence, pair_rows)
    aggregate = aggregate_with_bootstrap(pair_rows, int(config["evaluation"]["bootstrap_samples"]), int(config["evaluation"]["seed"]))
    _write_csv(output / "denoiser_metrics_aggregate.csv", aggregate)
    contrasts = paired_js_contrasts(pair_rows, int(config["evaluation"]["bootstrap_samples"]), int(config["evaluation"]["seed"]))
    _write_csv(output / "paired_js_contrasts.csv", contrasts)
    spectral, spectral_per_sequence = spectral_bridge_diagnostics(
        models["split_a_init0"], models["split_b_init0"], corruptions, config, device,
    )
    _write_csv(output / "hidden_spectral_bridge.csv", spectral)
    _write_csv(output / "hidden_spectral_bridge_per_sequence.csv", spectral_per_sequence)
    (output / "evaluation_summary.json").write_text(json.dumps(summarize_evaluation(aggregate, spectral, contrasts), indent=2))
    _plot_milestone(output, aggregate, spectral, Path("results/rmt/default/experiment_records.csv"), config["evaluation"].get("figure_name", "milestone_1.png"))
