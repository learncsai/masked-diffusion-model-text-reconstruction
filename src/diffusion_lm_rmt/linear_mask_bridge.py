"""Direct linear-masked-denoiser bridge for MDLM output consistency.

Token distributions are sketched with fixed Gaussian vocabulary probes.  For
one probe, a clean length-L token sequence becomes an L-dimensional vector,
and the saved MDLM target mask is exactly a coordinate mask on that vector.
Expected probe values under the MDLM softmax and linear reconstructions are
therefore in the same units.  Averaging squared differences over independent
Gaussian probes is an unbiased random projection of probability-space L2
disagreement.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from .rmt.core import (
    masked_corrupted_covariance,
    masked_linear_denoiser_operator,
    masking_perturbative_split_prediction,
    perturbative_split_prediction,
    reduce_perturbative_prediction,
    resolvent_filter,
)


def gaussian_token_probes(
    vocab_size: int,
    count: int,
    *,
    seed: int,
    excluded_ids: tuple[int, ...] = (),
) -> np.ndarray:
    """Return reproducible N(0,1) vocabulary probes."""
    if vocab_size <= 0 or count <= 0:
        raise ValueError("vocab_size and probe count must be positive")
    probes = np.random.default_rng(seed).standard_normal((count, vocab_size))
    for token_id in excluded_ids:
        probes[:, int(token_id)] = 0.0
    return probes


def split_feature_statistics(
    token_ids: np.ndarray,
    probes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return probe-coded sequences, per-position means, and covariances."""
    coded = probes[:, token_ids].transpose(1, 2, 0).astype(np.float64, copy=False)
    means = coded.mean(axis=0).T
    covariances = np.empty((probes.shape[0], token_ids.shape[1], token_ids.shape[1]), dtype=np.float64)
    for probe in range(probes.shape[0]):
        centered = coded[:, :, probe] - means[probe]
        covariances[probe] = centered.T @ centered / centered.shape[0]
    return coded, means, covariances


@torch.no_grad()
def project_model_outputs(
    model: torch.nn.Module,
    corrupted_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    times: torch.Tensor,
    probes: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    excluded_ids: tuple[int, ...] = (),
) -> np.ndarray:
    """Project softmax outputs without retaining vocabulary-sized tensors."""
    probe_tensor = torch.as_tensor(probes.T, dtype=torch.float32, device=device)
    parts: list[np.ndarray] = []
    for start in range(0, corrupted_ids.shape[0], batch_size):
        stop = min(start + batch_size, corrupted_ids.shape[0])
        logits = model(
            corrupted_ids[start:stop].to(device),
            attention_mask[start:stop].to(device),
            times[start:stop].to(device),
        ).float()
        for token_id in excluded_ids:
            logits[..., int(token_id)] = -torch.inf
        probabilities = torch.softmax(logits, dim=-1)
        projected = probabilities @ probe_tensor
        parts.append(projected.cpu().numpy())
        del logits, probabilities, projected
    return np.concatenate(parts, axis=0).astype(np.float64, copy=False)


def _safe_correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float:
    left, right = np.asarray(left), np.asarray(right)
    usable = np.isfinite(left) & np.isfinite(right)
    if usable.sum() < 3 or np.ptp(left[usable]) == 0 or np.ptp(right[usable]) == 0:
        return float("nan")
    statistic = spearmanr(left[usable], right[usable]).statistic if rank else pearsonr(left[usable], right[usable]).statistic
    return float(statistic)


def _masked_sequence_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = mask[..., None]
    return np.sum(values * weights, axis=(1, 2)) / np.maximum(np.sum(weights, axis=(1, 2)) * values.shape[2], 1)


def _masked_flat(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return values[np.broadcast_to(mask[..., None], values.shape)]


def _observable_rows(
    *,
    covariance: np.ndarray,
    covariance_a: np.ndarray,
    covariance_b: np.ndarray,
    q: float,
    n_a: int,
    n_b: int,
    input_covariance: np.ndarray,
    probe: int,
    base: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = np.maximum(eigenvalues[order], 0.0), eigenvectors[:, order]
    exact = masking_perturbative_split_prediction(
        covariance, q, n_a, n_b, basis=eigenvectors, input_covariance=input_covariance,
    )
    c = float(eigenvalues.mean())
    gamma = c * (1.0 - q) / q
    simplified = perturbative_split_prediction(eigenvalues, gamma, n_a, n_b)
    input_in_basis = eigenvectors.T @ input_covariance @ eigenvectors

    exact_a = masked_linear_denoiser_operator(covariance_a, q)
    exact_b = masked_linear_denoiser_operator(covariance_b, q)
    empirical_difference = eigenvectors.T @ (exact_a - exact_b) @ eigenvectors
    empirical_entrywise = empirical_difference**2
    simplified_a = resolvent_filter(covariance_a, gamma)
    simplified_b = resolvent_filter(covariance_b, gamma)
    approximation_error = 0.5 * (
        np.linalg.norm(exact_a - simplified_a) / max(np.linalg.norm(exact_a), 1e-30)
        + np.linalg.norm(exact_b - simplified_b) / max(np.linalg.norm(exact_b), 1e-30)
    )

    exact_modes = {
        observable: reduce_perturbative_prediction(exact, observable=observable).per_output_mode
        for observable in ("diagonal_operator", "rowsum_operator", "prediction_covariance_weighted")
    }
    simplified_modes = {
        observable: reduce_perturbative_prediction(
            simplified,
            observable=observable,
            input_covariance_in_eigenbasis=(
                input_in_basis if observable == "prediction_covariance_weighted" else None
            ),
        ).per_output_mode
        for observable in ("diagonal_operator", "rowsum_operator", "prediction_covariance_weighted")
    }
    empirical_modes = {
        "diagonal_operator": np.diag(empirical_entrywise),
        "rowsum_operator": empirical_entrywise.sum(axis=1),
        "prediction_covariance_weighted": np.diag(
            eigenvectors.T @ (exact_a - exact_b) @ input_covariance @ (exact_a - exact_b).T @ eigenvectors
        ),
    }
    aggregate_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    for observable in exact_modes:
        exact_values = np.asarray(exact_modes[observable])
        simple_values = np.asarray(simplified_modes[observable])
        empirical_values = np.asarray(empirical_modes[observable])
        exact_aggregate = float(exact_values.mean())
        simple_aggregate = float(simple_values.mean())
        aggregate_rows.append({
            **base,
            "probe": probe,
            "observable": observable,
            "exact_masking_prediction": exact_aggregate,
            "scalar_gamma_prediction": simple_aggregate,
            "exact_to_scalar_ratio": exact_aggregate / max(simple_aggregate, 1e-30),
            "empirical_split_operator": float(empirical_values.mean()),
            "empirical_to_exact_ratio": float(empirical_values.mean()) / max(exact_aggregate, 1e-30),
            "mean_diagonal_variance": float(np.diag(covariance).mean()),
            "diagonal_coefficient_of_variation": float(np.std(np.diag(covariance)) / max(np.mean(np.diag(covariance)), 1e-30)),
            "gamma": gamma,
            "exact_vs_scalar_operator_relative_error": float(approximation_error),
        })
        for mode in range(eigenvalues.size):
            mode_rows.append({
                **base,
                "probe": probe,
                "mode": mode,
                "eigenvalue": float(eigenvalues[mode]),
                "observable": observable,
                "exact_masking_prediction": float(exact_values[mode]),
                "scalar_gamma_prediction": float(simple_values[mode]),
                "empirical_split_operator": float(empirical_values[mode]),
            })
    return aggregate_rows, mode_rows, eigenvectors


def evaluate_condition(
    *,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    train_a_ids: np.ndarray,
    train_b_ids: np.ndarray,
    corruptions: dict[str, Any],
    probes: np.ndarray,
    mask_fraction: float,
    device: torch.device,
    batch_size: int,
    excluded_ids: tuple[int, ...],
    base: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate one replicate/n/mask-rate condition on paired inputs."""
    _, means_a, covariances_a = split_feature_statistics(train_a_ids, probes)
    _, means_b, covariances_b = split_feature_statistics(train_b_ids, probes)
    eval_ids = corruptions["input_ids"].cpu().numpy()
    eval_coded = probes[:, eval_ids].transpose(1, 2, 0).astype(np.float64, copy=False)
    data = corruptions["fractions"][str(mask_fraction)]
    target_mask = data["target_mask"].cpu().numpy().astype(bool)
    visible = (~target_mask).astype(np.float64)
    q = 1.0 - mask_fraction

    neural_a = project_model_outputs(
        model_a, data["corrupted_ids"], corruptions["attention_mask"], data["times"], probes,
        device=device, batch_size=batch_size, excluded_ids=excluded_ids,
    )
    neural_b = project_model_outputs(
        model_b, data["corrupted_ids"], corruptions["attention_mask"], data["times"], probes,
        device=device, batch_size=batch_size, excluded_ids=excluded_ids,
    )

    linear_a = np.empty_like(neural_a)
    linear_b = np.empty_like(neural_b)
    aggregate_rows: list[dict[str, Any]] = []
    rmt_mode_rows: list[dict[str, Any]] = []
    bases: list[np.ndarray] = []
    for probe in range(probes.shape[0]):
        operator_a = masked_linear_denoiser_operator(covariances_a[probe], q)
        operator_b = masked_linear_denoiser_operator(covariances_b[probe], q)
        centered_a = (eval_coded[:, :, probe] - means_a[probe]) * visible / q
        centered_b = (eval_coded[:, :, probe] - means_b[probe]) * visible / q
        linear_a[:, :, probe] = centered_a @ operator_a.T + means_a[probe]
        linear_b[:, :, probe] = centered_b @ operator_b.T + means_b[probe]
        reference_covariance = 0.5 * (covariances_a[probe] + covariances_b[probe])
        reference_corrupted = 0.5 * (centered_a + centered_b)
        input_covariance = reference_corrupted.T @ reference_corrupted / reference_corrupted.shape[0]
        one_aggregate, one_modes, basis = _observable_rows(
            covariance=reference_covariance,
            covariance_a=covariances_a[probe],
            covariance_b=covariances_b[probe],
            q=q,
            n_a=train_a_ids.shape[0],
            n_b=train_b_ids.shape[0],
            input_covariance=input_covariance,
            probe=probe,
            base=base,
        )
        aggregate_rows.extend(one_aggregate)
        rmt_mode_rows.extend(one_modes)
        bases.append(basis)

    neural_difference = neural_a - neural_b
    linear_difference = linear_a - linear_b
    neural_sequence = _masked_sequence_mean(neural_difference**2, target_mask)
    linear_sequence = _masked_sequence_mean(linear_difference**2, target_mask)
    direct_a_sequence = _masked_sequence_mean((neural_a - linear_a) ** 2, target_mask)
    direct_b_sequence = _masked_sequence_mean((neural_b - linear_b) ** 2, target_mask)
    truth = eval_coded
    neural_target_mse = 0.5 * (
        _masked_sequence_mean((neural_a - truth) ** 2, target_mask)
        + _masked_sequence_mean((neural_b - truth) ** 2, target_mask)
    )
    linear_target_mse = 0.5 * (
        _masked_sequence_mean((linear_a - truth) ** 2, target_mask)
        + _masked_sequence_mean((linear_b - truth) ** 2, target_mask)
    )
    sequence_rows = [{
        **base,
        "sequence_index": index,
        "masked_tokens": int(target_mask[index].sum()),
        "neural_cross_split_disagreement": float(neural_sequence[index]),
        "linear_cross_split_disagreement": float(linear_sequence[index]),
        "model_a_vs_linear_mse": float(direct_a_sequence[index]),
        "model_b_vs_linear_mse": float(direct_b_sequence[index]),
        "neural_target_mse": float(neural_target_mse[index]),
        "linear_target_mse": float(linear_target_mse[index]),
    } for index in range(eval_ids.shape[0])]

    neural_flat = _masked_flat(neural_difference, target_mask)
    linear_flat = _masked_flat(linear_difference, target_mask)
    alignment = float(np.dot(neural_flat, linear_flat) / max(np.linalg.norm(neural_flat) * np.linalg.norm(linear_flat), 1e-30))
    summary = [{
        **base,
        "evaluation_sequences": eval_ids.shape[0],
        "masked_tokens": int(target_mask.sum()),
        "probes": probes.shape[0],
        "neural_cross_split_disagreement": float(neural_sequence.mean()),
        "linear_cross_split_disagreement": float(linear_sequence.mean()),
        "linear_to_neural_disagreement_ratio": float(linear_sequence.mean() / max(neural_sequence.mean(), 1e-30)),
        "inputwise_pearson": _safe_correlation(neural_sequence, linear_sequence),
        "inputwise_spearman": _safe_correlation(neural_sequence, linear_sequence, rank=True),
        "cross_split_difference_cosine": alignment,
        "model_a_vs_linear_mse": float(direct_a_sequence.mean()),
        "model_b_vs_linear_mse": float(direct_b_sequence.mean()),
        "neural_target_mse": float(neural_target_mse.mean()),
        "linear_target_mse": float(linear_target_mse.mean()),
        "direct_output_pearson_a": _safe_correlation(_masked_flat(neural_a, target_mask), _masked_flat(linear_a, target_mask)),
        "direct_output_pearson_b": _safe_correlation(_masked_flat(neural_b, target_mask), _masked_flat(linear_b, target_mask)),
    }]

    spectral_rows: list[dict[str, Any]] = []
    neural_spectrum = np.zeros(eval_ids.shape[1])
    linear_spectrum = np.zeros(eval_ids.shape[1])
    for probe, basis in enumerate(bases):
        masked_neural = neural_difference[:, :, probe] * target_mask
        masked_linear = linear_difference[:, :, probe] * target_mask
        one_neural = np.mean((masked_neural @ basis) ** 2, axis=0)
        one_linear = np.mean((masked_linear @ basis) ** 2, axis=0)
        neural_spectrum += one_neural / probes.shape[0]
        linear_spectrum += one_linear / probes.shape[0]
    for mode in range(neural_spectrum.size):
        spectral_rows.append({
            **base,
            "mode": mode,
            "neural_disagreement": float(neural_spectrum[mode]),
            "linear_disagreement": float(linear_spectrum[mode]),
        })
    positive = (neural_spectrum > 0) & (linear_spectrum > 0)
    summary[0]["spectral_log_pearson"] = _safe_correlation(
        np.log(neural_spectrum[positive]), np.log(linear_spectrum[positive]),
    )
    return {
        "summary": summary,
        "sequences": sequence_rows,
        "spectral": spectral_rows,
        "rmt_aggregates": aggregate_rows,
        "rmt_modes": rmt_mode_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def combine_markers(output_dir: Path) -> dict[str, Any]:
    """Combine resumable per-condition JSON markers into stable tables."""
    categories = ("summary", "sequences", "spectral", "rmt_aggregates", "rmt_modes")
    combined: dict[str, list[dict[str, Any]]] = {category: [] for category in categories}
    markers = sorted((output_dir / "raw").glob("r*_n*_q*.json"))
    if not markers:
        raise FileNotFoundError("no linear-mask bridge markers found")
    for marker in markers:
        payload = json.loads(marker.read_text())
        for category in categories:
            combined[category].extend(payload[category])
    names = {
        "summary": "linear_mask_bridge_summary.csv",
        "sequences": "linear_mask_bridge_per_sequence.csv",
        "spectral": "linear_mask_bridge_spectral.csv",
        "rmt_aggregates": "masking_rmt_observable_comparison.csv",
        "rmt_modes": "masking_rmt_observable_modes.csv",
    }
    for category, filename in names.items():
        write_csv(output_dir / filename, combined[category])
    metadata = {"condition_markers": len(markers), **{f"{key}_rows": len(value) for key, value in combined.items()}}
    (output_dir / "linear_mask_bridge_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
