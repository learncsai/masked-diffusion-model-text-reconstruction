"""Predeclared context-sensitivity and hidden-rank acceptance gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from .diffusion.corruption import corrupt_tokens
from .evaluation import load_checkpoint_model
from .neural_data import ChunkDataset
from .training import select_device


def _effective_rank(states: torch.Tensor) -> tuple[float, float]:
    normalized = torch.nn.functional.layer_norm(states.float(), (states.shape[-1],)).double()
    normalized -= normalized.mean(dim=0, keepdim=True)
    covariance = normalized.T @ normalized / max(normalized.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum().clamp_min(1e-30)
    probabilities = eigenvalues / total
    effective_rank = float(torch.exp(-(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()))
    participation_ratio = float(total.square() / eigenvalues.square().sum().clamp_min(1e-30))
    return effective_rank, participation_ratio


@torch.no_grad()
def evaluate_representation_gate(config_path: Path, checkpoint_path: Path, output_path: Path) -> dict[str, Any]:
    """Evaluate a checkpoint against thresholds fixed in its YAML config."""
    config = yaml.safe_load(config_path.read_text())
    thresholds = config["representation_gate"]
    tokenizer = json.loads((Path(config["data"]["tokenizer_dir"]) / "tokenizer_metadata.json").read_text())
    dataset = ChunkDataset(Path(config["data"]["processed_dir"]) / "eval.pt", int(thresholds["evaluation_sequences"]))
    times = torch.full((len(dataset),), float(thresholds["mask_fraction"]))
    special = [tokenizer["mask_token_id"], tokenizer["pad_token_id"], tokenizer["eos_token_id"]]
    corruption = corrupt_tokens(dataset.input_ids, dataset.attention_mask, times, mask_token_id=tokenizer["mask_token_id"], special_token_ids=special, generator=torch.Generator().manual_seed(int(config["evaluation"]["seed"])), ensure_masked=True)
    device = select_device(config.get("device", "auto"))
    model = load_checkpoint_model(checkpoint_path, device)
    batch_size = int(config["evaluation"]["batch_size"])
    top_predictions: list[torch.Tensor] = []
    correct = 0; target_count = 0; negative_log_likelihood = 0.0
    context_correlations: list[torch.Tensor] = []
    permuted_correct = 0; permuted_nll = 0.0
    layer_states: dict[int, list[torch.Tensor]] = {index: [] for index in range(model.config.num_layers)}
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        ids = corruption.corrupted_ids[start:stop]; attention = dataset.attention_mask[start:stop]
        target_mask = corruption.target_mask[start:stop]; clean = dataset.input_ids[start:stop]; batch_times = times[start:stop]
        logits, hidden = model(ids.to(device), attention.to(device), batch_times.to(device), return_hidden_states=True)
        logits = logits.float().cpu(); selected_logits = logits[target_mask]; selected_clean = clean[target_mask]
        top_predictions.append(selected_logits.argmax(dim=-1))
        correct += int((selected_logits.argmax(dim=-1) == selected_clean).sum()); target_count += selected_clean.numel()
        negative_log_likelihood += float(torch.nn.functional.cross_entropy(selected_logits, selected_clean, reduction="sum"))
        # Roll visible tokens across stories while retaining each row's mask pattern.
        altered = ids.clone(); visible = attention.bool() & ~target_mask
        donor = ids.roll(shifts=1, dims=0); altered[visible] = donor[visible]
        altered_logits = model(altered.to(device), attention.to(device), batch_times.to(device)).float().cpu()[target_mask]
        permuted_correct += int((altered_logits.argmax(dim=-1) == selected_clean).sum())
        permuted_nll += float(torch.nn.functional.cross_entropy(altered_logits, selected_clean, reduction="sum"))
        centered = selected_logits - selected_logits.mean(dim=-1, keepdim=True)
        altered_centered = altered_logits - altered_logits.mean(dim=-1, keepdim=True)
        context_correlations.append(torch.nn.functional.cosine_similarity(centered, altered_centered, dim=-1))
        active = attention.bool().reshape(-1)
        for layer, state in enumerate(hidden):
            layer_states[layer].append(state.float().cpu().reshape(-1, state.shape[-1])[active])
    predictions = torch.cat(top_predictions); _, counts = torch.unique(predictions, return_counts=True)
    ranks = {}
    for layer, chunks in layer_states.items():
        effective, participation = _effective_rank(torch.cat(chunks))
        ranks[str(layer)] = {"normalized_effective_rank": effective, "normalized_participation_ratio": participation}
    metrics = {
        "checkpoint": str(checkpoint_path), "parameter_count": model.parameter_count,
        "evaluation_sequences": len(dataset), "masked_tokens": target_count,
        "unique_top1_tokens": int(counts.numel()),
        "dominant_top1_fraction": float(counts.max() / counts.sum()),
        "context_permutation_centered_logit_correlation": float(torch.cat(context_correlations).mean()),
        "heldout_masked_ce": negative_log_likelihood / target_count,
        "heldout_accuracy": correct / target_count,
        "permuted_context_masked_ce": permuted_nll / target_count,
        "permuted_context_accuracy": permuted_correct / target_count,
        "layers": ranks,
    }
    if "maximum_context_replacement_correlation" in thresholds:
        context_check = metrics["context_permutation_centered_logit_correlation"] <= float(thresholds["maximum_context_replacement_correlation"])
    else:
        accuracy_retention = metrics["permuted_context_accuracy"] / max(metrics["heldout_accuracy"], 1e-30)
        ce_increase = metrics["permuted_context_masked_ce"] - metrics["heldout_masked_ce"]
        context_check = accuracy_retention <= float(thresholds["maximum_permuted_context_accuracy_retention"]) and ce_increase >= float(thresholds["minimum_permuted_context_ce_increase"])
        metrics["permuted_context_accuracy_retention"] = accuracy_retention
        metrics["permuted_context_ce_increase"] = ce_increase
    checks = {
        "unique_top1_tokens": metrics["unique_top1_tokens"] >= int(thresholds["minimum_unique_top1_tokens"]),
        "dominant_top1_fraction": metrics["dominant_top1_fraction"] <= float(thresholds["maximum_dominant_top1_fraction"]),
        "context_sensitivity": context_check,
        "hidden_effective_rank": max(value["normalized_effective_rank"] for value in ranks.values()) >= float(thresholds["minimum_normalized_effective_rank"]),
        "heldout_masked_ce": metrics["heldout_masked_ce"] <= float(thresholds["maximum_heldout_masked_ce"]),
        "heldout_accuracy": metrics["heldout_accuracy"] >= float(thresholds["minimum_heldout_accuracy"]),
    }
    result = {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds, "metrics": metrics}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result
