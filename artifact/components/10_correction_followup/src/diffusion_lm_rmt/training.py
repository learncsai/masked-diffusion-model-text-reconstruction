"""Resumable, explicitly seeded MDLM training pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml

from .diffusion.corruption import corrupt_tokens
from .diffusion.losses import diffusion_loss
from .diffusion.model import DenoiserConfig, MaskedDiffusionTransformer
from .neural_data import ChunkDataset


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, then MPS, then CPU unless explicitly requested."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _model_config(config: dict[str, Any], vocab_size: int) -> DenoiserConfig:
    return DenoiserConfig(vocab_size=vocab_size, **config["model"])


def _batch(dataset: ChunkDataset, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return dataset.input_ids[indices], dataset.attention_mask[indices]


def _sample_times(batch_size: int, config: dict[str, Any], generator: torch.Generator) -> torch.Tensor:
    low, high = float(config["diffusion"]["t_min"]), float(config["diffusion"]["t_max"])
    return low + (high - low) * torch.rand(batch_size, generator=generator)


@torch.no_grad()
def validation_metrics(
    model: MaskedDiffusionTransformer,
    dataset: ChunkDataset,
    config: dict[str, Any],
    *,
    mask_token_id: int,
    special_ids: list[int],
    device: torch.device,
    seed: int = 99173,
) -> dict[str, float]:
    """Evaluate on a fixed corruption stream independent of training state."""
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = int(config["training"]["microbatch_size"])
    losses, cross_entropies, accuracies = [], [], []
    for batch_index in range(int(config["training"]["validation_batches"])):
        indices = torch.arange(batch_index * batch_size, (batch_index + 1) * batch_size) % len(dataset)
        clean, attention = _batch(dataset, indices)
        times = _sample_times(batch_size, config, generator)
        corruption = corrupt_tokens(clean, attention, times, mask_token_id=mask_token_id, special_token_ids=special_ids, generator=generator)
        logits = model(corruption.corrupted_ids.to(device), attention.to(device), times.to(device))
        loss, metrics = diffusion_loss(logits, clean.to(device), corruption.target_mask.to(device), corruption.eligible_mask.to(device), times.to(device), mode=config["diffusion"]["objective"], weight_cap=config["diffusion"].get("weight_cap"))
        losses.append(float(loss)); cross_entropies.append(metrics["masked_ce"]); accuracies.append(metrics["masked_accuracy"])
    model.train()
    return {"validation_loss": sum(losses) / len(losses), "validation_masked_ce": sum(cross_entropies) / len(cross_entropies), "validation_accuracy": sum(accuracies) / len(accuracies)}


def _write_metadata(run_dir: Path, config: dict[str, Any], source_config: Path, data_dir: Path, tokenizer_dir: Path, model: MaskedDiffusionTransformer, device: torch.device, seeds: dict[str, int]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_config, run_dir / "config.yaml")
    shutil.copy2(data_dir / "split_manifest.json", run_dir / "split_manifest.json")
    shutil.copy2(tokenizer_dir / "tokenizer_metadata.json", run_dir / "tokenizer_metadata.json")
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        commit = result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"
        commit = commit or "UNCOMMITTED"
    except OSError:
        commit = "UNAVAILABLE"
    (run_dir / "git_commit.txt").write_text(commit + "\n")
    system = {
        "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
        "torch": torch.__version__, "device": str(device), "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(), "physical_memory_bytes": psutil.virtual_memory().total,
        "parameter_count": model.parameter_count, "initial_state_sha256": state_dict_sha256(model.state_dict()),
        "seeds": seeds, "model_config": model.config.to_dict(), "objective": config["diffusion"]["objective"],
    }
    (run_dir / "system_info.json").write_text(json.dumps(system, indent=2))


def train_run(
    config_path: Path,
    *,
    run_dir: Path,
    train_split: str,
    init_seed: int,
    training_seed: int,
    corruption_seed: int,
    resume: bool = False,
    initialize_from: Path | None = None,
) -> dict[str, Any]:
    """Train one controlled run and save a complete resumable checkpoint."""
    config = yaml.safe_load(config_path.read_text())
    data_dir, tokenizer_dir = Path(config["data"]["processed_dir"]), Path(config["data"]["tokenizer_dir"])
    tokenizer_metadata = json.loads((tokenizer_dir / "tokenizer_metadata.json").read_text())
    dataset = ChunkDataset(data_dir / f"{train_split}.pt", config["training"].get("train_chunk_limit"))
    validation = ChunkDataset(data_dir / "eval.pt")
    if len(dataset) == 0:
        raise RuntimeError("empty training split")
    device = select_device(config.get("device", "auto"))
    torch.manual_seed(init_seed)
    model = MaskedDiffusionTransformer(_model_config(config, tokenizer_metadata["vocab_size"])).to(device)
    parent_checkpoint = None
    if initialize_from is not None:
        parent_checkpoint = torch.load(initialize_from, map_location="cpu", weights_only=False)
        if parent_checkpoint["model_config"] != model.config.to_dict():
            raise ValueError("parent checkpoint model configuration does not match")
        model.load_state_dict(parent_checkpoint["model"])
    seeds = {"data_seed": config["data"]["seed"], "initialization_seed": init_seed, "training_seed": training_seed, "corruption_seed": corruption_seed}
    _write_metadata(run_dir, config, config_path, data_dir, tokenizer_dir, model, device, seeds)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimizer_config["learning_rate"]), betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])), weight_decay=float(optimizer_config["weight_decay"]))
    total_steps, warmup = int(optimizer_config["total_steps"]), int(optimizer_config["warmup_steps"])
    minimum_lr_ratio = float(optimizer_config.get("minimum_lr_ratio", 0.0))
    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    data_generator = torch.Generator(device="cpu").manual_seed(training_seed)
    corruption_generator = torch.Generator(device="cpu").manual_seed(corruption_seed)
    if parent_checkpoint is not None:
        data_generator.set_state(parent_checkpoint["data_generator_state"])
        corruption_generator.set_state(parent_checkpoint["corruption_generator_state"])
    start_step = 0
    tokens_seen = 0
    checkpoint_path = run_dir / "checkpoint.pt"
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"])
        data_generator.set_state(checkpoint["data_generator_state"]); corruption_generator.set_state(checkpoint["corruption_generator_state"])
        start_step, tokens_seen = checkpoint["step"], checkpoint["tokens_seen"]

    special_ids = [tokenizer_metadata["mask_token_id"], tokenizer_metadata["pad_token_id"], tokenizer_metadata["eos_token_id"]]
    microbatch = int(config["training"]["microbatch_size"])
    accumulation = int(config["training"]["gradient_accumulation"])
    metrics_path = run_dir / "metrics.jsonl"
    curve_rows: list[dict[str, Any]] = []
    if not resume:
        metrics_path.write_text("", encoding="utf-8")
    if (run_dir / "training_curve.csv").exists() and resume:
        with (run_dir / "training_curve.csv").open() as handle:
            curve_rows = list(csv.DictReader(handle))
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    peak_rss = process.memory_info().rss

    def save_checkpoint(step: int) -> None:
        torch.save({"model": model.state_dict(), "model_config": model.config.to_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "tokens_seen": tokens_seen, "data_generator_state": data_generator.get_state(), "corruption_generator_state": corruption_generator.get_state(), "seeds": seeds, "train_split": train_split, "parent_checkpoint": str(initialize_from) if initialize_from else None}, checkpoint_path)

    model.train()
    for step in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0; step_accuracy = 0.0; step_tokens = 0
        for _ in range(accumulation):
            indices = torch.randint(len(dataset), (microbatch,), generator=data_generator)
            clean, attention = _batch(dataset, indices)
            times = _sample_times(microbatch, config, corruption_generator)
            corrupted = corrupt_tokens(clean, attention, times, mask_token_id=tokenizer_metadata["mask_token_id"], special_token_ids=special_ids, generator=corruption_generator)
            logits = model(corrupted.corrupted_ids.to(device), attention.to(device), times.to(device))
            loss, batch_metrics = diffusion_loss(logits, clean.to(device), corrupted.target_mask.to(device), corrupted.eligible_mask.to(device), times.to(device), mode=config["diffusion"]["objective"], weight_cap=config["diffusion"].get("weight_cap"))
            if not torch.isfinite(loss):
                save_checkpoint(step); raise FloatingPointError(f"non-finite loss at step {step}")
            (loss / accumulation).backward()
            step_loss += float(loss.detach()) / accumulation
            step_accuracy += batch_metrics["masked_accuracy"] / accumulation
            non_padding = int(attention.sum().item()); step_tokens += non_padding; tokens_seen += non_padding
            del logits, loss
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimizer_config["gradient_clip"])))
        if not math.isfinite(gradient_norm):
            save_checkpoint(step); raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step(); scheduler.step()
        peak_rss = max(peak_rss, process.memory_info().rss)
        elapsed = time.perf_counter() - start_time
        row: dict[str, Any] = {"step": step + 1, "training_loss": step_loss, "training_accuracy": step_accuracy, "gradient_norm": gradient_norm, "learning_rate": scheduler.get_last_lr()[0], "tokens_seen": tokens_seen, "tokens_per_second": tokens_seen / max(elapsed, 1e-9), "peak_rss_bytes": peak_rss}
        if (step + 1) % int(config["training"]["validation_interval"]) == 0 or step == start_step:
            row.update(validation_metrics(model, validation, config, mask_token_id=tokenizer_metadata["mask_token_id"], special_ids=special_ids, device=device))
        curve_rows.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if (step + 1) % int(config["training"]["checkpoint_interval"]) == 0:
            save_checkpoint(step + 1)
    save_checkpoint(total_steps)
    keys = sorted({key for row in curve_rows for key in row})
    with (run_dir / "training_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(curve_rows)
    summary = {"run_dir": str(run_dir), "train_split": train_split, "steps": total_steps, "chunks_n": len(dataset), "documents": len({record["document_id"] for record in dataset.chunk_records}), "tokens_seen": tokens_seen, "cumulative_tokens_seen": tokens_seen + (int(parent_checkpoint["tokens_seen"]) if parent_checkpoint else 0), "parent_checkpoint": str(initialize_from) if initialize_from else None, "parameter_count": model.parameter_count, "device": str(device), "elapsed_seconds": time.perf_counter() - start_time, "tokens_per_second": tokens_seen / max(time.perf_counter() - start_time, 1e-9), "peak_rss_bytes": peak_rss, "initial_state_sha256": json.loads((run_dir / "system_info.json").read_text())["initial_state_sha256"], "final_training_loss": curve_rows[-1]["training_loss"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
