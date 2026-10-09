"""Deliberately small, genuinely nonlinear denoiser (rung 4 of the bridge hierarchy).

Rungs 1-3 all approximated the masked-denoising task in an 8-dimensional
probe-projected output space. This rung instead trains an actual (but much
smaller) instance of the same architecture family as the MDLM -- a
bidirectional Transformer with tied token embeddings -- from scratch on the
real corrupted-token -> clean-token task, using the exact same corruption
process (`corrupt_tokens`) and training objective (`diffusion_loss`,
mdlm_weighted) as the reference/data_variant checkpoints. It shares the
`MaskedDiffusionTransformer` forward signature, so it can be evaluated with
the same `project_model_outputs` helper used for the real MDLM. No
pretrained MDLM parameters are touched or reused; this is a fresh, much
smaller model trained independently per split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .diffusion.corruption import corrupt_tokens
from .diffusion.losses import diffusion_loss
from .diffusion.model import DenoiserConfig, MaskedDiffusionTransformer
from .diffusion.schedule import LinearSchedule


@dataclass(frozen=True)
class SmallDenoiserTrainingConfig:
    hidden_size: int = 16
    num_layers: int = 2
    num_heads: int = 2
    mlp_size: int = 32
    dropout: float = 0.0
    steps: int = 3000
    microbatch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 5.0e-4
    minimum_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    warmup_steps: int = 150
    gradient_clip: float = 1.0
    t_min: float = 0.05
    t_max: float = 0.95
    weight_cap: float = 20.0
    seed: int = 0


def _lr_schedule(config: SmallDenoiserTrainingConfig):
    def scale(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
        return config.minimum_lr_ratio + (1.0 - config.minimum_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return scale


def fit_small_denoiser(
    token_ids: np.ndarray,
    attention_mask: np.ndarray,
    *,
    vocab_size: int,
    mask_token_id: int,
    special_ids: tuple[int, ...],
    config: SmallDenoiserTrainingConfig,
    device: torch.device,
    log_every: int | None = None,
) -> MaskedDiffusionTransformer:
    """Train a small MDLM-architecture denoiser from scratch on one split."""
    torch.manual_seed(config.seed)
    model_config = DenoiserConfig(
        vocab_size=vocab_size,
        max_length=token_ids.shape[1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_size=config.mlp_size,
        dropout=config.dropout,
    )
    model = MaskedDiffusionTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_schedule(config))

    ids_t = torch.as_tensor(token_ids, dtype=torch.long)
    attention_t = torch.as_tensor(attention_mask, dtype=torch.long)
    schedule = LinearSchedule()
    cpu_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    data_generator = torch.Generator(device="cpu").manual_seed(config.seed + 2)
    example_count = ids_t.shape[0]

    model.train()
    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        running_loss = 0.0
        for _ in range(config.gradient_accumulation):
            indices = torch.randint(0, example_count, (config.microbatch_size,), generator=data_generator)
            clean, attention = ids_t[indices], attention_t[indices]
            times = config.t_min + (config.t_max - config.t_min) * torch.rand(config.microbatch_size, generator=cpu_generator)
            corruption = corrupt_tokens(
                clean, attention, times, mask_token_id=mask_token_id,
                special_token_ids=special_ids, generator=cpu_generator, schedule=schedule,
            )
            logits = model(corruption.corrupted_ids.to(device), attention.to(device), times.to(device))
            loss, _ = diffusion_loss(
                logits, clean.to(device), corruption.target_mask.to(device), corruption.eligible_mask.to(device),
                times.to(device), mode="mdlm_weighted", weight_cap=config.weight_cap,
            )
            (loss / config.gradient_accumulation).backward()
            running_loss += loss.detach().item() / config.gradient_accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()
        if log_every and step % log_every == 0:
            print(f"    step {step}/{config.steps} loss={running_loss:.4f}", flush=True)
    model.eval()
    return model
