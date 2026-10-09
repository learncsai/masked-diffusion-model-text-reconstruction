"""Input-dependent (attention) sequence denoiser for the bridge-test hierarchy.

The sequence-linear bridge (``sequence_linear_bridge.py``) fits one fixed
global linear map from masked probe codes to clean probe codes. This module
tests whether letting the mixing across positions depend on the corrupted
input content -- via a single bidirectional multi-head self-attention layer,
the minimal input-dependent generalization of that linear map -- closes any
of the remaining gap to the frozen trained MDLM. It is a small model trained
from scratch with gradient descent; no MDLM parameters are touched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def masked_position_features(coded_tokens: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    """Per-position visible probe codes with an appended mask indicator."""
    coded_tokens = np.asarray(coded_tokens, dtype=np.float32)
    target_mask = np.asarray(target_mask, dtype=bool)
    if coded_tokens.ndim != 3 or target_mask.shape != coded_tokens.shape[:2]:
        raise ValueError("coded tokens must be (sequences, positions, features) with a matching mask")
    visible_codes = coded_tokens * (~target_mask)[..., None]
    return np.concatenate([visible_codes, target_mask.astype(np.float32)[..., None]], axis=-1)


class AttentionBridgeDenoiser(nn.Module):
    """One bidirectional self-attention layer mixing positions conditionally on the input."""

    def __init__(
        self,
        *,
        feature_count: int,
        sequence_length: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        mlp_hidden: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.sequence_length = sequence_length
        self.input_proj = nn.Linear(feature_count + 1, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(sequence_length, embed_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, embed_dim),
        )
        self.readout = nn.Linear(embed_dim, feature_count)

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features) + self.position_embedding[: features.shape[1]]
        key_padding_mask = ~attention_mask.bool()
        attention_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + attention_out)
        x = self.norm2(x + self.mlp(x))
        return self.readout(x)

    @torch.no_grad()
    def predict(
        self, coded_tokens: np.ndarray, target_mask: np.ndarray, attention_mask: np.ndarray, *, batch_size: int = 128,
    ) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        features = masked_position_features(coded_tokens, target_mask)
        outputs = []
        for start in range(0, features.shape[0], batch_size):
            batch_features = torch.from_numpy(features[start : start + batch_size]).to(device)
            batch_attention = torch.from_numpy(attention_mask[start : start + batch_size]).to(device)
            outputs.append(self(batch_features, batch_attention).cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float64, copy=False)


@dataclass(frozen=True)
class AttentionBridgeTrainingConfig:
    embed_dim: int = 32
    num_heads: int = 4
    mlp_hidden: int = 64
    dropout: float = 0.0
    epochs: int = 150
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    batch_size: int = 256
    seed: int = 0


def fit_attention_bridge_denoiser(
    coded_tokens: np.ndarray,
    masks: np.ndarray,
    attention_mask: np.ndarray,
    *,
    config: AttentionBridgeTrainingConfig,
    device: torch.device,
) -> AttentionBridgeDenoiser:
    """Train a single-attention-layer denoiser on the same mask tape used for ridge fitting."""
    coded_tokens = np.asarray(coded_tokens, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    attention_mask = np.asarray(attention_mask, dtype=np.float32)
    if coded_tokens.ndim != 3 or masks.ndim != 3 or masks.shape[1:] != coded_tokens.shape[:2]:
        raise ValueError("masks must have shape (augmentations, sequences, positions)")

    torch.manual_seed(config.seed)
    model = AttentionBridgeDenoiser(
        feature_count=coded_tokens.shape[2],
        sequence_length=coded_tokens.shape[1],
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        mlp_hidden=config.mlp_hidden,
        dropout=config.dropout,
    ).to(device)

    features = np.concatenate(
        [masked_position_features(coded_tokens, mask) for mask in masks], axis=0,
    )
    targets = np.tile(coded_tokens, (masks.shape[0], 1, 1))
    tiled_attention = np.tile(attention_mask, (masks.shape[0], 1))
    features_t = torch.from_numpy(features)
    targets_t = torch.from_numpy(targets.astype(np.float32, copy=False))
    attention_t = torch.from_numpy(tiled_attention)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    num_examples = features_t.shape[0]
    model.train()
    for _ in range(config.epochs):
        permutation = torch.randperm(num_examples, generator=generator)
        for start in range(0, num_examples, config.batch_size):
            indices = permutation[start : start + config.batch_size]
            batch_features = features_t[indices].to(device)
            batch_targets = targets_t[indices].to(device)
            batch_attention = attention_t[indices].to(device)
            prediction = model(batch_features, batch_attention)
            loss = torch.mean((prediction - batch_targets) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model
