"""Bidirectional time-conditioned Transformer denoiser."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DenoiserConfig:
    vocab_size: int
    max_length: int = 128
    hidden_size: int = 192
    num_layers: int = 4
    num_heads: int = 4
    mlp_size: int = 768
    dropout: float = 0.0
    norm_eps: float = 1e-5

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        half = self.hidden_size // 2
        frequencies = torch.exp(-math.log(10_000.0) * torch.arange(half, device=times.device) / max(half - 1, 1))
        angles = times.float()[:, None] * frequencies[None, :] * 1000.0
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.hidden_size:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class TimeConditionedBlock(nn.Module):
    """Pre-LN bidirectional self-attention with additive time conditioning."""

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_heads, dropout=config.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_size), nn.GELU(),
            nn.Linear(config.mlp_size, config.hidden_size),
        )
        self.time_attention = nn.Linear(config.hidden_size, config.hidden_size)
        self.time_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor, time_embedding: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        conditioned = self.norm1(hidden) + self.time_attention(time_embedding)[:, None, :]
        attended, _ = self.attention(conditioned, conditioned, conditioned, key_padding_mask=padding_mask, need_weights=False)
        hidden = hidden + self.dropout(attended)
        conditioned = self.norm2(hidden) + self.time_mlp(time_embedding)[:, None, :]
        return hidden + self.dropout(self.mlp(conditioned))


class MaskedDiffusionTransformer(nn.Module):
    """A non-causal Transformer predicting clean tokens from ``(x_t,t)``."""

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size * 2), nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
        )
        self.blocks = nn.ModuleList([TimeConditionedBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        diffusion_times: torch.Tensor,
        *,
        return_hidden_states: bool = False,
        return_block_inputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError("sequence exceeds configured max_length")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        time_embedding = self.time_embedding(diffusion_times)
        padding_mask = ~attention_mask.bool()
        states: list[torch.Tensor] = []
        block_inputs: list[torch.Tensor] = []
        for block in self.blocks:
            if return_block_inputs:
                block_inputs.append(hidden)
            hidden = block(hidden, time_embedding, padding_mask)
            if return_hidden_states:
                states.append(hidden)
        hidden = self.final_norm(hidden)
        logits = torch.nn.functional.linear(hidden, self.token_embedding.weight, self.output_bias)
        if return_hidden_states and return_block_inputs:
            return logits, states, block_inputs
        return (logits, states) if return_hidden_states else logits
