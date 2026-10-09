"""Local Jacobian surrogate for the frozen trained MDLM (rung 3 of the bridge hierarchy).

Earlier bridges fit an *approximate* linear or attention model to the
denoising task. This module instead treats the actual trained network as its
own local linear surrogate around a fixed reference point:

    D(z0 + dz) - D(z0) ~= J(z0) dz

where ``z0`` is the token embedding of the fully masked ("all noise") input
and ``J(z0)`` is the exact Jacobian of the probe-projected softmax output
with respect to those embeddings, computed by autodiff on the frozen
checkpoint. No MDLM parameters are changed. The model is tiny (~5M
parameters, hidden size 96, 2 layers) so the full Jacobian is cheap to
materialize exactly via ``torch.func.jacrev`` rather than approximated with
randomized JVPs/power iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jacrev


def forward_from_embeddings(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    diffusion_times: torch.Tensor,
) -> torch.Tensor:
    """Replicate ``MaskedDiffusionTransformer.forward`` starting from continuous embeddings."""
    positions = torch.arange(embeddings.shape[1], device=embeddings.device)
    hidden = embeddings + model.position_embedding(positions)[None, :, :]
    time_embedding = model.time_embedding(diffusion_times)
    padding_mask = ~attention_mask.bool()
    for block in model.blocks:
        hidden = block(hidden, time_embedding, padding_mask)
    hidden = model.final_norm(hidden)
    return F.linear(hidden, model.token_embedding.weight, model.output_bias)


def probe_output_from_embeddings(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    diffusion_times: torch.Tensor,
    probe_tensor: torch.Tensor,
    exclude_mask: torch.Tensor,
) -> torch.Tensor:
    logits = forward_from_embeddings(model, embeddings, attention_mask, diffusion_times)
    logits = logits.masked_fill(exclude_mask, -1.0e9)
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities @ probe_tensor


@dataclass(frozen=True)
class LocalJacobianOperator:
    base_embeddings: np.ndarray  # (L, H)
    base_output: np.ndarray  # (L, P)
    jacobian: np.ndarray  # (L*P, L*H)
    sequence_length: int
    hidden_size: int
    probe_count: int

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """First-order Taylor prediction of the probe output at ``embeddings``."""
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 3 or embeddings.shape[1:] != (self.sequence_length, self.hidden_size):
            raise ValueError("embeddings must have shape (sequences, sequence_length, hidden_size)")
        delta = embeddings.reshape(embeddings.shape[0], -1) - self.base_embeddings.reshape(1, -1)
        correction = delta @ self.jacobian.T
        prediction = self.base_output.reshape(1, -1) + correction
        return prediction.reshape(embeddings.shape[0], self.sequence_length, self.probe_count)

    def singular_values(self) -> np.ndarray:
        return np.linalg.svd(self.jacobian, compute_uv=False)


def fit_local_jacobian_operator(
    model: torch.nn.Module,
    base_ids: np.ndarray,
    attention_mask: np.ndarray,
    diffusion_time: float,
    probes: np.ndarray,
    *,
    device: torch.device,
    excluded_ids: tuple[int, ...] = (),
) -> LocalJacobianOperator:
    """Compute the exact local Jacobian of the frozen model's probe output at ``base_ids``."""
    model = model.to(device)
    base_ids_t = torch.as_tensor(np.asarray(base_ids), dtype=torch.long, device=device)[None, :]
    attention_t = torch.as_tensor(np.asarray(attention_mask), dtype=torch.float32, device=device)[None, :]
    time_t = torch.as_tensor([diffusion_time], dtype=torch.float32, device=device)
    probe_tensor = torch.as_tensor(probes.T, dtype=torch.float32, device=device)
    exclude_mask = torch.zeros(probes.shape[1], dtype=torch.bool, device=device)
    for token_id in excluded_ids:
        exclude_mask[int(token_id)] = True

    with torch.no_grad():
        base_embeddings = model.token_embedding(base_ids_t)

    def probe_output(embeddings: torch.Tensor) -> torch.Tensor:
        return probe_output_from_embeddings(model, embeddings, attention_t, time_t, probe_tensor, exclude_mask)[0]

    jacobian = jacrev(probe_output)(base_embeddings)
    length, probe_count = jacobian.shape[0], jacobian.shape[1]
    jacobian = jacobian.reshape(length * probe_count, -1)
    with torch.no_grad():
        base_output = probe_output(base_embeddings)

    return LocalJacobianOperator(
        base_embeddings=base_embeddings[0].detach().cpu().numpy().astype(np.float64, copy=False),
        base_output=base_output.detach().cpu().numpy().astype(np.float64, copy=False),
        jacobian=jacobian.detach().cpu().numpy().astype(np.float64, copy=False),
        sequence_length=length,
        hidden_size=base_embeddings.shape[-1],
        probe_count=probe_count,
    )


def per_probe_induced_operators(jacobian: np.ndarray, probe_count: int) -> np.ndarray:
    """Reduce a rectangular (position*probe, position*hidden) Jacobian to per-probe,
    position-by-position induced output covariances under isotropic input noise.

    ``J`` maps each model's own (unaligned across models) embedding space to the
    shared probe-projected output space, so ``J_a - J_b`` is not directly
    meaningful entrywise. ``M^(r) = J^(r) (J^(r))^T`` -- the covariance the r-th
    probe's 64-position output block would have under an isotropic input
    perturbation -- lives entirely in the shared output space and is invariant
    to any orthogonal reparameterization of the (arbitrary, unaligned) input
    embedding coordinates, so ``M_a^(r)`` and ``M_b^(r)`` are directly
    comparable with the existing square-operator RMT machinery
    (``operator_disagreement``, ``spectral_disagreement`` in ``rmt/core.py``).
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    total_rows = jacobian.shape[0]
    if total_rows % probe_count != 0:
        raise ValueError("jacobian row count must be a multiple of probe_count")
    length = total_rows // probe_count
    operators = np.empty((probe_count, length, length), dtype=np.float64)
    for probe in range(probe_count):
        block = jacobian[probe::probe_count]
        operators[probe] = block @ block.T
    return operators


@torch.no_grad()
def embed_tokens(model: torch.nn.Module, token_ids: torch.Tensor, *, device: torch.device, batch_size: int) -> np.ndarray:
    """Batch the frozen model's own token embedding lookup for a set of sequences."""
    parts: list[np.ndarray] = []
    for start in range(0, token_ids.shape[0], batch_size):
        stop = min(start + batch_size, token_ids.shape[0])
        parts.append(model.token_embedding(token_ids[start:stop].to(device)).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float64, copy=False)
