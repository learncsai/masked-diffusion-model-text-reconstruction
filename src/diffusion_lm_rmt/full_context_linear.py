"""Full-vocabulary affine ridge denoising conditional on the visible positions.

The kernel is exactly the inner product of concatenated one-hot tokens, divided
by the number of visible positions. No random features or token pruning are used.
For each mask, every input coefficient is estimated jointly. The intercept is
unpenalized. This is a family of affine maps indexed by mask, not one linear map
that must also infer which coordinates have been hidden.
"""
from __future__ import annotations

import numpy as np
import torch


class FullContextRidge:
    def __init__(self, tokens, *, device="cpu"):
        self.tokens = np.asarray(tokens, dtype=np.int64)
        if self.tokens.ndim != 2 or len(self.tokens) < 2:
            raise ValueError("Expected at least two complete token sequences")
        self.device = torch.device(device)
        self.tensor = torch.as_tensor(self.tokens, device=self.device)
        # Exact integer match counts survive float32 storage. All subsequent
        # centering and linear algebra use float64.
        self.matches = (self.tensor.T[:, :, None] == self.tensor.T[:, None, :]).float()
        self.maximum_relative_residual = 0.0

    def weights(self, query, hidden, ridge):
        """Return a dual weight vector whose sum is one, and solver diagnostics.

        For every hidden output position, the prediction is the weighted sum of
        training one-hot targets at that position. It solves mean squared loss
        plus ridge * ||W||_F^2 in the normalized one-hot feature coordinates.
        """
        if ridge <= 0:
            raise ValueError("Ridge must be strictly positive")
        hidden = np.asarray(hidden, dtype=bool)
        query = np.asarray(query, dtype=np.int64)
        n, length = self.tokens.shape
        if query.shape != (length,) or hidden.shape != (length,):
            raise ValueError("Query and mask must match sequence length")
        visible = ~hidden
        if not visible.any():
            return np.full(n, 1/n), dict(relative_residual=0.0, sum_error=0.0)
        v = torch.as_tensor(visible, device=self.device, dtype=torch.float32)
        kernel = (v @ self.matches.reshape(length, -1)).reshape(n, n).double()
        kernel /= int(visible.sum())
        row_mean = kernel.mean(1)
        grand_mean = row_mean.mean()
        centered = kernel - row_mean[:, None] - row_mean[None, :] + grand_mean
        q = torch.as_tensor(query, device=self.device)
        cross = ((self.tensor == q) * v.bool()).sum(1).double() / int(visible.sum())
        rhs = cross - row_mean - cross.mean() + grand_mean
        centered.diagonal().add_(n * ridge)
        chol = torch.linalg.cholesky(centered)
        alpha = torch.cholesky_solve(rhs[:, None], chol).flatten()
        residual = float(torch.linalg.vector_norm(centered @ alpha - rhs) /
                         torch.linalg.vector_norm(rhs).clamp_min(1e-30))
        if residual > 1e-8:
            raise ArithmeticError(f"Ridge normal-equation residual {residual}")
        weights = (alpha - alpha.mean() + 1/n).cpu().numpy()
        self.maximum_relative_residual = max(self.maximum_relative_residual, residual)
        return weights, dict(relative_residual=residual,
                             sum_error=float(abs(weights.sum()-1)))


def scores_from_weights(train, weights, positions, classes):
    """Full-vocabulary affine scores. No clipping or softmax is applied."""
    return np.asarray([np.bincount(train[:, i], weights=weights, minlength=classes)
                       for i in positions], dtype=np.float64)


def prediction_metrics(scores, targets, ks=(1, 2, 4, 8, 16, 32, 64)):
    """Exact Brier/MSE and target ranks, with ascending token ID for ties."""
    targets = np.asarray(targets, dtype=np.int64)
    truth = scores[np.arange(len(targets)), targets]
    rank = 1 + (scores > truth[:, None]).sum(1)
    rank += ((scores == truth[:, None]) &
             (np.arange(scores.shape[1])[None, :] < targets[:, None])).sum(1)
    return dict(mse=float(np.mean(1-2*truth+np.square(scores).sum(1))),
                **{f"accuracy_{k}": float(np.mean(rank <= k)) for k in ks})
