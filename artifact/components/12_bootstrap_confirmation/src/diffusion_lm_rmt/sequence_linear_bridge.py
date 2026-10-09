"""Sequence-wide token-feature linear denoiser for a richer bridge test.

Unlike the 64-by-64 scalar-probe covariance denoiser, this model lifts every
token to several fixed Gaussian identity features and fits one full linear map
across all position-feature blocks.  It is a closed-form ridge analysis model;
no MDLM parameters are updated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SequenceRidgeDenoiser:
    x_mean: np.ndarray
    y_mean: np.ndarray
    coefficients: np.ndarray
    sequence_length: int
    feature_count: int

    def predict(self, coded_tokens: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
        features = masked_sequence_features(coded_tokens, target_mask)
        prediction = (features - self.x_mean) @ self.coefficients + self.y_mean
        return prediction.reshape(coded_tokens.shape[0], self.sequence_length, self.feature_count)


def masked_sequence_features(coded_tokens: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    """Flatten visible token-identity features and append mask indicators."""
    coded_tokens = np.asarray(coded_tokens, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=bool)
    if coded_tokens.ndim != 3 or target_mask.shape != coded_tokens.shape[:2]:
        raise ValueError("coded tokens must be (sequences, positions, features) with a matching mask")
    visible_codes = coded_tokens * (~target_mask)[..., None]
    return np.concatenate(
        [visible_codes.reshape(coded_tokens.shape[0], -1), target_mask.astype(np.float64)], axis=1,
    )


def deterministic_training_masks(
    token_ids: np.ndarray,
    attention_mask: np.ndarray,
    *,
    q: float,
    augmentations: int,
    seed: int,
    special_ids: tuple[int, ...],
) -> np.ndarray:
    """Create a shared, reproducible coordinate-mask tape for ridge fitting."""
    if not 0.0 < q < 1.0 or augmentations <= 0:
        raise ValueError("q must be in (0,1) and augmentations must be positive")
    token_ids = np.asarray(token_ids)
    eligible = np.asarray(attention_mask, dtype=bool).copy()
    for token_id in special_ids:
        eligible &= token_ids != int(token_id)
    rng = np.random.default_rng(seed)
    masks = (rng.random((augmentations,) + token_ids.shape) >= q) & eligible[None, ...]
    for augmentation in range(augmentations):
        empty = np.flatnonzero(~masks[augmentation].any(axis=1) & eligible.any(axis=1))
        for row in empty:
            positions = np.flatnonzero(eligible[row])
            masks[augmentation, row, positions[int(rng.integers(positions.size))]] = True
    return masks


def fit_sequence_ridge_denoiser(
    coded_tokens: np.ndarray,
    masks: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> SequenceRidgeDenoiser:
    """Fit the full block-linear conditional mean by one stable ridge solve."""
    coded_tokens = np.asarray(coded_tokens, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    if coded_tokens.ndim != 3 or masks.ndim != 3 or masks.shape[1:] != coded_tokens.shape[:2]:
        raise ValueError("masks must have shape (augmentations, sequences, positions)")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    x = np.concatenate([masked_sequence_features(coded_tokens, mask) for mask in masks], axis=0)
    target = coded_tokens.reshape(coded_tokens.shape[0], -1)
    y = np.tile(target, (masks.shape[0], 1))
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x_centered, y_centered = x - x_mean, y - y_mean
    gram = x_centered.T @ x_centered / x.shape[0]
    cross = x_centered.T @ y_centered / x.shape[0]
    penalty = ridge * float(np.trace(gram)) / max(gram.shape[0], 1)
    coefficients = np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0], dtype=np.float64), cross,
    )
    return SequenceRidgeDenoiser(
        x_mean=x_mean,
        y_mean=y_mean,
        coefficients=coefficients,
        sequence_length=coded_tokens.shape[1],
        feature_count=coded_tokens.shape[2],
    )
