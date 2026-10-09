"""Mask-pattern-conditioned optimal linear reconstruction in probe space.

The global masking bridge marginalizes over Bernoulli masks and therefore
uses one operator at a fixed survival rate for every evaluation sequence.
Here the realized mask is treated as observed side information.  For each
sequence and probe, the missing coordinates are reconstructed from the
actually visible coordinates using the split-specific Gaussian conditional
mean, which is also the best affine least-squares predictor.
"""

from __future__ import annotations

import numpy as np


def pattern_conditioned_linear_denoise(
    coded_tokens: np.ndarray,
    target_mask: np.ndarray,
    attention_mask: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Reconstruct each realized mask pattern with its conditional linear map.

    ``coded_tokens`` has shape ``(sequences, positions)`` for one fixed
    vocabulary probe. ``target_mask`` is true at coordinates hidden from the
    denoiser. Values outside ``attention_mask`` are neither conditioning
    variables nor reconstruction targets.

    At missing coordinates ``T`` and observed coordinates ``O``, the result is

        mean[T] + covariance[T, O] covariance[O, O]^{-1}
                  (coded_tokens[O] - mean[O]).

    A scale-relative ridge stabilizes singular empirical sub-covariances. The
    visible coordinates are copied through, although bridge metrics are
    evaluated only at target coordinates.
    """
    coded_tokens = np.asarray(coded_tokens, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=bool)
    attention_mask = np.asarray(attention_mask, dtype=bool)
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)

    if coded_tokens.ndim != 2:
        raise ValueError("coded_tokens must have shape (sequences, positions)")
    if target_mask.shape != coded_tokens.shape or attention_mask.shape != coded_tokens.shape:
        raise ValueError("target_mask and attention_mask must match coded_tokens")
    positions = coded_tokens.shape[1]
    if mean.shape != (positions,) or covariance.shape != (positions, positions):
        raise ValueError("mean/covariance dimensions must match the sequence length")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    prediction = coded_tokens.copy()
    covariance = 0.5 * (covariance + covariance.T)
    for row in range(coded_tokens.shape[0]):
        missing = np.flatnonzero(target_mask[row] & attention_mask[row])
        if missing.size == 0:
            continue
        observed = np.flatnonzero((~target_mask[row]) & attention_mask[row])
        if observed.size == 0:
            prediction[row, missing] = mean[missing]
            continue

        covariance_oo = covariance[np.ix_(observed, observed)]
        scale = float(np.trace(covariance_oo) / observed.size)
        penalty = ridge * max(scale, np.finfo(np.float64).eps)
        system = covariance_oo + penalty * np.eye(observed.size, dtype=np.float64)
        centered_observed = coded_tokens[row, observed] - mean[observed]
        try:
            coefficients = np.linalg.solve(system, centered_observed)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(system, centered_observed, rcond=None)[0]
        prediction[row, missing] = (
            mean[missing]
            + covariance[np.ix_(missing, observed)] @ coefficients
        )
    return prediction
