"""Synthetic Gaussian RMT tools for diffusion-model reproducibility studies."""

from .rmt.core import (
    MaskingPerturbativePrediction,
    estimate_known_mean_covariance,
    frechet_divided_difference,
    linear_denoiser_operator,
    masked_corrupted_covariance,
    masked_linear_denoiser_operator,
    masking_perturbative_split_prediction,
    operator_disagreement,
    perturbative_split_prediction,
    prediction_disagreement,
    resolvent_filter,
    sample_gaussian,
    spectral_disagreement,
)
from .rmt.covariance import CovarianceModel, generate_covariance

__all__ = [
    "CovarianceModel",
    "MaskingPerturbativePrediction",
    "estimate_known_mean_covariance",
    "frechet_divided_difference",
    "generate_covariance",
    "linear_denoiser_operator",
    "masked_corrupted_covariance",
    "masked_linear_denoiser_operator",
    "masking_perturbative_split_prediction",
    "operator_disagreement",
    "perturbative_split_prediction",
    "prediction_disagreement",
    "resolvent_filter",
    "sample_gaussian",
    "spectral_disagreement",
]
