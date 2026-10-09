"""Stationary covariance-resolvent Wiener denoiser for masked diffusion.

This is the closest MDLM analog of the Wiener denoiser that random matrix
theory is written for: a spectral function of one empirical covariance,
applied to a continuous feature vector, with a scalar noise parameter set by
the corruption rate.

Tokens are mapped through a fixed embedding ``e`` and the sequence becomes
``y = (e(x_1), ..., e(x_L))`` in ``R^{Ld}``.  Masking zeroes whole position
blocks of the centred vector, so with independent block masks at visibility
``r = 1 - q``

    Sigma_yz = r Sigma,     Sigma_zz = r^2 Sigma + r(1-r) B,

where ``B`` is the block diagonal of ``Sigma``.  The population linear MMSE
predictor is therefore

    W_r = Sigma_yz Sigma_zz^{-1} = Sigma [ r Sigma + (1-r) B ]^{-1},

and in the basis that whitens ``B`` this is exactly the resolvent

    W_r = (1/r) Sigma' (Sigma' + lambda_q I)^{-1},   lambda_q = q / (1-q),

which is the form the RMT machinery in ``rmt/core.py`` consumes.  The two are
algebraically identical; ``wiener_operator`` can emit either.

Two modelling choices are forced by the data rather than by taste.  The
position-specific empirical covariance has *negative* denoising skill here:
at L=64 and d=96 it is a 6144-dimensional covariance estimated from at most
4096 sequences, and its long-range blocks are sampling noise that the inverse
amplifies.  Estimating a **stationary** (block-Toeplitz) covariance by pooling
over position pairs at equal separation is what makes the estimator work, and
is the discrete analog of the translation-invariant covariance that makes
linear image denoising viable.  Note also that the cross-covariance blocks are
oriented as ``Sigma_{i,i+delta} = C(delta)``; the transpose is a different and
much worse operator.
"""

from __future__ import annotations

import numpy as np

__all__ = ["stationary_autocovariance", "block_toeplitz", "block_whitener",
           "wiener_operator", "fit_probe_readout"]


def stationary_autocovariance(
    features: np.ndarray,
    *,
    taper: bool = True,
) -> dict[int, np.ndarray]:
    """Return ``C(delta) = E[y_i y_{i+delta}^T]`` pooled over positions.

    ``features`` has shape ``(sequences, positions, dimension)`` and must
    already be centred on the position-independent training mean, which is the
    mean stationarity implies.  With ``taper`` the Bartlett normalisation
    ``1/(nL)`` is used for every lag, which guarantees the assembled
    block-Toeplitz matrix is positive semidefinite at the cost of shrinking
    long lags; without it each lag is normalised by its own ``n(L-|delta|)``,
    which is unbiased but can leave the assembly indefinite.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError("features must have shape (sequences, positions, dimension)")
    sequences, positions, _ = features.shape

    lags: dict[int, np.ndarray] = {}
    for lag in range(positions):
        pairs = sequences * (positions if taper else positions - lag)
        block = np.einsum(
            "sid,sie->de", features[:, : positions - lag], features[:, lag:],
        ) / pairs
        lags[lag] = block
        lags[-lag] = block.T
    return lags


def block_toeplitz(lags: dict[int, np.ndarray], positions: int) -> np.ndarray:
    """Assemble the stationary sequence covariance from its lag blocks.

    Block ``(i, j)`` is ``C(j - i)``, so that the block indexed by a positive
    separation is the covariance of a position with a *later* one. Reversing
    this orientation transposes every block and yields a different operator.
    """
    dimension = lags[0].shape[0]
    size = positions * dimension
    covariance = np.zeros((size, size), dtype=np.float64)
    for i in range(positions):
        for j in range(positions):
            covariance[
                i * dimension:(i + 1) * dimension, j * dimension:(j + 1) * dimension
            ] = lags[j - i]
    return 0.5 * (covariance + covariance.T)


def block_whitener(
    within_position: np.ndarray,
    positions: int,
    *,
    floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the block whitener and its inverse for ``B = I_L kron C(0)``.

    Whitening by ``C(0)^{-1/2}`` at every position sends the block diagonal of
    the covariance to the identity, which is the basis in which the masking
    Wiener filter becomes a plain resolvent with ridge ``lambda_q``.
    """
    eigenvalues, basis = np.linalg.eigh(0.5 * (within_position + within_position.T))
    eigenvalues = np.maximum(eigenvalues, floor)
    inverse_root = basis @ np.diag(eigenvalues ** -0.5) @ basis.T
    root = basis @ np.diag(eigenvalues ** 0.5) @ basis.T
    identity = np.eye(positions)
    return np.kron(identity, inverse_root), np.kron(identity, root)


def wiener_operator(
    covariance: np.ndarray,
    within_position: np.ndarray,
    positions: int,
    q: float,
    *,
    form: str = "resolvent",
    extra_ridge: float = 0.0,
) -> np.ndarray:
    """Return the masked Wiener denoiser at mask rate ``q``.

    ``form="general"`` builds ``Sigma [ r Sigma + (1-r) B ]^{-1}`` by a direct
    solve. ``form="resolvent"`` whitens ``B`` and applies the spectral filter
    ``(1/r) s / (s + lambda_q)`` to the eigenvalues of the whitened covariance,
    which is the representation the RMT perturbation formulas act on. The two
    agree to machine precision; the resolvent form additionally exposes the
    spectrum, so prefer it whenever the eigenvalues are wanted downstream.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between 0 and 1")
    if form not in ("general", "resolvent"):
        raise ValueError("form must be 'general' or 'resolvent'")

    visibility = 1.0 - q
    size = covariance.shape[0]
    dimension = size // positions
    block = np.zeros_like(covariance)
    for index in range(positions):
        span = slice(index * dimension, (index + 1) * dimension)
        block[span, span] = within_position

    if form == "general":
        system = visibility * covariance + (1.0 - visibility) * block
        if extra_ridge:
            system = system + extra_ridge * np.trace(covariance) / size * np.eye(size)
        return np.linalg.solve(system, covariance.T).T

    inverse_root, root = block_whitener(within_position, positions)
    whitened = inverse_root @ covariance @ inverse_root
    eigenvalues, basis = np.linalg.eigh(0.5 * (whitened + whitened.T))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    ridge = q / (1.0 - q) + extra_ridge
    spectral = (1.0 / visibility) * eigenvalues / (eigenvalues + ridge)
    return root @ (basis * spectral) @ basis.T @ inverse_root


def fit_probe_readout(
    features: np.ndarray,
    probe_values: np.ndarray,
    *,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the linear map from a feature vector to probe values.

    The denoiser predicts an embedding, while every bridge metric is stated in
    probe space, so a readout is required. It is fit by least squares over the
    training token distribution, hence weighted by token frequency: a random
    probe is not a linear function of a low-dimensional embedding in general,
    but the frequent tokens that carry most of the mass are well captured
    (measured R^2 is 0.40 to 0.50 at d=96, against a naive d/K ceiling of
    0.03). Returns the coefficient matrix and the intercept.
    """
    features = np.asarray(features, dtype=np.float64)
    probe_values = np.asarray(probe_values, dtype=np.float64)
    design = np.hstack([features, np.ones((features.shape[0], 1))])
    gram = design.T @ design
    penalty = ridge * np.trace(gram) / design.shape[1]
    solution = np.linalg.solve(
        gram + penalty * np.eye(design.shape[1]), design.T @ probe_values.T,
    )
    return solution[:-1], solution[-1]
