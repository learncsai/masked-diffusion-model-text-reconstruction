"""Linear Gaussian denoisers and first-order split-disagreement theory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
RMTObservable = Literal[
    "diagonal_operator",
    "rowsum_operator",
    "prediction_covariance_weighted",
]


@dataclass(frozen=True)
class PerturbativePrediction:
    """Entrywise and normalized aggregate first-order disagreement."""

    entrywise: FloatArray
    aggregate: float


@dataclass(frozen=True)
class MaskingPerturbativePrediction:
    """First-order fluctuation law for the exact coordinate-mask operator.

    ``entrywise`` is expressed in the requested output/input basis.  When an
    input covariance is supplied, ``covariance_weighted_per_mode`` retains
    the full covariance between entries in each output row; unlike the
    scalar-resolvent special case, off-diagonal input covariances do not in
    general drop out for the exact masking derivative.
    """

    entrywise: FloatArray
    aggregate: float
    covariance_weighted_per_mode: FloatArray | None = None


@dataclass(frozen=True)
class ReducedPrediction:
    """One explicitly defined reduction of an entrywise RMT prediction.

    ``per_output_mode`` is deliberately not divided by the representation
    width.  ``aggregate`` is the normalized scalar ``sum_i per_mode_i / d``.
    Keeping these conventions separate avoids silently changing scale when
    representation widths are compared.
    """

    observable: RMTObservable
    per_output_mode: FloatArray
    aggregate: float


@dataclass(frozen=True)
class SpectralDisagreement:
    """Disagreement resolved in a supplied orthonormal eigenbasis."""

    entrywise: FloatArray
    per_mode: FloatArray
    aggregate: float


def sample_gaussian(
    covariance: FloatArray, n: int, *, seed: int | None = None, rng: np.random.Generator | None = None
) -> FloatArray:
    """Draw rows ``x_a ~ N(0, covariance)`` with a local float64 RNG."""
    if n <= 0:
        raise ValueError("n must be positive")
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")
    covariance = np.asarray(covariance, dtype=np.float64)
    values, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    if values.min() <= 0:
        raise ValueError("covariance must be positive definite")
    local_rng = rng if rng is not None else np.random.default_rng(seed)
    z = local_rng.standard_normal((n, covariance.shape[0]))
    square_root = vectors * np.sqrt(values)
    return np.asarray(z @ square_root.T, dtype=np.float64)


def estimate_known_mean_covariance(samples: FloatArray) -> FloatArray:
    """Estimate ``Sigma_hat = X^T X / n`` for a known population mean of zero."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("samples must be a nonempty two-dimensional array")
    return np.asarray(samples.T @ samples / samples.shape[0], dtype=np.float64)


def resolvent_filter(covariance: FloatArray, gamma: float) -> FloatArray:
    """Compute ``Sigma (Sigma + gamma I)^-1`` by symmetric spectral filtering."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    covariance = np.asarray(covariance, dtype=np.float64)
    values, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    filtered = values / (values + gamma)
    return np.asarray((vectors * filtered) @ vectors.T, dtype=np.float64)


def linear_denoiser_operator(cross_covariance: FloatArray, corrupted_covariance: FloatArray) -> FloatArray:
    """Compute the optimal linear map ``Cov(x,y) Cov(y)^-1`` without inversion."""
    cross_covariance = np.asarray(cross_covariance, dtype=np.float64)
    corrupted_covariance = np.asarray(corrupted_covariance, dtype=np.float64)
    return np.asarray(np.linalg.solve(corrupted_covariance.T, cross_covariance.T).T, dtype=np.float64)


def masked_corrupted_covariance(covariance: FloatArray, q: float) -> FloatArray:
    """Return ``Cov(Mx/q) = Sigma + (1-q)/q diag(Sigma)``."""
    if not 0 < q <= 1:
        raise ValueError("q must lie in (0, 1]")
    covariance = np.asarray(covariance, dtype=np.float64)
    return covariance + ((1.0 - q) / q) * np.diag(np.diag(covariance))


def masked_linear_denoiser_operator(covariance: FloatArray, q: float) -> FloatArray:
    """Compute the exact optimal linear reconstruction operator for coordinate masking."""
    return linear_denoiser_operator(covariance, masked_corrupted_covariance(covariance, q))


def masking_perturbative_split_prediction(
    covariance: FloatArray,
    q: float,
    n_a: int,
    n_b: int,
    *,
    basis: FloatArray | None = None,
    input_covariance: FloatArray | None = None,
) -> MaskingPerturbativePrediction:
    """Wishart/Fréchet prediction for ``Sigma(Sigma+rho diag(Sigma))^-1``.

    This is the exact first derivative of the coordinate-masking operator,
    not the isotropic ``gamma I`` replacement.  The covariance perturbation
    follows the same Gaussian Wishart law used by
    :func:`perturbative_split_prediction`.

    ``basis`` supplies orthonormal columns in the original coordinate system.
    Both operator indices are reported in this basis.  If ``input_covariance``
    is given in original coordinates, the returned covariance-weighted
    observable is ``E[Delta W C_z Delta W^T]`` resolved by output mode, with
    all cross-column fluctuation covariances included.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between 0 and 1")
    if n_a <= 0 or n_b <= 0:
        raise ValueError("split sizes must be positive")
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or covariance.shape[0] == 0:
        raise ValueError("covariance must be a nonempty square matrix")
    covariance = (covariance + covariance.T) * 0.5
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("covariance must be finite and positive semidefinite")
    d = covariance.shape[0]
    if basis is None:
        transform = np.eye(d, dtype=np.float64)
    else:
        transform = np.asarray(basis, dtype=np.float64)
        if transform.shape != (d, d):
            raise ValueError(f"basis must have shape {(d, d)}")
        if not np.allclose(transform.T @ transform, np.eye(d), rtol=1e-8, atol=1e-10):
            raise ValueError("basis must be orthonormal")

    rho = (1.0 - q) / q
    corrupted = masked_corrupted_covariance(covariance, q)
    inverse = np.linalg.pinv(corrupted, hermitian=True, rcond=1e-12)
    operator = covariance @ inverse
    left = np.eye(d, dtype=np.float64) - operator
    diagonal_left = -rho * operator

    # For output basis vector u_i and input basis vector u_j,
    # u_i^T DW[H]u_j = a_i^T H b_j + g_ij^T diag(H), where
    # a_i=B^T u_i, b_j=C u_j, and g_ij=e_i .* b_j.
    a = left.T @ transform
    b = inverse @ transform
    e = diagonal_left.T @ transform
    sigma_a = covariance @ a
    sigma_b = covariance @ b
    a_sigma_a = np.sum(a * sigma_a, axis=0)
    a_sigma_b = a.T @ sigma_b
    sigma_squared = covariance * covariance
    alpha = 1.0 / n_a + 1.0 / n_b

    entrywise = np.empty((d, d), dtype=np.float64)
    weighted = None
    if input_covariance is not None:
        input_covariance = np.asarray(input_covariance, dtype=np.float64)
        if input_covariance.shape != (d, d):
            raise ValueError(f"input covariance must have shape {(d, d)}")
        input_covariance = (input_covariance + input_covariance.T) * 0.5
        if not np.isfinite(input_covariance).all() or np.linalg.eigvalsh(input_covariance).min() < -1e-10:
            raise ValueError("input covariance must be finite and positive semidefinite")
        input_in_basis = transform.T @ input_covariance @ transform
        weighted = np.empty(d, dtype=np.float64)

    b_sigma_b = b.T @ sigma_b
    for output_mode in range(d):
        # Q[j,k] = Cov(Delta W'_{ij}, Delta W'_{ik}).  Its diagonal is
        # the requested entrywise fluctuation row.  Keeping Q here is what
        # makes the covariance-weighted reduction exact for masking.
        first = (
            a_sigma_a[output_mode] * b_sigma_b
            + np.outer(a_sigma_b[output_mode], a_sigma_b[output_mode])
        )
        g = e[:, output_mode, None] * b
        diagonal = 2.0 * (g.T @ sigma_squared @ g)
        cross_one_way = 2.0 * ((sigma_a[:, output_mode, None] * sigma_b).T @ g)
        row_covariance = alpha * (first + diagonal + cross_one_way + cross_one_way.T)
        row_covariance = (row_covariance + row_covariance.T) * 0.5
        entrywise[output_mode] = np.maximum(np.diag(row_covariance), 0.0)
        if weighted is not None:
            weighted[output_mode] = max(float(np.sum(row_covariance * input_in_basis)), 0.0)

    return MaskingPerturbativePrediction(
        entrywise=entrywise,
        aggregate=float(entrywise.sum() / d),
        covariance_weighted_per_mode=weighted,
    )


def frechet_divided_difference(eigenvalues: FloatArray, gamma: float) -> FloatArray:
    """Matrix of divided differences for ``f(lambda)=lambda/(lambda+gamma)``.

    The closed form ``gamma / ((a+gamma)(b+gamma))`` is continuous when
    eigenvalues repeat and equals the derivative on the diagonal.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    values = np.asarray(eigenvalues, dtype=np.float64)
    return gamma / ((values[:, None] + gamma) * (values[None, :] + gamma))


def perturbative_split_prediction(
    eigenvalues: FloatArray, gamma: float, n_a: int, n_b: int
) -> PerturbativePrediction:
    """Predict split disagreement from the Gaussian Wishart covariance law.

    This is a first-order, large-sample approximation in the population
    eigenbasis. The aggregate is ``sum_ij E[Delta W_ij^2] / d``.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("split sizes must be positive")
    values = np.asarray(eigenvalues, dtype=np.float64)
    derivative = frechet_divided_difference(values, gamma)
    wishart = (1.0 / n_a + 1.0 / n_b) * np.outer(values, values)
    wishart *= 1.0 + np.eye(values.size, dtype=np.float64)
    entrywise = derivative**2 * wishart
    return PerturbativePrediction(entrywise, float(entrywise.sum() / values.size))


def reduce_perturbative_prediction(
    prediction: PerturbativePrediction | MaskingPerturbativePrediction | FloatArray,
    *,
    observable: RMTObservable,
    input_covariance_in_eigenbasis: FloatArray | None = None,
    input_variances_in_eigenbasis: FloatArray | None = None,
) -> ReducedPrediction:
    """Reduce entrywise operator fluctuations to a declared observable.

    Let ``P_ij = E[Delta W_ij^2]`` in the population covariance eigenbasis.

    - ``diagonal_operator`` returns ``P_ii``.  It measures mode-preserving
      operator coefficients; it is not generic output disagreement.
    - ``rowsum_operator`` returns ``sum_j P_ij``.  It is the squared row norm
      and also the prediction variance for isotropic unit-covariance inputs.
    - ``prediction_covariance_weighted`` returns
      ``sum_j P_ij (C_z)_jj``.  Under the first-order Gaussian/Wishart law,
      ``E[Delta W_ij Delta W_ik]`` vanishes for ``j != k`` in this basis, so
      off-diagonal entries of ``C_z`` do not contribute.  This simplification
      is a theorem-model assumption, not a generic neural-network identity.

    Pass covariance quantities already expressed in the theory eigenbasis.
    """
    prediction_object = isinstance(prediction, (PerturbativePrediction, MaskingPerturbativePrediction))
    entrywise = np.asarray(prediction.entrywise if prediction_object else prediction, dtype=np.float64)
    if entrywise.ndim != 2 or entrywise.shape[0] != entrywise.shape[1] or entrywise.shape[0] == 0:
        raise ValueError("entrywise prediction must be a nonempty square matrix")
    if not np.isfinite(entrywise).all() or (entrywise < 0).any():
        raise ValueError("entrywise prediction must be finite and nonnegative")
    d = entrywise.shape[0]

    if observable == "diagonal_operator":
        if input_covariance_in_eigenbasis is not None or input_variances_in_eigenbasis is not None:
            raise ValueError("diagonal_operator does not accept input covariance weights")
        per_mode = np.diag(entrywise).copy()
    elif observable == "rowsum_operator":
        if input_covariance_in_eigenbasis is not None or input_variances_in_eigenbasis is not None:
            raise ValueError("rowsum_operator does not accept input covariance weights")
        per_mode = entrywise.sum(axis=1)
    elif observable == "prediction_covariance_weighted":
        if isinstance(prediction, MaskingPerturbativePrediction):
            if input_covariance_in_eigenbasis is not None or input_variances_in_eigenbasis is not None:
                raise ValueError("exact masking prediction already contains its declared input covariance")
            if prediction.covariance_weighted_per_mode is None:
                raise ValueError("exact masking prediction was computed without an input covariance")
            per_mode = np.asarray(prediction.covariance_weighted_per_mode, dtype=np.float64)
            return ReducedPrediction(observable, per_mode, float(per_mode.sum() / d))
        if input_covariance_in_eigenbasis is not None and input_variances_in_eigenbasis is not None:
            raise ValueError("pass either input covariance or input variances, not both")
        if input_covariance_in_eigenbasis is not None:
            covariance = np.asarray(input_covariance_in_eigenbasis, dtype=np.float64)
            if covariance.shape != (d, d):
                raise ValueError(f"input covariance must have shape {(d, d)}")
            if not np.isfinite(covariance).all():
                raise ValueError("input covariance must be finite")
            symmetric = (covariance + covariance.T) * 0.5
            if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
                raise ValueError("input covariance must be symmetric")
            if np.linalg.eigvalsh(symmetric).min() < -1e-10:
                raise ValueError("input covariance must be positive semidefinite")
            variances = np.diag(symmetric)
        elif input_variances_in_eigenbasis is not None:
            variances = np.asarray(input_variances_in_eigenbasis, dtype=np.float64)
            if variances.shape != (d,):
                raise ValueError(f"input variances must have shape {(d,)}")
        else:
            raise ValueError("prediction_covariance_weighted requires input covariance or variances")
        if not np.isfinite(variances).all() or (variances < -1e-12).any():
            raise ValueError("input variances must be finite and nonnegative")
        per_mode = entrywise @ np.maximum(variances, 0.0)
    else:
        raise ValueError(f"unknown RMT observable: {observable}")

    return ReducedPrediction(observable, np.asarray(per_mode, dtype=np.float64), float(per_mode.sum() / d))


def raw_wishart_disagreement_ratio(
    operator_a: FloatArray, operator_b: FloatArray, n_a: int, n_b: int,
) -> float:
    """Realized/predicted ratio under the plain (pre-resolvent) Wishart 4th-moment law.

    Treats ``0.5*(operator_a+operator_b)`` as the population covariance that
    ``operator_a`` and ``operator_b`` independently estimate from ``n_a``,
    ``n_b`` samples, and compares the realized squared disagreement against
    the standard prediction ``(1/n_a+1/n_b)(Sigma_ii Sigma_jj + Sigma_ij^2)``.
    A ratio near 1 means the object fluctuates like an ordinary sample
    covariance; a ratio far from 1 means it does not. This is a genuine,
    falsifiable per-condition test (unlike comparing an operator's own
    linearization against its own realized behavior): the prediction uses
    only the reference operator and the real sample sizes, not anything
    that requires having already observed both realizations.
    """
    operator_a = np.asarray(operator_a, dtype=np.float64)
    operator_b = np.asarray(operator_b, dtype=np.float64)
    if n_a <= 0 or n_b <= 0:
        raise ValueError("n_a and n_b must be positive")
    reference = 0.5 * (operator_a + operator_b)
    diagonal = np.diag(reference)
    predicted = (1.0 / n_a + 1.0 / n_b) * (np.outer(diagonal, diagonal) + reference**2)
    realized = (operator_a - operator_b) ** 2
    return float(realized.sum() / predicted.sum())


def operator_disagreement(operator_a: FloatArray, operator_b: FloatArray) -> tuple[float, float]:
    """Return normalized squared Frobenius disagreement and its RMS form."""
    difference = np.asarray(operator_a, dtype=np.float64) - np.asarray(operator_b, dtype=np.float64)
    variance_like = float(np.sum(difference * difference) / difference.shape[0])
    return variance_like, float(np.sqrt(variance_like))


def prediction_disagreement(
    operator_a: FloatArray, operator_b: FloatArray, corrupted_covariance: FloatArray
) -> float:
    """Analytically compute shared-input prediction disagreement per coordinate."""
    difference = np.asarray(operator_a, dtype=np.float64) - np.asarray(operator_b, dtype=np.float64)
    covariance = np.asarray(corrupted_covariance, dtype=np.float64)
    return float(np.einsum("ij,jk,ik->", difference, covariance, difference) / difference.shape[0])


def monte_carlo_masked_prediction_disagreement(
    operator_a: FloatArray,
    operator_b: FloatArray,
    covariance: FloatArray,
    q: float,
    *,
    draws: int = 10_000,
    seed: int = 0,
) -> float:
    """Estimate prediction disagreement using shared ``Mx/q`` corruptions."""
    rng = np.random.default_rng(seed)
    clean = sample_gaussian(covariance, draws, rng=rng)
    mask = rng.binomial(1, q, size=clean.shape)
    corrupted = clean * mask / q
    difference = np.asarray(operator_a) - np.asarray(operator_b)
    predictions = corrupted @ difference.T
    return float(np.mean(np.sum(predictions * predictions, axis=1)) / clean.shape[1])


def spectral_disagreement(
    operator_a: FloatArray, operator_b: FloatArray, eigenvectors: FloatArray
) -> SpectralDisagreement:
    """Resolve squared operator disagreement in a population eigenbasis."""
    difference = np.asarray(operator_a, dtype=np.float64) - np.asarray(operator_b, dtype=np.float64)
    basis = np.asarray(eigenvectors, dtype=np.float64)
    transformed = basis.T @ difference @ basis
    entrywise = transformed * transformed
    d = difference.shape[0]
    return SpectralDisagreement(entrywise, entrywise.sum(axis=1) / d, float(entrywise.sum() / d))
