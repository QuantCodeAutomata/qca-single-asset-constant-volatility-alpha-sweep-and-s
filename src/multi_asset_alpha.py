"""
Multi-asset constant-volatility alpha-convention model.

Experiment 2: Replicates multi-asset Merton results under alpha-interpreted
integration, including Remark 4.2 covariance examples and Proposition 4.2
law-preserving invariance.

Convention: dS_t = D(S_t)(mu dt + Sigma ∘_alpha dW_t)
Itô form:   mu_Ito = mu + alpha * d_V,  where d_V = diag(V), V = Sigma Sigma^T

Reference: Paper §4.2 multi-asset constant-volatility case.
Custom — Context7 found no library equivalent (formula-based closed-form).
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as nla
import pandas as pd
import scipy.linalg as sla


# ---------------------------------------------------------------------------
# Covariance helpers
# ---------------------------------------------------------------------------


def covariance_from_sigma(Sigma: np.ndarray) -> np.ndarray:
    """
    Compute the covariance matrix V = Sigma @ Sigma.T.

    Parameters
    ----------
    Sigma : (n, m) volatility loading matrix

    Returns
    -------
    V : (n, n) symmetric positive semi-definite covariance matrix
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    return Sigma @ Sigma.T


def variance_vector(V: np.ndarray) -> np.ndarray:
    """
    Extract the variance vector d_V = diag(V).

    Parameters
    ----------
    V : (n, n) covariance matrix

    Returns
    -------
    d_V : (n,) vector of diagonal elements of V
    """
    V = np.asarray(V, dtype=np.float64)
    return np.diag(V).copy()


def check_positive_definite(V: np.ndarray) -> bool:
    """
    Return True if V is symmetric positive definite, False otherwise.

    Checks via the minimum eigenvalue.

    Parameters
    ----------
    V : (n, n) matrix

    Returns
    -------
    bool : True if positive definite, False otherwise
    """
    eigs = nla.eigvalsh(np.asarray(V, dtype=np.float64))
    return bool(eigs.min() > 0)


def _assert_positive_definite(V: np.ndarray) -> np.ndarray:
    """
    Assert V is positive definite; raise ValueError and return eigenvalues.

    Parameters
    ----------
    V : (n, n) matrix

    Returns
    -------
    eigs : sorted ascending eigenvalues

    Raises
    ------
    ValueError if V is not positive definite.
    """
    eigs = nla.eigvalsh(np.asarray(V, dtype=np.float64))
    if eigs.min() <= 0:
        raise ValueError(
            f"V is not positive definite; min eigenvalue = {eigs.min():.6e}"
        )
    return eigs


# ---------------------------------------------------------------------------
# Itô conversion and optimal controls
# ---------------------------------------------------------------------------


def ito_drift_multi(
    mu: np.ndarray, V: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Convert nominal alpha-convention drift vector to effective Itô drift.

    mu_Ito = mu + alpha * d_V,  where d_V = diag(V)

    Parameters
    ----------
    mu    : (n,) nominal drift vector
    V     : (n, n) covariance matrix V = Sigma Sigma^T
    alpha : convention parameter in [0, 1]

    Returns
    -------
    mu_ito : (n,) effective Itô drift vector
    """
    mu = np.asarray(mu, dtype=np.float64)
    d_V = variance_vector(np.asarray(V, dtype=np.float64))
    return mu + alpha * d_V


def optimal_theta_multi(
    mu: np.ndarray,
    r: float,
    V: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Log-utility optimal risky weight vector under convention alpha.

    theta_alpha^* = V^{-1}(mu - r*1) + alpha * V^{-1} * d_V
                  = V^{-1}[(mu - r*1) + alpha * d_V]

    Solved as a linear system V @ theta = rhs for numerical stability.

    Parameters
    ----------
    mu    : (n,) nominal drift vector
    r     : scalar risk-free rate
    V     : (n, n) covariance matrix
    alpha : convention parameter in [0, 1]

    Returns
    -------
    theta : (n,) optimal risky weight vector
    """
    mu = np.asarray(mu, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    _assert_positive_definite(V)
    d_V = variance_vector(V)
    n = len(mu)
    rhs = (mu - r * np.ones(n)) + alpha * d_V
    # Solve V theta = rhs  (more stable than explicit inverse)
    theta = nla.solve(V, rhs)
    return theta


def alpha_sensitivity_multi(V: np.ndarray) -> np.ndarray:
    """
    Compute the alpha sensitivity d theta_alpha^*/d alpha = V^{-1} d_V.

    Parameters
    ----------
    V : (n, n) covariance matrix

    Returns
    -------
    sensitivity : (n,) vector V^{-1} d_V
    """
    V = np.asarray(V, dtype=np.float64)
    _assert_positive_definite(V)
    d_V = variance_vector(V)
    return nla.solve(V, d_V)


# ---------------------------------------------------------------------------
# Law-preserving transformation (Proposition 4.2)
# ---------------------------------------------------------------------------


def law_preserving_mu(
    bar_mu: np.ndarray, d_V: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Compute the convention-specific nominal drift that preserves the
    equivalent Itô law.

    mu^(alpha) = bar_mu - alpha * d_V

    This ensures mu^(alpha) + alpha * d_V = bar_mu for all alpha.

    Parameters
    ----------
    bar_mu : (n,) target effective Itô drift vector
    d_V    : (n,) variance vector diag(V)
    alpha  : convention parameter in [0, 1]

    Returns
    -------
    mu_nominal : (n,) convention-specific nominal drift
    """
    bar_mu = np.asarray(bar_mu, dtype=np.float64)
    d_V = np.asarray(d_V, dtype=np.float64)
    return bar_mu - alpha * d_V


def verify_law_preserving_invariance(
    bar_mu: np.ndarray,
    r: float,
    V: np.ndarray,
    alpha_grid: np.ndarray,
) -> dict:
    """
    Verify Proposition 4.2: under the law-preserving parameterisation, the
    optimal portfolio theta^* is invariant across all alpha values.

    Parameters
    ----------
    bar_mu     : (n,) target effective Itô drift
    r          : risk-free rate
    V          : (n, n) covariance matrix
    alpha_grid : 1-D array of alpha values to test

    Returns
    -------
    dict with:
        'theta_reference' : (n,) invariant optimal weight
        'max_deviation'   : max L∞ deviation across the alpha grid
        'is_invariant'    : bool (True if deviation < 1e-12)
    """
    bar_mu = np.asarray(bar_mu, dtype=np.float64)
    d_V = variance_vector(V)
    thetas = []
    for alpha in alpha_grid:
        mu_alpha = law_preserving_mu(bar_mu, d_V, alpha)
        theta = optimal_theta_multi(mu_alpha, r, V, alpha)
        thetas.append(theta)
    thetas = np.array(thetas)
    theta_ref = thetas[0]
    deviations = np.max(np.abs(thetas - theta_ref), axis=1)
    max_dev = float(np.max(deviations))
    return {
        "theta_reference": theta_ref,
        "max_deviation": max_dev,
        "is_invariant": max_dev < 1e-12,
    }


# ---------------------------------------------------------------------------
# Remark 4.2 paper-specified matrices
# ---------------------------------------------------------------------------

# Matrix A (4×4): eigenvalues 2, 4, 4, 6; V^{-1} d_V = 1
MATRIX_A = np.array(
    [
        [4.0, 1.0, -1.0, 0.0],
        [1.0, 4.0, 0.0, -1.0],
        [-1.0, 0.0, 4.0, 1.0],
        [0.0, -1.0, 1.0, 4.0],
    ]
)

# Matrix B (2×2): d_V = (1, 4)^T; V^{-1} d_V ≈ (-1.143, 1.429)
MATRIX_B = np.array([[1.0, 1.5], [1.5, 4.0]])

# Matrix C (3×3): unit diagonal, d_V = 1; V^{-1} 1 ≈ (0.803, 0.619, 0.734)
MATRIX_C = np.array(
    [
        [1.0, 0.2, 0.1],
        [0.2, 1.0, 0.3],
        [0.1, 0.3, 1.0],
    ]
)


def remark42_sensitivity_table() -> pd.DataFrame:
    """
    Compute and return the Remark 4.2 sensitivity vectors V^{-1} d_V for
    the three paper-specified covariance matrices A, B, C.

    Returns
    -------
    pd.DataFrame summarising d_V and V^{-1} d_V for each matrix.
    """
    records = []
    for label, V in [("A (4×4)", MATRIX_A), ("B (2×2)", MATRIX_B), ("C (3×3)", MATRIX_C)]:
        eigs = _assert_positive_definite(V)
        d_V = variance_vector(V)
        sensitivity = alpha_sensitivity_multi(V)
        records.append(
            {
                "label": label,
                "d_V": d_V.tolist(),
                "V^{-1} d_V": sensitivity.tolist(),
                "min_eigenvalue": float(eigs.min()),
            }
        )
    return pd.DataFrame(records)
