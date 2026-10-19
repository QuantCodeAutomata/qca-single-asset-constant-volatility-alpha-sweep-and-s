"""
General factor-driven model: alpha-correction decomposition.

Experiment 3: One-risky-asset / one-state-factor log-utility model with
state-dependent volatility and correlated shocks.

Model (alpha-interpreted):
    dS_t/S_t  = mu(X_t) dt + sigma(X_t) ∘_alpha dW_t^S
    dX_t      = b(X_t)  dt + nu(X_t)    ∘_alpha dW_t^X
    d<W^S, W^X>_t = varrho dt

Itô conversions (paper §5 general factor model):
    mu_eff(x)  = mu(x) + alpha * [sigma(x)^2 + varrho * sigma'(x) * nu(x)]
    b_alpha(x) = b(x)  + alpha * nu(x) * nu'(x)

Optimal controls (log-utility):
    c_t^* = rho * A_t
    pi_t^* = [mu_eff(X_t) - r] / sigma(X_t)^2

Custom — Context7 found no library equivalent (paper general-factor formulas).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ScalarFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Itô drift conversions
# ---------------------------------------------------------------------------


def mu_eff_factor(
    x: np.ndarray,
    mu_fn: ScalarFn,
    sigma_fn: ScalarFn,
    dsigma_fn: ScalarFn,
    nu_fn: ScalarFn,
    alpha: float,
    varrho: float,
) -> np.ndarray:
    """
    Effective Itô drift of the risky asset in the factor model.

    mu_eff(x) = mu(x) + alpha * [sigma(x)^2 + varrho * sigma'(x) * nu(x)]

    Decomposed into:
        own-variance channel : alpha * sigma(x)^2
        covariation channel  : alpha * varrho * sigma'(x) * nu(x)

    Parameters
    ----------
    x          : state-variable grid (1-D array)
    mu_fn      : callable mu(x) - nominal return drift
    sigma_fn   : callable sigma(x) - return volatility (must be != 0)
    dsigma_fn  : callable sigma'(x) - derivative of sigma w.r.t. x
    nu_fn      : callable nu(x) - factor volatility
    alpha      : convention parameter in [0, 1]
    varrho     : shock correlation in [-1, 1]

    Returns
    -------
    mu_eff : array same shape as x
    """
    x = np.asarray(x, dtype=np.float64)
    own_var = alpha * sigma_fn(x) ** 2
    covar = alpha * varrho * dsigma_fn(x) * nu_fn(x)
    return mu_fn(x) + own_var + covar


def b_alpha_factor(
    x: np.ndarray,
    b_fn: ScalarFn,
    nu_fn: ScalarFn,
    dnu_fn: ScalarFn,
    alpha: float,
) -> np.ndarray:
    """
    Effective Itô drift of the state factor under convention alpha.

    b_alpha(x) = b(x) + alpha * nu(x) * nu'(x)

    Parameters
    ----------
    x       : state-variable grid (1-D array)
    b_fn    : callable b(x) - factor drift under Itô
    nu_fn   : callable nu(x) - factor volatility
    dnu_fn  : callable nu'(x) - derivative of nu w.r.t. x
    alpha   : convention parameter in [0, 1]

    Returns
    -------
    b_alpha : array same shape as x
    """
    x = np.asarray(x, dtype=np.float64)
    return b_fn(x) + alpha * nu_fn(x) * dnu_fn(x)


def alpha_correction_decomposition(
    x: np.ndarray,
    sigma_fn: ScalarFn,
    dsigma_fn: ScalarFn,
    nu_fn: ScalarFn,
    alpha: float,
    varrho: float,
) -> dict:
    """
    Decompose the alpha correction in mu_eff(x) - mu(x) into channels.

    Returns
    -------
    dict with keys:
        'own_variance'   : alpha * sigma(x)^2   (always present)
        'covariation'    : alpha * varrho * sigma'(x) * nu(x)
        'total_correction': sum of the two channels
    """
    x = np.asarray(x, dtype=np.float64)
    own_var = alpha * sigma_fn(x) ** 2
    covar = alpha * varrho * dsigma_fn(x) * nu_fn(x)
    return {
        "own_variance": own_var,
        "covariation": covar,
        "total_correction": own_var + covar,
    }


# ---------------------------------------------------------------------------
# Optimal portfolio
# ---------------------------------------------------------------------------


def pi_star_factor(
    x: np.ndarray,
    r: float,
    mu_fn: ScalarFn,
    sigma_fn: ScalarFn,
    dsigma_fn: ScalarFn,
    nu_fn: ScalarFn,
    alpha: float,
    varrho: float,
) -> np.ndarray:
    """
    Optimal risky fraction under convention alpha.

    pi_t^* = [mu_eff(X_t) - r] / sigma(X_t)^2

    Parameters
    ----------
    x          : state-variable grid
    r          : risk-free rate
    (other)    : same as mu_eff_factor

    Returns
    -------
    pi_star : array same shape as x
    """
    x = np.asarray(x, dtype=np.float64)
    sigma = sigma_fn(x)
    if np.any(np.abs(sigma) < 1e-15):
        raise ValueError("sigma(x) must be bounded away from zero on the grid")
    mu_e = mu_eff_factor(x, mu_fn, sigma_fn, dsigma_fn, nu_fn, alpha, varrho)
    return (mu_e - r) / sigma ** 2


def special_case_checks(
    x: np.ndarray,
    r: float,
    mu_fn: ScalarFn,
    sigma_fn: ScalarFn,
    dsigma_fn: ScalarFn,
    nu_fn: ScalarFn,
    alpha: float,
) -> pd.DataFrame:
    """
    Verify the three covariation-channel elimination cases from the paper.

    Cases:
        (i)   varrho = 0        → covariation channel = 0
        (ii)  sigma'(x) = 0     → covariation channel = 0
        (iii) nu(x) = 0         → covariation channel = 0

    In all three cases the own-variance channel alpha still contributes.

    Parameters
    ----------
    x         : state grid
    r         : risk-free rate
    mu_fn     : nominal drift
    sigma_fn  : return volatility (must be non-zero and constant for case ii)
    dsigma_fn : derivative of sigma (set to zero fn for case ii)
    nu_fn     : factor volatility (set to zero fn for case iii)
    alpha     : convention parameter (must be > 0 to see the channel)

    Returns
    -------
    pd.DataFrame summarising each special case with covariation values.
    """
    zero_fn: ScalarFn = lambda xx: np.zeros_like(xx, dtype=np.float64)

    rows = []

    # Case (i): varrho = 0
    decomp_i = alpha_correction_decomposition(
        x, sigma_fn, dsigma_fn, nu_fn, alpha, varrho=0.0
    )
    rows.append(
        {
            "case": "(i) varrho=0",
            "max_covariation": float(np.max(np.abs(decomp_i["covariation"]))),
            "max_own_variance": float(np.max(np.abs(decomp_i["own_variance"]))),
        }
    )

    # Case (ii): sigma'(x) = 0 (constant sigma)
    decomp_ii = alpha_correction_decomposition(
        x, sigma_fn, zero_fn, nu_fn, alpha, varrho=0.7
    )
    rows.append(
        {
            "case": "(ii) sigma'=0",
            "max_covariation": float(np.max(np.abs(decomp_ii["covariation"]))),
            "max_own_variance": float(np.max(np.abs(decomp_ii["own_variance"]))),
        }
    )

    # Case (iii): nu(x) = 0
    decomp_iii = alpha_correction_decomposition(
        x, sigma_fn, dsigma_fn, zero_fn, alpha, varrho=0.7
    )
    rows.append(
        {
            "case": "(iii) nu=0",
            "max_covariation": float(np.max(np.abs(decomp_iii["covariation"]))),
            "max_own_variance": float(np.max(np.abs(decomp_iii["own_variance"]))),
        }
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Value-function ODE (optional Feynman–Kac check)
# ---------------------------------------------------------------------------


def ode_source_g(
    x: np.ndarray,
    r: float,
    rho: float,
    mu_fn: ScalarFn,
    sigma_fn: ScalarFn,
    dsigma_fn: ScalarFn,
    nu_fn: ScalarFn,
    alpha: float,
    varrho: float,
) -> np.ndarray:
    """
    Source term g(x) in the value-function ODE:
        g(x) = ln(rho) - 1 + r/rho + (1/(2*rho)) * [(mu_eff(x)-r)/sigma(x)]^2

    Parameters
    ----------
    (all as above)

    Returns
    -------
    g : array same shape as x
    """
    x = np.asarray(x, dtype=np.float64)
    mu_e = mu_eff_factor(x, mu_fn, sigma_fn, dsigma_fn, nu_fn, alpha, varrho)
    sigma = sigma_fn(x)
    sharpe_sq = ((mu_e - r) / sigma) ** 2
    return np.log(rho) - 1.0 + r / rho + sharpe_sq / (2.0 * rho)


# ---------------------------------------------------------------------------
# Default test-function specification (implementer choice, documented)
# ---------------------------------------------------------------------------

# Implementer-chosen test functions for numeric verification (§exp_3):
#   sigma(x) = 0.15 + 0.2 * x        (affine, bounded away from 0 on x>0)
#   nu(x)    = 0.1 + 0.05 * x        (affine)
#   mu(x)    = 0.08 + 0.1 * x        (affine)
#   b(x)     = 2.0 * (0.05 - x)      (mean-reverting)
# These are not imposed by the paper; they are chosen for illustration.


def default_sigma(x: np.ndarray) -> np.ndarray:
    """sigma(x) = 0.15 + 0.20*x  (implementer choice)."""
    return 0.15 + 0.20 * np.asarray(x, dtype=np.float64)


def default_dsigma(x: np.ndarray) -> np.ndarray:
    """sigma'(x) = 0.20  (constant)."""
    return 0.20 * np.ones_like(x, dtype=np.float64)


def default_nu(x: np.ndarray) -> np.ndarray:
    """nu(x) = 0.10 + 0.05*x  (implementer choice)."""
    return 0.10 + 0.05 * np.asarray(x, dtype=np.float64)


def default_dnu(x: np.ndarray) -> np.ndarray:
    """nu'(x) = 0.05  (constant)."""
    return 0.05 * np.ones_like(x, dtype=np.float64)


def default_mu(x: np.ndarray) -> np.ndarray:
    """mu(x) = 0.08 + 0.10*x  (implementer choice)."""
    return 0.08 + 0.10 * np.asarray(x, dtype=np.float64)


def default_b(x: np.ndarray) -> np.ndarray:
    """b(x) = 2.0*(0.05 - x)  (mean-reverting, implementer choice)."""
    return 2.0 * (0.05 - np.asarray(x, dtype=np.float64))
