"""
Single-asset constant-volatility alpha-convention model.

Experiment 1: Replicates the one-risky-asset log-utility Merton results under
an alpha-interpreted stochastic integral convention.

Convention: dS_t/S_t = mu dt + sigma ∘_alpha dW_t
Itô form:   dS_t/S_t = (mu + alpha*sigma^2) dt + sigma dW_t

Reference: Paper §4 single-asset constant-volatility case.
Custom — Context7 found no library equivalent (formula-based closed-form).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core formula functions
# ---------------------------------------------------------------------------


def convert_to_ito_single(mu: float, sigma: float, alpha: float) -> float:
    """
    Convert the alpha-interpreted nominal drift to the effective Itô drift.

    The alpha-to-Itô rule for a one-asset constant-volatility model is:
        mu_Ito = mu + alpha * sigma^2

    Parameters
    ----------
    mu    : nominal drift coefficient (alpha-convention)
    sigma : constant volatility (sigma > 0)
    alpha : stochastic-integral convention parameter in [0, 1]
              0 → Itô, 0.5 → Stratonovich, 1 → full-anticipating

    Returns
    -------
    mu_Ito : effective Itô drift

    Raises
    ------
    ValueError if sigma <= 0 or alpha not in [0, 1].
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return mu + alpha * sigma ** 2


def optimal_controls_single(
    mu: float,
    r: float,
    sigma: float,
    alpha: float,
    rho: float | None = None,
) -> dict:
    """
    Log-utility optimal controls for the single-asset Merton problem under
    convention alpha.

    The optimal risky weight is:
        theta_alpha^* = (mu + alpha*sigma^2 - r) / sigma^2
                      = (mu - r)/sigma^2 + alpha

    The optimal consumption ratio is c_t^* / a_t = rho (independent of alpha).

    Parameters
    ----------
    mu    : nominal drift (alpha-convention)
    r     : risk-free rate (r > 0)
    sigma : constant volatility (sigma > 0)
    alpha : convention parameter in [0, 1]
    rho   : subjective discount rate (required for value-function evaluation)

    Returns
    -------
    dict with keys:
        'theta'             : optimal risky weight theta_alpha^*
        'consumption_ratio' : c_t^*/a_t = rho (None if rho not provided)
        'mu_ito'            : effective Itô drift = mu + alpha*sigma^2
    """
    mu_ito = convert_to_ito_single(mu, sigma, alpha)
    theta = (mu_ito - r) / sigma ** 2
    return {
        "theta": theta,
        "consumption_ratio": rho,
        "mu_ito": mu_ito,
    }


def value_constant_single(
    mu: float,
    r: float,
    sigma: float,
    alpha: float,
    rho: float,
) -> float:
    """
    Value-function constant beta_0 for the log-utility Merton problem under
    convention alpha with effective Itô drift mu_eff = mu + alpha*sigma^2.

    Classical Itô formula (equation from §4 of the paper):
        beta_0 = (1/rho)*(r/rho + ln(rho) - 1) + (1/(2*rho^2))*((mu_eff-r)^2/sigma^2)

    Parameters
    ----------
    mu    : nominal drift (alpha-convention)
    r     : risk-free rate
    sigma : constant volatility
    alpha : convention parameter in [0, 1]
    rho   : subjective discount rate (rho > 0)

    Returns
    -------
    beta_0 : scalar value-function constant
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    mu_eff = convert_to_ito_single(mu, sigma, alpha)
    # Custom — Context7 found no library equivalent (paper Eq. beta_0 formula)
    beta_0 = (1.0 / rho) * (r / rho + np.log(rho) - 1.0) + (
        (mu_eff - r) ** 2 / sigma ** 2
    ) / (2.0 * rho ** 2)
    return beta_0


# ---------------------------------------------------------------------------
# Figure 1 sweep
# ---------------------------------------------------------------------------

# Paper-defined parameter sets for Figure 1 replication
FIGURE1_PARAMS: list[dict] = [
    {
        "label": "Low-vol defensive",
        "mu_minus_r": 0.03,
        "sigma": 0.12,
        "theta_ito": 0.03 / 0.12 ** 2,  # ≈ 2.0833
    },
    {
        "label": "Broad-equity index",
        "mu_minus_r": 0.05,
        "sigma": 0.18,
        "theta_ito": 0.05 / 0.18 ** 2,  # ≈ 1.5432
    },
    {
        "label": "High-vol growth",
        "mu_minus_r": 0.08,
        "sigma": 0.35,
        "theta_ito": 0.08 / 0.35 ** 2,  # ≈ 0.6531
    },
]


def sweep_theta_alpha(
    mu_minus_r: float,
    sigma: float,
    alpha_grid: np.ndarray,
    r: float = 0.0,
) -> np.ndarray:
    """
    Evaluate the optimal risky weight theta_alpha^* = (mu-r)/sigma^2 + alpha
    over an alpha grid, holding nominal excess drift mu-r fixed.

    Parameters
    ----------
    mu_minus_r : nominal excess drift mu - r (fixed across alpha)
    sigma      : constant volatility
    alpha_grid : 1-D array of alpha values in [0, 1]
    r          : risk-free rate (default 0, only used to compute theta_0)

    Returns
    -------
    theta_alpha : array of optimal weights, shape = (len(alpha_grid),)
    """
    theta_ito = mu_minus_r / sigma ** 2
    # theta_alpha^* = theta_ito + alpha  (slope = 1 exactly)
    return theta_ito + alpha_grid


def build_figure1_table(
    alpha_grid: np.ndarray,
    params: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Build a summary DataFrame of theta_alpha^* for every parameter set and
    selected alpha values (0, 0.5, 1).

    Parameters
    ----------
    alpha_grid : full alpha grid (used to check slope)
    params     : list of dicts with keys label, mu_minus_r, sigma

    Returns
    -------
    pd.DataFrame with columns: label, alpha_0, alpha_0.5, alpha_1,
                                theta_ito, slope_check
    """
    if params is None:
        params = FIGURE1_PARAMS
    rows = []
    for p in params:
        thetas = sweep_theta_alpha(p["mu_minus_r"], p["sigma"], alpha_grid)
        # slope should be 1.0 within floating-point tolerance
        slope = np.polyfit(alpha_grid, thetas, 1)[0]
        rows.append(
            {
                "label": p["label"],
                "sigma": p["sigma"],
                "mu_minus_r": p["mu_minus_r"],
                "theta_ito (alpha=0)": thetas[alpha_grid == 0.0][0],
                "theta (alpha=0.5)": thetas[np.isclose(alpha_grid, 0.5)][0],
                "theta (alpha=1)": thetas[alpha_grid == 1.0][0],
                "slope": slope,
                "Strat_shift": thetas[np.isclose(alpha_grid, 0.5)][0]
                - thetas[alpha_grid == 0.0][0],
            }
        )
    return pd.DataFrame(rows)


def stratonovich_shift(sigma: float) -> float:
    """
    Compute the Stratonovich-vs-Itô risky weight shift.

    For constant volatility: delta = theta_{1/2}^* - theta_0^* = 1/2.

    Parameters
    ----------
    sigma : constant volatility (not used in the result but kept for signature
            clarity in the multi-asset generalisation)

    Returns
    -------
    0.5  (exact)
    """
    # The shift is always exactly alpha = 0.5, independent of sigma.
    return 0.5
