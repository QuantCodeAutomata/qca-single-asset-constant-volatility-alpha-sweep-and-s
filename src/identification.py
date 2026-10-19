"""
Identification and non-identification of alpha from return laws.

Experiment 5: Shows that in the constant-volatility one-asset model, alpha
is not separately identified from return data absent structural restrictions
on mu, because only the effective Itô drift mu + alpha*sigma^2 enters the
return law.

Custom — Context7 found no library equivalent (identification argument).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Core identification functions
# ---------------------------------------------------------------------------


def equivalent_ito_drift_single(
    mu: float, sigma: float, alpha: float
) -> float:
    """
    Compute effective Itô drift mu_eff = mu + alpha * sigma^2.

    Parameters
    ----------
    mu    : nominal drift (alpha-convention)
    sigma : constant volatility
    alpha : convention parameter in [0, 1]

    Returns
    -------
    mu_eff : float
    """
    return mu + alpha * sigma ** 2


def logreturn_moments(
    mu_eff: float, sigma: float, T: float
) -> dict:
    """
    Log-return moments for an Itô GBM over horizon T.

    ln(S_T/S_0) ~ Normal(mean, variance) with:
        mean     = (mu_eff - 0.5*sigma^2) * T
        variance = sigma^2 * T

    Parameters
    ----------
    mu_eff : effective Itô drift
    sigma  : constant volatility
    T      : time horizon (years)

    Returns
    -------
    dict with keys 'mean', 'variance', 'std'
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    mean = (mu_eff - 0.5 * sigma ** 2) * T
    variance = sigma ** 2 * T
    return {"mean": mean, "variance": variance, "std": np.sqrt(variance)}


def construct_identification_pairs(
    m_bar: float,
    sigma: float,
    alpha_list: list[float],
) -> pd.DataFrame:
    """
    Construct multiple (mu, alpha) pairs sharing the same effective Itô drift
    m_bar = mu + alpha * sigma^2.

    Demonstrates non-identification: all pairs generate the same return law.

    Parameters
    ----------
    m_bar      : target effective Itô drift (held constant)
    sigma      : constant volatility
    alpha_list : list of alpha values in [0, 1]

    Returns
    -------
    DataFrame with columns: alpha, nominal_mu, mu_eff
    """
    rows = []
    for alpha in alpha_list:
        mu_nominal = m_bar - alpha * sigma ** 2
        mu_eff = equivalent_ito_drift_single(mu_nominal, sigma, alpha)
        rows.append(
            {
                "alpha": alpha,
                "nominal_mu": mu_nominal,
                "mu_eff": mu_eff,
            }
        )
    df = pd.DataFrame(rows)
    assert np.allclose(df["mu_eff"], m_bar), (
        "All mu_eff values should equal m_bar; identification construction failed"
    )
    return df


def identification_table(
    m_bar: float,
    r: float,
    sigma: float,
    alpha_list: list[float],
    T: float = 1.0,
) -> pd.DataFrame:
    """
    Full identification table with return law moments and optimal theta.

    Under the law-preserving parameterisation mu(alpha) = m_bar - alpha*sigma^2,
    theta^* = (m_bar - r)/sigma^2 is invariant in alpha.

    Parameters
    ----------
    m_bar      : target effective Itô drift
    r          : risk-free rate
    sigma      : constant volatility
    alpha_list : list of alpha values
    T          : return horizon for moments

    Returns
    -------
    DataFrame with columns:
        alpha, nominal_mu, mu_eff,
        mean_log_return_T, var_log_return_T,
        theta_star_law_preserving
    """
    pairs = construct_identification_pairs(m_bar, sigma, alpha_list)
    theta_invariant = (m_bar - r) / sigma ** 2

    records = []
    for _, row in pairs.iterrows():
        moments = logreturn_moments(row["mu_eff"], sigma, T)
        records.append(
            {
                "alpha": row["alpha"],
                "nominal_mu": row["nominal_mu"],
                "mu_eff": row["mu_eff"],
                "mean_log_return_T": moments["mean"],
                "var_log_return_T": moments["variance"],
                "theta_star_law_preserving": theta_invariant,
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Optional: multi-asset extension
# ---------------------------------------------------------------------------


def identification_table_multi(
    bar_mu: np.ndarray,
    r: float,
    V: np.ndarray,
    alpha_list: list[float],
) -> pd.DataFrame:
    """
    Multi-asset identification table.

    Under mu^(alpha) = bar_mu - alpha * d_V, the optimal policy
    theta^* = V^{-1}(bar_mu - r*1) is invariant in alpha.

    Parameters
    ----------
    bar_mu     : (n,) target effective Itô drift
    r          : risk-free rate
    V          : (n, n) covariance matrix
    alpha_list : list of alpha values

    Returns
    -------
    DataFrame with one row per alpha showing nominal mu, mu_eff, theta_star
    """
    import numpy.linalg as nla

    bar_mu = np.asarray(bar_mu, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d_V = np.diag(V)
    n = len(bar_mu)
    theta_invariant = nla.solve(V, bar_mu - r * np.ones(n))

    records = []
    for alpha in alpha_list:
        mu_nominal = bar_mu - alpha * d_V
        mu_eff = mu_nominal + alpha * d_V  # should equal bar_mu
        records.append(
            {
                "alpha": alpha,
                "nominal_mu": mu_nominal.tolist(),
                "mu_eff": mu_eff.tolist(),
                "theta_star": theta_invariant.tolist(),
                "mu_eff_matches_bar_mu": np.allclose(mu_eff, bar_mu),
            }
        )
    return pd.DataFrame(records)
