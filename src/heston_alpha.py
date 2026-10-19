"""
Heston-model specialisation: alpha-convention policy curves.

Experiment 4: Replicates the paper's Heston-model alpha results, including
converted variance dynamics, closed-form optimal risky fraction, Figure 2
policy curves, sign-change threshold, strict Feller admissibility, and
stationary short-region probability.

Heston model (alpha-interpreted):
    dS_t/S_t  = mu dt + sqrt(V_t) ∘_alpha dW_t^S
    dV_t      = kappa*(theta - V_t) dt + xi*sqrt(V_t) ∘_alpha dW_t^X
    d<W^S,W^X>_t = varrho dt

Itô conversions (paper §5 Heston specialisation):
    S-process:  dS_t/S_t = [mu + alpha*(V_t + varrho*xi/2)] dt + sqrt(V_t) dW_t^S
    V-process:  dV_t = kappa*(theta_alpha - V_t) dt + xi*sqrt(V_t) dW_t^X
                with theta_alpha = theta + alpha*xi^2 / (2*kappa)

Optimal controls (log-utility):
    c_t^* = rho * A_t
    pi_t^* = alpha + K_alpha / V_t
    K_alpha = (mu - r) + alpha * varrho * xi / 2

Paper calibration (Table 2 / Figure 2):
    kappa=2, theta=0.045, xi=0.4, varrho=-0.5, mu-r=0.06

Custom — Context7 found no library equivalent (paper Heston formulas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist


# ---------------------------------------------------------------------------
# Paper calibration
# ---------------------------------------------------------------------------

PAPER_PARAMS = {
    "kappa": 2.0,
    "theta": 0.045,
    "xi": 0.4,
    "varrho": -0.5,
    "mu_minus_r": 0.06,
}


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def heston_theta_alpha(
    theta: float, xi: float, kappa: float, alpha: float
) -> float:
    """
    Long-run mean of the converted Heston variance process under convention alpha.

    theta_alpha = theta + alpha * xi^2 / (2 * kappa)

    Parameters
    ----------
    theta  : Itô (alpha=0) long-run variance mean
    xi     : variance volatility (vol-of-vol)
    kappa  : mean-reversion speed
    alpha  : convention parameter in [0, 1]

    Returns
    -------
    theta_alpha : shifted long-run mean
    """
    return theta + alpha * xi ** 2 / (2.0 * kappa)


def heston_K_alpha(
    mu_minus_r: float, varrho: float, xi: float, alpha: float
) -> float:
    """
    Policy coefficient K_alpha in pi_t^* = alpha + K_alpha / V_t.

    K_alpha = (mu - r) + alpha * varrho * xi / 2

    Parameters
    ----------
    mu_minus_r : excess return mu - r
    varrho     : correlation between W^S and W^X
    xi         : variance volatility
    alpha      : convention parameter in [0, 1]

    Returns
    -------
    K_alpha : policy coefficient
    """
    return mu_minus_r + alpha * varrho * xi / 2.0


def heston_pi_star(
    V: np.ndarray, alpha: float, K_alpha: float
) -> np.ndarray:
    """
    Optimal risky fraction in the Heston log-utility problem.

    pi_t^* = alpha + K_alpha / V_t

    Parameters
    ----------
    V       : variance level(s) (must be > 0)
    alpha   : convention parameter in [0, 1]
    K_alpha : policy coefficient from heston_K_alpha

    Returns
    -------
    pi_star : array same shape as V
    """
    V = np.asarray(V, dtype=np.float64)
    if np.any(V <= 0):
        raise ValueError("V must be strictly positive")
    return alpha + K_alpha / V


def sign_change_threshold(alpha: float, K_alpha: float) -> float | None:
    """
    Variance level V^* at which pi_t^* = 0.

    V^* = -K_alpha / alpha  (exists when alpha > 0 and K_alpha < 0)

    Parameters
    ----------
    alpha   : convention parameter (must be > 0)
    K_alpha : policy coefficient

    Returns
    -------
    V_star : float if a positive threshold exists, else None
    """
    if alpha == 0.0:
        return None
    threshold = -K_alpha / alpha
    return float(threshold) if threshold > 0 else None


# ---------------------------------------------------------------------------
# Strict Feller admissibility
# ---------------------------------------------------------------------------


def strict_feller(kappa: float, theta_alpha: float, xi: float) -> dict:
    """
    Check the strict Feller condition 2*kappa*theta_alpha > xi^2.

    The condition ensures the CIR/Heston variance process stays strictly
    positive and that the inverse-moment E[1/V_t] exists under the stationary
    distribution, which is required for the log-utility value function.

    Parameters
    ----------
    kappa       : mean-reversion speed
    theta_alpha : long-run mean under convention alpha
    xi          : variance volatility

    Returns
    -------
    dict with keys:
        'lhs'        : 2 * kappa * theta_alpha
        'rhs'        : xi^2
        'satisfied'  : bool
        'margin'     : lhs - rhs
    """
    lhs = 2.0 * kappa * theta_alpha
    rhs = xi ** 2
    return {
        "lhs": lhs,
        "rhs": rhs,
        "satisfied": bool(lhs > rhs),
        "margin": lhs - rhs,
    }


# ---------------------------------------------------------------------------
# Stationary distribution (CIR Gamma invariant)
# ---------------------------------------------------------------------------


def stationary_gamma_params(
    kappa: float, theta_alpha: float, xi: float
) -> dict:
    """
    Parameters of the Gamma stationary distribution of the CIR variance
    process under convention alpha.

    The stationary law is Gamma(a, b) where:
        a = 2 * kappa * theta_alpha / xi^2   (shape)
        b = xi^2 / (2 * kappa)               (scale)

    Parameters
    ----------
    kappa       : mean-reversion speed
    theta_alpha : long-run mean
    xi          : variance volatility

    Returns
    -------
    dict with keys 'shape' (a) and 'scale' (b)
    """
    a = 2.0 * kappa * theta_alpha / xi ** 2
    b = xi ** 2 / (2.0 * kappa)
    return {"shape": a, "scale": b}


def stationary_short_prob(
    threshold_V: float, kappa: float, theta_alpha: float, xi: float
) -> float:
    """
    Probability that the stationary variance V is below threshold_V under
    the Gamma invariant distribution of the converted CIR process.

    P(V < threshold_V) evaluated using the Gamma CDF.

    # Using scipy.stats.gamma — Context7 confirmed

    Parameters
    ----------
    threshold_V : variance threshold (e.g. 0.04 for 20% vol)
    kappa       : mean-reversion speed
    theta_alpha : long-run mean under convention alpha
    xi          : variance volatility

    Returns
    -------
    prob : float in [0, 1]
    """
    params = stationary_gamma_params(kappa, theta_alpha, xi)
    return float(gamma_dist.cdf(threshold_V, a=params["shape"], scale=params["scale"]))


# ---------------------------------------------------------------------------
# Summary table (Figure 2 key quantities)
# ---------------------------------------------------------------------------


def figure2_summary_table(
    alpha_values: list[float] | None = None,
) -> pd.DataFrame:
    """
    Reproduce the Figure 2 key-quantity summary table for the paper calibration.

    Computes for each alpha: theta_alpha, sqrt(theta_alpha), K_alpha,
    pi^*(theta_alpha), Feller check, sign threshold, stationary short prob.

    Parameters
    ----------
    alpha_values : list of alpha values (default [0, 0.5, 1])

    Returns
    -------
    pd.DataFrame
    """
    if alpha_values is None:
        alpha_values = [0.0, 0.5, 1.0]
    kappa = PAPER_PARAMS["kappa"]
    theta = PAPER_PARAMS["theta"]
    xi = PAPER_PARAMS["xi"]
    varrho = PAPER_PARAMS["varrho"]
    mu_minus_r = PAPER_PARAMS["mu_minus_r"]

    rows = []
    for alpha in alpha_values:
        ta = heston_theta_alpha(theta, xi, kappa, alpha)
        Ka = heston_K_alpha(mu_minus_r, varrho, xi, alpha)
        feller = strict_feller(kappa, ta, xi)
        pi_at_mean = heston_pi_star(np.array([ta]), alpha, Ka)[0]
        V_star = sign_change_threshold(alpha, Ka)
        short_prob = stationary_short_prob(0.04, kappa, ta, xi)
        rows.append(
            {
                "alpha": alpha,
                "theta_alpha": ta,
                "sqrt(theta_alpha)": np.sqrt(ta),
                "K_alpha": Ka,
                "pi*(theta_alpha)": pi_at_mean,
                "Feller_satisfied": feller["satisfied"],
                "sign_threshold_V": V_star,
                "sign_threshold_vol": np.sqrt(V_star) if V_star is not None else None,
                "P(V<0.04)": short_prob,
            }
        )
    return pd.DataFrame(rows)
