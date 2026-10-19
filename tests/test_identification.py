"""
Tests for src/identification.py – Experiment 5.

Verifies:
  - equivalent_ito_drift_single = mu + alpha*sigma^2
  - Log-return moments identical for law-preserving pairs
  - Optimal theta invariant under law-preserving parameterisation
  - Multi-asset extension
"""

import numpy as np
import pytest

from src.identification import (
    construct_identification_pairs,
    equivalent_ito_drift_single,
    identification_table,
    identification_table_multi,
    logreturn_moments,
)
from src.multi_asset_alpha import MATRIX_B


# ---------------------------------------------------------------------------
# equivalent_ito_drift_single
# ---------------------------------------------------------------------------


def test_equivalent_ito_alpha0():
    """At alpha=0, mu_eff = mu."""
    assert np.isclose(equivalent_ito_drift_single(0.08, 0.20, 0.0), 0.08)


def test_equivalent_ito_alpha1():
    """At alpha=1, mu_eff = mu + sigma^2."""
    sigma = 0.20
    mu = 0.08
    assert np.isclose(equivalent_ito_drift_single(mu, sigma, 1.0), mu + sigma ** 2)


def test_equivalent_ito_linearity():
    """mu_eff is linear in alpha."""
    sigma, mu = 0.25, 0.10
    mu_eff_0 = equivalent_ito_drift_single(mu, sigma, 0.0)
    mu_eff_1 = equivalent_ito_drift_single(mu, sigma, 1.0)
    for alpha in np.linspace(0, 1, 21):
        mu_eff_a = equivalent_ito_drift_single(mu, sigma, alpha)
        expected = (1 - alpha) * mu_eff_0 + alpha * mu_eff_1
        assert np.isclose(mu_eff_a, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# logreturn_moments
# ---------------------------------------------------------------------------


def test_logreturn_mean():
    """Mean log-return = (mu_eff - 0.5*sigma^2)*T."""
    mu_eff, sigma, T = 0.08, 0.20, 1.0
    mom = logreturn_moments(mu_eff, sigma, T)
    expected_mean = (mu_eff - 0.5 * sigma ** 2) * T
    assert np.isclose(mom["mean"], expected_mean)


def test_logreturn_variance():
    """Variance of log-return = sigma^2 * T."""
    sigma, T = 0.20, 2.0
    mom = logreturn_moments(0.08, sigma, T)
    assert np.isclose(mom["variance"], sigma ** 2 * T)


def test_logreturn_std():
    """std = sqrt(variance)."""
    mom = logreturn_moments(0.08, 0.20, 1.0)
    assert np.isclose(mom["std"], np.sqrt(mom["variance"]))


def test_logreturn_invalid_sigma():
    """sigma <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="sigma"):
        logreturn_moments(0.08, 0.0, 1.0)


def test_logreturn_invalid_T():
    """T <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="T"):
        logreturn_moments(0.08, 0.20, -1.0)


# ---------------------------------------------------------------------------
# construct_identification_pairs
# ---------------------------------------------------------------------------


def test_identification_pairs_mu_eff():
    """All pairs must share the same effective Itô drift m_bar."""
    m_bar, sigma = 0.08, 0.20
    pairs = construct_identification_pairs(m_bar, sigma, [0.0, 0.5, 1.0])
    assert np.allclose(pairs["mu_eff"], m_bar, atol=1e-14)


def test_identification_pairs_alpha():
    """Alpha column must match input list."""
    alpha_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    pairs = construct_identification_pairs(0.09, 0.20, alpha_list)
    assert list(pairs["alpha"]) == alpha_list


def test_identification_pairs_mu_decreases():
    """Nominal mu must decrease as alpha increases (for fixed m_bar)."""
    pairs = construct_identification_pairs(0.08, 0.20, [0.0, 0.5, 1.0])
    mus = list(pairs["nominal_mu"])
    assert mus[0] > mus[1] > mus[2]


def test_identification_pairs_edge_alpha0():
    """At alpha=0, nominal mu = m_bar."""
    m_bar = 0.09
    pairs = construct_identification_pairs(m_bar, 0.20, [0.0])
    assert np.isclose(float(pairs["nominal_mu"].iloc[0]), m_bar, atol=1e-14)


# ---------------------------------------------------------------------------
# identification_table
# ---------------------------------------------------------------------------


def test_id_table_theta_star_invariant():
    """theta_star_law_preserving must equal (m_bar-r)/sigma^2 for all alpha."""
    m_bar, sigma, r = 0.08, 0.20, 0.03
    expected_theta = (m_bar - r) / sigma ** 2
    table = identification_table(m_bar, r, sigma, [0.0, 0.25, 0.5, 0.75, 1.0], T=1.0)
    assert np.allclose(
        table["theta_star_law_preserving"], expected_theta, atol=1e-14
    )


def test_id_table_moments_identical():
    """All rows must have the same mean and variance under law-preserving parameterisation."""
    table = identification_table(0.08, 0.03, 0.20, [0.0, 0.5, 1.0], T=1.0)
    means = table["mean_log_return_T"].values
    vars_ = table["var_log_return_T"].values
    assert np.allclose(means, means[0], atol=1e-14)
    assert np.allclose(vars_, vars_[0], atol=1e-14)


def test_id_table_mu_eff_constant():
    """mu_eff column must be constant across all rows."""
    m_bar = 0.08
    table = identification_table(m_bar, 0.03, 0.20, [0.0, 0.5, 1.0], T=1.0)
    assert np.allclose(table["mu_eff"], m_bar, atol=1e-14)


# ---------------------------------------------------------------------------
# identification_table_multi
# ---------------------------------------------------------------------------


def test_id_table_multi_mu_eff_matches():
    """Multi-asset: mu_eff must equal bar_mu for all alpha."""
    bar_mu = np.array([0.10, 0.14])
    table = identification_table_multi(bar_mu, 0.03, MATRIX_B,
                                       [0.0, 0.5, 1.0])
    assert table["mu_eff_matches_bar_mu"].all()


def test_id_table_multi_theta_star_invariant():
    """Multi-asset: theta_star should be the same for all alpha (stored as list)."""
    bar_mu = np.array([0.10, 0.14])
    table = identification_table_multi(bar_mu, 0.03, MATRIX_B, [0.0, 0.5, 1.0])
    # All theta_star entries should be equal (as numpy arrays)
    theta_0 = table["theta_star"].iloc[0]
    for theta in table["theta_star"]:
        assert np.allclose(theta, theta_0, atol=1e-12)


def test_id_table_multi_alpha_column():
    alpha_list = [0.0, 0.5, 1.0]
    bar_mu = np.array([0.10, 0.14])
    table = identification_table_multi(bar_mu, 0.03, MATRIX_B, alpha_list)
    assert list(table["alpha"]) == alpha_list
