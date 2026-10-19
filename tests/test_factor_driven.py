"""
Tests for src/factor_driven_alpha.py – Experiment 3.

Verifies:
  - mu_eff decomposition formula
  - b_alpha factor drift conversion
  - pi* optimality formula
  - Special-case elimination of covariation channel
  - Edge cases and input validation
"""

import numpy as np
import pytest

from src.factor_driven_alpha import (
    alpha_correction_decomposition,
    b_alpha_factor,
    default_b,
    default_dsigma,
    default_dnu,
    default_mu,
    default_nu,
    default_sigma,
    mu_eff_factor,
    pi_star_factor,
    special_case_checks,
)

X_GRID = np.linspace(0.05, 0.50, 100)
R = 0.03


# ---------------------------------------------------------------------------
# mu_eff_factor
# ---------------------------------------------------------------------------


def test_mu_eff_alpha0_equals_nominal():
    """At alpha=0, mu_eff must equal mu (no correction)."""
    mu_e = mu_eff_factor(X_GRID, default_mu, default_sigma, default_dsigma, default_nu,
                          alpha=0.0, varrho=0.5)
    assert np.allclose(mu_e, default_mu(X_GRID), atol=1e-14)


def test_mu_eff_own_variance_channel():
    """With varrho=0, correction equals alpha*sigma(x)^2 only."""
    alpha = 0.7
    mu_e = mu_eff_factor(X_GRID, default_mu, default_sigma, default_dsigma, default_nu,
                          alpha=alpha, varrho=0.0)
    expected = default_mu(X_GRID) + alpha * default_sigma(X_GRID) ** 2
    assert np.allclose(mu_e, expected, atol=1e-14)


def test_mu_eff_full_decomposition():
    """mu_eff - mu = alpha*sigma^2 + alpha*varrho*sigma'*nu."""
    alpha, varrho = 0.5, -0.3
    mu_e = mu_eff_factor(X_GRID, default_mu, default_sigma, default_dsigma, default_nu,
                          alpha, varrho)
    correction = mu_e - default_mu(X_GRID)
    expected_correction = (
        alpha * default_sigma(X_GRID) ** 2
        + alpha * varrho * default_dsigma(X_GRID) * default_nu(X_GRID)
    )
    assert np.allclose(correction, expected_correction, atol=1e-14)


def test_mu_eff_constant_sigma_no_covariation():
    """If sigma is constant (dsigma=0), covariation channel vanishes."""
    const_sigma = lambda x: np.full_like(x, 0.20)
    zero_dsigma = lambda x: np.zeros_like(x)
    alpha, varrho = 0.8, 0.6
    mu_e = mu_eff_factor(X_GRID, default_mu, const_sigma, zero_dsigma, default_nu,
                          alpha, varrho)
    expected = default_mu(X_GRID) + alpha * const_sigma(X_GRID) ** 2
    assert np.allclose(mu_e, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# b_alpha_factor
# ---------------------------------------------------------------------------


def test_b_alpha_alpha0():
    """At alpha=0, b_alpha = b (unchanged)."""
    b_a = b_alpha_factor(X_GRID, default_b, default_nu, default_dnu, alpha=0.0)
    assert np.allclose(b_a, default_b(X_GRID), atol=1e-14)


def test_b_alpha_correction_formula():
    """b_alpha = b + alpha*nu*nu' for all alpha."""
    for alpha in [0.25, 0.5, 1.0]:
        b_a = b_alpha_factor(X_GRID, default_b, default_nu, default_dnu, alpha)
        expected = default_b(X_GRID) + alpha * default_nu(X_GRID) * default_dnu(X_GRID)
        assert np.allclose(b_a, expected, atol=1e-14)


def test_b_alpha_zero_nu():
    """If nu=0, b_alpha = b for any alpha."""
    zero_nu = lambda x: np.zeros_like(x)
    zero_dnu = lambda x: np.zeros_like(x)
    for alpha in [0.0, 0.5, 1.0]:
        b_a = b_alpha_factor(X_GRID, default_b, zero_nu, zero_dnu, alpha)
        assert np.allclose(b_a, default_b(X_GRID), atol=1e-14)


# ---------------------------------------------------------------------------
# alpha_correction_decomposition
# ---------------------------------------------------------------------------


def test_decomposition_own_variance_correct():
    """Own-variance component = alpha * sigma(x)^2."""
    alpha = 0.6
    decomp = alpha_correction_decomposition(
        X_GRID, default_sigma, default_dsigma, default_nu, alpha, varrho=0.4
    )
    expected_ov = alpha * default_sigma(X_GRID) ** 2
    assert np.allclose(decomp["own_variance"], expected_ov, atol=1e-14)


def test_decomposition_total_is_sum():
    """Total correction = own-variance + covariation."""
    alpha, varrho = 0.5, 0.3
    decomp = alpha_correction_decomposition(
        X_GRID, default_sigma, default_dsigma, default_nu, alpha, varrho
    )
    assert np.allclose(
        decomp["total_correction"],
        decomp["own_variance"] + decomp["covariation"],
        atol=1e-14,
    )


# ---------------------------------------------------------------------------
# pi_star_factor
# ---------------------------------------------------------------------------


def test_pi_star_positive_when_mu_exceeds_r():
    """For x where mu(x) > r and sigma(x) > 0, pi* should be positive (at alpha=0)."""
    pi = pi_star_factor(X_GRID, R, default_mu, default_sigma, default_dsigma, default_nu,
                         alpha=0.0, varrho=0.0)
    assert np.all(pi > 0)


def test_pi_star_formula_explicit():
    """pi* = (mu_eff - r) / sigma^2."""
    alpha, varrho = 0.4, 0.2
    pi = pi_star_factor(X_GRID, R, default_mu, default_sigma, default_dsigma, default_nu,
                         alpha, varrho)
    mu_e = mu_eff_factor(X_GRID, default_mu, default_sigma, default_dsigma, default_nu,
                          alpha, varrho)
    expected = (mu_e - R) / default_sigma(X_GRID) ** 2
    assert np.allclose(pi, expected, atol=1e-14)


def test_pi_star_single_point():
    """Test pi_star works on a single-element array."""
    x = np.array([0.30])
    pi = pi_star_factor(x, R, default_mu, default_sigma, default_dsigma, default_nu,
                         0.5, 0.0)
    assert pi.shape == (1,)
    assert np.isfinite(pi[0])


# ---------------------------------------------------------------------------
# special_case_checks
# ---------------------------------------------------------------------------


def test_special_cases_covariation_vanishes():
    """All three special cases must have near-zero covariation channel."""
    df = special_case_checks(X_GRID, R, default_mu, default_sigma, default_dsigma,
                              default_nu, alpha=0.7)
    for _, row in df.iterrows():
        assert row["max_covariation"] < 1e-14, (
            f"Non-zero covariation in {row['case']}: {row['max_covariation']}"
        )


def test_special_cases_own_variance_persists():
    """Own-variance channel must persist in all three special cases."""
    df = special_case_checks(X_GRID, R, default_mu, default_sigma, default_dsigma,
                              default_nu, alpha=0.7)
    for _, row in df.iterrows():
        assert row["max_own_variance"] > 0, (
            f"Own-variance vanished in {row['case']}"
        )


def test_special_cases_three_rows():
    """special_case_checks must return exactly 3 rows."""
    df = special_case_checks(X_GRID, R, default_mu, default_sigma, default_dsigma,
                              default_nu, alpha=0.5)
    assert len(df) == 3
