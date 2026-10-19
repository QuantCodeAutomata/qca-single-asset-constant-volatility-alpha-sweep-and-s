"""
Tests for src/multi_asset_alpha.py – Experiment 2.

Verifies:
  - Itô drift conversion componentwise
  - Optimal theta formula via linear solve
  - Alpha sensitivity d theta / d alpha = V^{-1} d_V
  - Remark 4.2 specific matrix outputs
  - Law-preserving invariance
  - Input validation
"""

import numpy as np
import pytest

from src.multi_asset_alpha import (
    MATRIX_A,
    MATRIX_B,
    MATRIX_C,
    alpha_sensitivity_multi,
    check_positive_definite,
    ito_drift_multi,
    law_preserving_mu,
    optimal_theta_multi,
    remark42_sensitivity_table,
    variance_vector,
    verify_law_preserving_invariance,
)


# ---------------------------------------------------------------------------
# variance_vector
# ---------------------------------------------------------------------------


def test_variance_vector_diagonal():
    V = np.array([[4.0, 1.0], [1.0, 9.0]])
    d_V = variance_vector(V)
    assert np.allclose(d_V, np.array([4.0, 9.0]))


# ---------------------------------------------------------------------------
# check_positive_definite
# ---------------------------------------------------------------------------


def test_positive_definite_ok():
    V = np.eye(3) * 2.0
    assert check_positive_definite(V) is True


def test_positive_definite_fails():
    V = np.array([[1.0, 2.0], [2.0, 1.0]])  # not PD
    assert check_positive_definite(V) is False


# ---------------------------------------------------------------------------
# ito_drift_multi
# ---------------------------------------------------------------------------


def test_ito_drift_alpha0():
    """At alpha=0, mu_Ito = mu."""
    V = MATRIX_B
    mu = np.array([0.08, 0.12])
    mu_ito = ito_drift_multi(mu, V, 0.0)
    assert np.allclose(mu_ito, mu)


def test_ito_drift_alpha1():
    """At alpha=1, mu_Ito = mu + d_V."""
    V = MATRIX_B
    mu = np.array([0.08, 0.12])
    d_V = variance_vector(V)
    mu_ito = ito_drift_multi(mu, V, 1.0)
    assert np.allclose(mu_ito, mu + d_V)


def test_ito_drift_stratonovich():
    """At alpha=0.5, mu_Ito = mu + 0.5*d_V."""
    V = MATRIX_C
    mu = np.ones(3) * 0.09
    d_V = variance_vector(V)
    mu_ito = ito_drift_multi(mu, V, 0.5)
    assert np.allclose(mu_ito, mu + 0.5 * d_V)


# ---------------------------------------------------------------------------
# optimal_theta_multi
# ---------------------------------------------------------------------------


def test_theta_multi_ito_case():
    """At alpha=0, theta = V^{-1}(mu - r1)."""
    V = MATRIX_B
    mu = np.array([0.10, 0.15])
    r = 0.03
    theta = optimal_theta_multi(mu, r, V, 0.0)
    expected = np.linalg.solve(V, mu - r * np.ones(2))
    assert np.allclose(theta, expected, atol=1e-12)


def test_theta_multi_linear_in_alpha():
    """theta(alpha) = theta(0) + alpha * V^{-1} d_V exactly."""
    V = MATRIX_B
    mu = np.array([0.10, 0.15])
    r = 0.03
    theta_0 = optimal_theta_multi(mu, r, V, 0.0)
    sens = alpha_sensitivity_multi(V)
    for alpha in np.linspace(0, 1, 11):
        theta_a = optimal_theta_multi(mu, r, V, alpha)
        expected = theta_0 + alpha * sens
        assert np.allclose(theta_a, expected, atol=1e-12), (
            f"Linearity failed at alpha={alpha}"
        )


def test_theta_multi_invalid_V():
    """Non-positive-definite V should raise ValueError."""
    V_bad = np.array([[1.0, 3.0], [3.0, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        optimal_theta_multi(np.array([0.08, 0.12]), 0.03, V_bad, 0.5)


# ---------------------------------------------------------------------------
# alpha_sensitivity_multi – Remark 4.2
# ---------------------------------------------------------------------------


def test_remark42_matrix_A():
    """Matrix A: V^{-1} d_V must equal (1,1,1,1)."""
    sens = alpha_sensitivity_multi(MATRIX_A)
    assert np.allclose(sens, np.ones(4), atol=1e-10), (
        f"Matrix A sensitivity: {sens}"
    )


def test_remark42_matrix_B():
    """Matrix B: V^{-1} d_V ≈ (-1.143, 1.429)."""
    sens = alpha_sensitivity_multi(MATRIX_B)
    expected = np.array([-2.0 / 1.75, 2.5 / 1.75])
    assert np.allclose(sens, expected, atol=1e-6), (
        f"Matrix B sensitivity: {sens} vs {expected}"
    )


def test_remark42_matrix_B_negative_component():
    """Matrix B must have a negative first sensitivity component."""
    sens = alpha_sensitivity_multi(MATRIX_B)
    assert sens[0] < 0, "Expected negative first component for Matrix B"


def test_remark42_matrix_C():
    """Matrix C: V^{-1} d_V ≈ (0.803, 0.619, 0.734)."""
    sens = alpha_sensitivity_multi(MATRIX_C)
    expected = np.array([175.0, 135.0, 160.0]) / 218.0
    assert np.allclose(sens, expected, atol=0.01), (
        f"Matrix C sensitivity: {sens} vs {expected}"
    )


def test_remark42_matrix_C_all_positive():
    """Matrix C: all components positive (equal unit variances, positive correlations)."""
    sens = alpha_sensitivity_multi(MATRIX_C)
    assert np.all(sens > 0), f"Matrix C: expected all positive, got {sens}"


def test_remark42_table_shape():
    df = remark42_sensitivity_table()
    assert df.shape[0] == 3  # three matrices
    assert "label" in df.columns


# ---------------------------------------------------------------------------
# law_preserving_mu
# ---------------------------------------------------------------------------


def test_law_preserving_mu_alpha0():
    """At alpha=0, mu^(0) = bar_mu."""
    V = MATRIX_B
    bar_mu = np.array([0.10, 0.14])
    d_V = variance_vector(V)
    assert np.allclose(law_preserving_mu(bar_mu, d_V, 0.0), bar_mu)


def test_law_preserving_mu_effective_drift():
    """mu^(alpha) + alpha*d_V = bar_mu for all alpha."""
    V = MATRIX_B
    bar_mu = np.array([0.10, 0.14])
    d_V = variance_vector(V)
    for alpha in np.linspace(0, 1, 11):
        mu_a = law_preserving_mu(bar_mu, d_V, alpha)
        mu_eff = mu_a + alpha * d_V
        assert np.allclose(mu_eff, bar_mu, atol=1e-14)


# ---------------------------------------------------------------------------
# verify_law_preserving_invariance
# ---------------------------------------------------------------------------


def test_law_preserving_invariance_identity():
    """theta must be invariant across alpha under law-preserving parameterisation."""
    result = verify_law_preserving_invariance(
        np.array([0.10, 0.14]), 0.03, MATRIX_B, np.linspace(0, 1, 51)
    )
    assert result["is_invariant"], (
        f"Invariance failed; max_deviation={result['max_deviation']}"
    )
    assert result["max_deviation"] < 1e-12


def test_law_preserving_invariance_3d():
    """3-asset version should also be invariant."""
    bar_mu = np.array([0.09, 0.11, 0.13])
    result = verify_law_preserving_invariance(
        bar_mu, 0.03, MATRIX_C, np.linspace(0, 1, 51)
    )
    assert result["is_invariant"]
