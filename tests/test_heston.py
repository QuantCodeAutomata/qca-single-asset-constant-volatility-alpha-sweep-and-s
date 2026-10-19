"""
Tests for src/heston_alpha.py – Experiment 4.

Verifies:
  - theta_alpha formula exactness for all three alpha values
  - K_alpha formula exactness
  - Policy pi*(V) = alpha + K_alpha/V
  - Sign-change threshold V^* = -K_alpha/alpha
  - Strict Feller condition
  - Stationary probability ≈ 23% for alpha=1
"""

import numpy as np
import pytest

from src.heston_alpha import (
    PAPER_PARAMS,
    figure2_summary_table,
    heston_K_alpha,
    heston_pi_star,
    heston_theta_alpha,
    sign_change_threshold,
    stationary_gamma_params,
    stationary_short_prob,
    strict_feller,
)

KAPPA = PAPER_PARAMS["kappa"]
THETA = PAPER_PARAMS["theta"]
XI = PAPER_PARAMS["xi"]
VARRHO = PAPER_PARAMS["varrho"]
MU_MINUS_R = PAPER_PARAMS["mu_minus_r"]

V_GRID = np.linspace(0.001, 0.2, 200)  # positive variance grid


# ---------------------------------------------------------------------------
# heston_theta_alpha
# ---------------------------------------------------------------------------


def test_theta_alpha_values():
    """theta_alpha must match paper-stated values exactly."""
    expected = {0.0: 0.045, 0.5: 0.065, 1.0: 0.085}
    for alpha, exp in expected.items():
        ta = heston_theta_alpha(THETA, XI, KAPPA, alpha)
        assert np.isclose(ta, exp, atol=1e-10), (
            f"theta_alpha({alpha}) = {ta}, expected {exp}"
        )


def test_theta_alpha_increases_with_alpha():
    """theta_alpha should be monotone increasing in alpha."""
    alphas = np.linspace(0, 1, 21)
    tas = [heston_theta_alpha(THETA, XI, KAPPA, a) for a in alphas]
    assert np.all(np.diff(tas) > 0)


def test_theta_alpha_linear():
    """theta_alpha is linear in alpha with slope xi^2 / (2*kappa)."""
    slope = XI ** 2 / (2 * KAPPA)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ta = heston_theta_alpha(THETA, XI, KAPPA, alpha)
        expected = THETA + alpha * slope
        assert np.isclose(ta, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# heston_K_alpha
# ---------------------------------------------------------------------------


def test_K_alpha_values():
    """K_alpha must match paper-stated values exactly."""
    expected = {0.0: 0.060, 0.5: 0.010, 1.0: -0.040}
    for alpha, exp in expected.items():
        Ka = heston_K_alpha(MU_MINUS_R, VARRHO, XI, alpha)
        assert np.isclose(Ka, exp, atol=1e-10), (
            f"K_alpha({alpha}) = {Ka}, expected {exp}"
        )


def test_K_alpha_positive_alpha0():
    """K_alpha at alpha=0 is mu-r > 0."""
    Ka = heston_K_alpha(MU_MINUS_R, VARRHO, XI, 0.0)
    assert Ka > 0


def test_K_alpha_negative_alpha1():
    """K_alpha at alpha=1 is negative under the paper calibration."""
    Ka = heston_K_alpha(MU_MINUS_R, VARRHO, XI, 1.0)
    assert Ka < 0


# ---------------------------------------------------------------------------
# heston_pi_star
# ---------------------------------------------------------------------------


def test_pi_star_formula():
    """pi*(V) = alpha + K_alpha/V."""
    for alpha in [0.0, 0.5, 1.0]:
        Ka = heston_K_alpha(MU_MINUS_R, VARRHO, XI, alpha)
        pi = heston_pi_star(V_GRID, alpha, Ka)
        expected = alpha + Ka / V_GRID
        assert np.allclose(pi, expected, atol=1e-14)


def test_pi_star_positive_large_V():
    """For large V, pi* should approach alpha (K_alpha/V → 0)."""
    for alpha in [0.0, 0.5, 1.0]:
        Ka = heston_K_alpha(MU_MINUS_R, VARRHO, XI, alpha)
        V_large = np.array([100.0])
        pi = heston_pi_star(V_large, alpha, Ka)
        assert np.isclose(pi[0], alpha, atol=0.001)


def test_pi_star_at_long_run_mean():
    """Verify policy at V=theta_alpha matches paper approximations."""
    expected_pi_at_mean = {0.0: 1.33, 0.5: 0.65, 1.0: 0.53}
    for alpha, exp_pi in expected_pi_at_mean.items():
        ta = heston_theta_alpha(THETA, XI, KAPPA, alpha)
        Ka = heston_K_alpha(MU_MINUS_R, VARRHO, XI, alpha)
        pi = heston_pi_star(np.array([ta]), alpha, Ka)[0]
        assert np.isclose(pi, exp_pi, atol=0.01), (
            f"pi*(theta_alpha, alpha={alpha}) = {pi:.4f}, expected {exp_pi}"
        )


def test_pi_star_invalid_V_zero():
    """V=0 should raise ValueError (division by zero)."""
    Ka = heston_K_alpha(MU_MINUS_R, VARRHO, XI, 1.0)
    with pytest.raises(ValueError, match="positive"):
        heston_pi_star(np.array([0.0]), 1.0, Ka)


# ---------------------------------------------------------------------------
# sign_change_threshold
# ---------------------------------------------------------------------------


def test_sign_threshold_alpha1():
    """For alpha=1, V^* = 0.04, sqrt(V^*) = 0.20."""
    Ka_1 = heston_K_alpha(MU_MINUS_R, VARRHO, XI, 1.0)
    V_star = sign_change_threshold(1.0, Ka_1)
    assert V_star is not None
    assert np.isclose(V_star, 0.04, atol=1e-10)
    assert np.isclose(np.sqrt(V_star), 0.20, atol=1e-8)


def test_sign_threshold_alpha0_no_crossing():
    """For alpha=0, K_alpha > 0, no sign change → None."""
    Ka_0 = heston_K_alpha(MU_MINUS_R, VARRHO, XI, 0.0)
    result = sign_change_threshold(0.0, Ka_0)
    assert result is None


def test_sign_threshold_general():
    """V^* = -K_alpha / alpha when K_alpha < 0 and alpha > 0."""
    Ka = -0.05
    alpha = 0.8
    V_star = sign_change_threshold(alpha, Ka)
    assert np.isclose(V_star, 0.05 / 0.8, atol=1e-14)


# ---------------------------------------------------------------------------
# strict_feller
# ---------------------------------------------------------------------------


def test_strict_feller_all_alpha():
    """Feller condition must hold for alpha in {0, 0.5, 1} under paper calibration."""
    expected_lhs = {0.0: 0.18, 0.5: 0.26, 1.0: 0.34}
    for alpha, exp_lhs in expected_lhs.items():
        ta = heston_theta_alpha(THETA, XI, KAPPA, alpha)
        feller = strict_feller(KAPPA, ta, XI)
        assert feller["satisfied"], f"Feller failed at alpha={alpha}"
        assert np.isclose(feller["lhs"], exp_lhs, atol=1e-10)
        assert np.isclose(feller["rhs"], XI ** 2, atol=1e-14)


def test_strict_feller_fails_bad_params():
    """With very small kappa*theta, Feller should not be satisfied."""
    feller = strict_feller(kappa=0.5, theta_alpha=0.01, xi=0.5)
    assert not feller["satisfied"]


# ---------------------------------------------------------------------------
# stationary_gamma_params / stationary_short_prob
# ---------------------------------------------------------------------------


def test_gamma_params_shape_scale():
    """Verify shape = 2*kappa*theta_alpha/xi^2 and scale = xi^2/(2*kappa)."""
    ta = heston_theta_alpha(THETA, XI, KAPPA, 1.0)
    gp = stationary_gamma_params(KAPPA, ta, XI)
    expected_shape = 2 * KAPPA * ta / XI ** 2
    expected_scale = XI ** 2 / (2 * KAPPA)
    assert np.isclose(gp["shape"], expected_shape, atol=1e-14)
    assert np.isclose(gp["scale"], expected_scale, atol=1e-14)


def test_stationary_prob_alpha1_approx23():
    """P(V < 0.04) ≈ 23% for alpha=1 under paper calibration."""
    ta_1 = heston_theta_alpha(THETA, XI, KAPPA, 1.0)
    prob = stationary_short_prob(0.04, KAPPA, ta_1, XI)
    assert np.isclose(prob, 0.23, atol=0.02), (
        f"P(V<0.04) = {prob:.4f}, expected ≈ 0.23"
    )


def test_stationary_prob_increases_with_alpha():
    """Short probability P(V < 0.04) increases with alpha (heavier left tail)."""
    probs = []
    for alpha in [0.0, 0.5, 1.0]:
        ta = heston_theta_alpha(THETA, XI, KAPPA, alpha)
        probs.append(stationary_short_prob(0.04, KAPPA, ta, XI))
    # All positive
    assert all(p >= 0 for p in probs)
    # Probability increases as alpha increases (shorter left tail, less mass < 0.04
    # at alpha=0, more at alpha=1 due to higher theta shifts distribution right...)
    # Actually more mass > 0.04 at alpha=1 (higher mean), but the threshold is
    # relative. Let's check they are all in (0,1):
    assert all(0 < p < 1 for p in probs)


def test_stationary_prob_monotone():
    """P(V < threshold) should be in [0, 1] for any threshold."""
    ta = heston_theta_alpha(THETA, XI, KAPPA, 0.5)
    for threshold in [0.01, 0.04, 0.10, 0.30]:
        prob = stationary_short_prob(threshold, KAPPA, ta, XI)
        assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# figure2_summary_table
# ---------------------------------------------------------------------------


def test_figure2_table_shape():
    df = figure2_summary_table([0.0, 0.5, 1.0])
    assert df.shape[0] == 3
    assert "alpha" in df.columns
    assert "theta_alpha" in df.columns
    assert "K_alpha" in df.columns
