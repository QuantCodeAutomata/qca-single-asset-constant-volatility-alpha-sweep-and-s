"""
Tests for src/single_asset_alpha.py – Experiment 1.

Verifies:
  - Itô conversion formula
  - Optimal weight formula and Stratonovich shift
  - Value function constant
  - Figure 1 parameter-set exactness
  - Edge cases and input validation
"""

import numpy as np
import pytest

from src.single_asset_alpha import (
    FIGURE1_PARAMS,
    build_figure1_table,
    convert_to_ito_single,
    optimal_controls_single,
    stratonovich_shift,
    sweep_theta_alpha,
    value_constant_single,
)


# ---------------------------------------------------------------------------
# convert_to_ito_single
# ---------------------------------------------------------------------------


def test_ito_conversion_ito_case():
    """alpha=0 should leave mu unchanged."""
    assert np.isclose(convert_to_ito_single(0.08, 0.20, 0.0), 0.08)


def test_ito_conversion_stratonovich():
    """alpha=0.5 should add 0.5*sigma^2."""
    sigma, mu = 0.20, 0.08
    expected = mu + 0.5 * sigma ** 2
    assert np.isclose(convert_to_ito_single(mu, sigma, 0.5), expected)


def test_ito_conversion_full():
    """alpha=1 should add sigma^2."""
    sigma, mu = 0.30, 0.10
    expected = mu + sigma ** 2
    assert np.isclose(convert_to_ito_single(mu, sigma, 1.0), expected)


def test_ito_conversion_invalid_sigma():
    """sigma <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="sigma"):
        convert_to_ito_single(0.05, 0.0, 0.5)


def test_ito_conversion_invalid_alpha():
    """alpha outside [0,1] should raise ValueError."""
    with pytest.raises(ValueError, match="alpha"):
        convert_to_ito_single(0.05, 0.20, 1.5)


# ---------------------------------------------------------------------------
# optimal_controls_single
# ---------------------------------------------------------------------------


def test_theta_ito_benchmark():
    """At alpha=0, theta^* = (mu-r)/sigma^2."""
    mu, r, sigma = 0.10, 0.03, 0.20
    result = optimal_controls_single(mu, r, sigma, 0.0, rho=0.04)
    expected = (mu - r) / sigma ** 2
    assert np.isclose(result["theta"], expected)


def test_theta_stratonovich_shift():
    """theta_{0.5}^* - theta_0^* should equal exactly 0.5."""
    mu, r, sigma, rho = 0.10, 0.03, 0.20, 0.04
    theta_0 = optimal_controls_single(mu, r, sigma, 0.0, rho)["theta"]
    theta_05 = optimal_controls_single(mu, r, sigma, 0.5, rho)["theta"]
    assert np.isclose(theta_05 - theta_0, 0.5, atol=1e-14)


def test_theta_linear_in_alpha():
    """theta_alpha^* = theta_0^* + alpha, slope = 1."""
    mu, r, sigma, rho = 0.12, 0.03, 0.25, 0.05
    alphas = np.linspace(0.0, 1.0, 50)
    thetas = [optimal_controls_single(mu, r, sigma, a, rho)["theta"] for a in alphas]
    slope = np.polyfit(alphas, thetas, 1)[0]
    assert np.isclose(slope, 1.0, atol=1e-10)


def test_consumption_ratio_independent_of_alpha():
    """Consumption ratio rho must not change with alpha."""
    mu, r, sigma, rho = 0.09, 0.02, 0.18, 0.05
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = optimal_controls_single(mu, r, sigma, alpha, rho)
        assert result["consumption_ratio"] == rho


def test_theta_minus_theta0_equals_alpha_exactly():
    """theta(alpha) - theta(0) must equal alpha to machine precision."""
    mu, r, sigma = 0.08, 0.03, 0.15
    alpha_grid = np.linspace(0, 1, 101)
    theta_0 = optimal_controls_single(mu, r, sigma, 0.0)["theta"]
    for alpha in alpha_grid:
        theta_a = optimal_controls_single(mu, r, sigma, alpha)["theta"]
        assert np.isclose(theta_a - theta_0, alpha, atol=1e-14), (
            f"Shift mismatch at alpha={alpha}"
        )


# ---------------------------------------------------------------------------
# Figure 1 parameter sets
# ---------------------------------------------------------------------------


def test_figure1_param_set_theta_ito():
    """Verify that paper-stated theta_ito values are correct to 6 decimal places."""
    expected = {
        "Low-vol defensive": 0.03 / 0.12 ** 2,
        "Broad-equity index": 0.05 / 0.18 ** 2,
        "High-vol growth": 0.08 / 0.35 ** 2,
    }
    for p in FIGURE1_PARAMS:
        assert np.isclose(p["theta_ito"], expected[p["label"]], rtol=1e-9), (
            f"theta_ito mismatch for {p['label']}"
        )


def test_figure1_slope_exactly_one():
    """Each Figure 1 curve must have slope = 1 exactly."""
    alpha_grid = np.linspace(0.0, 1.0, 101)
    table = build_figure1_table(alpha_grid)
    for _, row in table.iterrows():
        assert np.isclose(row["slope"], 1.0, atol=1e-10), (
            f"Slope ≠ 1 for {row['label']}: {row['slope']}"
        )


def test_figure1_stratonovich_shift_exactly_half():
    """The Stratonovich–Itô shift must be exactly 0.5 for all parameter sets."""
    alpha_grid = np.array([0.0, 0.5, 1.0])
    table = build_figure1_table(alpha_grid)
    for _, row in table.iterrows():
        assert np.isclose(row["Strat_shift"], 0.5, atol=1e-14), (
            f"Strat shift ≠ 0.5 for {row['label']}: {row['Strat_shift']}"
        )


def test_figure1_endpoint_values():
    """
    Check approximate endpoint values at alpha=0 and alpha=1.
    Expected: (2.08→3.08), (1.54→2.54), (0.65→1.65).
    """
    expected_ends = [
        (2.0833, 3.0833),
        (1.5432, 2.5432),
        (0.6531, 1.6531),
    ]
    for p, (lo, hi) in zip(FIGURE1_PARAMS, expected_ends):
        theta_0 = p["theta_ito"]
        theta_1 = theta_0 + 1.0
        assert np.isclose(theta_0, lo, atol=0.001), (
            f"{p['label']}: theta(0)={theta_0:.4f}, expected≈{lo}"
        )
        assert np.isclose(theta_1, hi, atol=0.001), (
            f"{p['label']}: theta(1)={theta_1:.4f}, expected≈{hi}"
        )


# ---------------------------------------------------------------------------
# value_constant_single
# ---------------------------------------------------------------------------


def test_value_constant_increases_with_alpha():
    """beta_0 should increase with alpha since mu_eff increases (higher Sharpe^2)."""
    mu, r, sigma, rho = 0.10, 0.03, 0.20, 0.04
    b0 = value_constant_single(mu, r, sigma, 0.0, rho)
    b05 = value_constant_single(mu, r, sigma, 0.5, rho)
    b1 = value_constant_single(mu, r, sigma, 1.0, rho)
    assert b0 < b05 < b1


def test_value_constant_invalid_rho():
    """rho <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="rho"):
        value_constant_single(0.1, 0.03, 0.2, 0.5, rho=-0.01)


# ---------------------------------------------------------------------------
# stratonovich_shift
# ---------------------------------------------------------------------------


def test_stratonovich_shift_is_half():
    """stratonovich_shift should always return 0.5."""
    for sigma in [0.1, 0.2, 0.35]:
        assert stratonovich_shift(sigma) == 0.5


# ---------------------------------------------------------------------------
# sweep_theta_alpha
# ---------------------------------------------------------------------------


def test_sweep_returns_correct_shape():
    alpha_grid = np.linspace(0, 1, 51)
    result = sweep_theta_alpha(0.05, 0.20, alpha_grid)
    assert result.shape == alpha_grid.shape


def test_sweep_monotone_increasing():
    """theta_alpha^* is monotonically increasing in alpha."""
    alpha_grid = np.linspace(0, 1, 101)
    result = sweep_theta_alpha(0.06, 0.25, alpha_grid)
    assert np.all(np.diff(result) > 0)
