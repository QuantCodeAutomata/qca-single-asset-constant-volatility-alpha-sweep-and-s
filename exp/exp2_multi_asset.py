"""
Experiment 2 – Multi-asset constant-volatility comparative statics and
law-preserving invariance (Proposition 4.2).

Validates:
  1. Itô conversion: mu_Ito = mu + alpha * d_V
  2. Optimal theta: V^{-1}(mu-r1) + alpha * V^{-1} d_V
  3. Alpha sensitivity: d theta/d alpha = V^{-1} d_V
  4. Remark 4.2 matrices A, B, C
  5. Proposition 4.2 law-preserving invariance
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Experiment parameters
R = 0.03
RHO = 0.04
ALPHA_GRID = np.linspace(0.0, 1.0, 101)


def run_exp2() -> dict:
    """Run all Experiment 2 checks and produce comparison tables and plots."""

    # ---- Part A: Fixed-coefficient 2-asset example ----
    V2 = MATRIX_B  # 2×2 covariance matrix from Remark 4.2
    mu2 = np.array([0.08, 0.12])  # illustrative nominal drifts

    print("\n=== Part A: Fixed-coefficient multi-asset example (Matrix B) ===")
    for alpha in [0.0, 0.5, 1.0]:
        mu_ito = ito_drift_multi(mu2, V2, alpha)
        theta = optimal_theta_multi(mu2, R, V2, alpha)
        print(f"  alpha={alpha:.1f}: mu_Ito={mu_ito}, theta={theta}")
        # Verify linearity in alpha
        d_V = variance_vector(V2)
        sensitivity = alpha_sensitivity_multi(V2)
        expected_theta = optimal_theta_multi(mu2, R, V2, 0.0) + alpha * sensitivity
        assert np.allclose(theta, expected_theta, atol=1e-12), (
            f"Linearity check failed at alpha={alpha}"
        )

    # ---- Part B: Remark 4.2 matrices ----
    print("\n=== Part B: Remark 4.2 sensitivity table ===")
    remark_df = remark42_sensitivity_table()
    print(remark_df.to_string(index=False))

    # Specific Remark 4.2 verifications
    # Matrix A: V^{-1} d_V should be all-ones
    sens_A = alpha_sensitivity_multi(MATRIX_A)
    assert np.allclose(sens_A, np.ones(4), atol=1e-10), (
        f"Matrix A: V^{{-1}} d_V = {sens_A}, expected ones"
    )
    print(f"\nMatrix A: V^{{-1}} d_V = {sens_A} (expected all-1s)")

    # Matrix B: V^{-1} d_V ≈ (-1.143, 1.429)
    sens_B = alpha_sensitivity_multi(MATRIX_B)
    expected_B = np.array([-2.0 / 1.75, 2.5 / 1.75])
    assert np.allclose(sens_B, expected_B, atol=1e-6), (
        f"Matrix B: V^{{-1}} d_V = {sens_B}, expected {expected_B}"
    )
    print(f"Matrix B: V^{{-1}} d_V = {sens_B} (expected ≈ [-1.143, 1.429])")

    # Matrix C: V^{-1} 1 ≈ (0.803, 0.619, 0.734) — paper says d_V=1 (unit vector)
    sens_C = alpha_sensitivity_multi(MATRIX_C)
    # Paper reports (175,135,160)/218; verify within 1%
    expected_C_approx = np.array([175.0, 135.0, 160.0]) / 218.0
    assert np.allclose(sens_C, expected_C_approx, atol=0.01), (
        f"Matrix C: V^{{-1}} d_V = {sens_C}, expected ≈ {expected_C_approx}"
    )
    print(f"Matrix C: V^{{-1}} d_V = {sens_C} (expected ≈ [0.803, 0.619, 0.734])")

    # ---- Part C: Law-preserving invariance ----
    print("\n=== Part C: Law-preserving invariance (Proposition 4.2) ===")
    bar_mu = np.array([0.10, 0.14])
    result = verify_law_preserving_invariance(bar_mu, R, MATRIX_B, ALPHA_GRID)
    print(f"  theta_reference = {result['theta_reference']}")
    print(f"  max_deviation   = {result['max_deviation']:.2e}")
    print(f"  is_invariant    = {result['is_invariant']}")
    assert result["is_invariant"], (
        f"Law-preserving invariance FAILED; max_deviation={result['max_deviation']}"
    )

    # ---- Plot: alpha sensitivity bar chart ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (label, V) in zip(
        axes,
        [("A (4×4)", MATRIX_A), ("B (2×2)", MATRIX_B), ("C (3×3)", MATRIX_C)],
    ):
        sens = alpha_sensitivity_multi(V)
        ax.bar(range(len(sens)), sens, color="steelblue", edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Matrix {label}\nV⁻¹ d_V", fontsize=10)
        ax.set_xlabel("Asset index")
        ax.set_ylabel("Sensitivity")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Figure 2a – Remark 4.2: Alpha Sensitivity per Asset (V⁻¹ d_V)",
        fontsize=12,
    )
    fig.tight_layout()
    fig_path = RESULTS_DIR / "figure2a_remark42_sensitivity.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nRemark 4.2 sensitivity plot saved → {fig_path}")

    # ---- Plot: fixed vs law-preserving theta over alpha ----
    mu2_fixed = np.array([0.08, 0.12])
    fixed_thetas = np.array(
        [optimal_theta_multi(mu2_fixed, R, MATRIX_B, a) for a in ALPHA_GRID]
    )
    d_V_B = variance_vector(MATRIX_B)
    law_thetas = np.array(
        [
            optimal_theta_multi(law_preserving_mu(bar_mu, d_V_B, a), R, MATRIX_B, a)
            for a in ALPHA_GRID
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ALPHA_GRID, fixed_thetas[:, 0], "-b", linewidth=2, label="Fixed μ – Asset 1")
    ax.plot(ALPHA_GRID, fixed_thetas[:, 1], "-r", linewidth=2, label="Fixed μ – Asset 2")
    ax.plot(
        ALPHA_GRID,
        law_thetas[:, 0],
        "--b",
        linewidth=2,
        label="Law-preserving – Asset 1",
    )
    ax.plot(
        ALPHA_GRID,
        law_thetas[:, 1],
        "--r",
        linewidth=2,
        label="Law-preserving – Asset 2",
    )
    ax.set_xlabel("α (convention parameter)", fontsize=12)
    ax.set_ylabel("θ* (optimal weight)", fontsize=12)
    ax.set_title(
        "Multi-asset: Fixed-coefficient vs. Law-preserving theta\n(Matrix B, 2 assets)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path2 = RESULTS_DIR / "figure2b_law_preserving_invariance.png"
    fig.savefig(fig_path2, dpi=150)
    plt.close(fig)
    print(f"Law-preserving invariance plot saved → {fig_path2}")

    # Save tables
    remark_df.to_csv(RESULTS_DIR / "exp2_remark42_table.csv", index=False)

    return {
        "remark_df": remark_df,
        "invariance_result": result,
        "figure_path": str(fig_path),
        "figure_path2": str(fig_path2),
    }


if __name__ == "__main__":
    run_exp2()
