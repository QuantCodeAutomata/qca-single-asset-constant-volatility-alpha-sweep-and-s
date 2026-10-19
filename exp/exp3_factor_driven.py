"""
Experiment 3 – General factor-driven model: alpha correction decomposition.

Validates:
  1. Itô conversion of risky asset: mu_eff = mu + alpha*(sigma^2 + varrho*sigma'*nu)
  2. Itô conversion of factor: b_alpha = b + alpha*nu*nu'
  3. Optimal pi* = (mu_eff - r)/sigma^2
  4. Own-variance and covariation channels in pi*
  5. Three special-case eliminations of the covariation channel
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

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# State grid and parameters
X_GRID = np.linspace(0.05, 0.50, 200)  # positive state space
R = 0.03
VARRHO = 0.6   # non-zero correlation to see covariation channel
ALPHA_VALUES = [0.0, 0.5, 1.0]


def run_exp3() -> dict:
    """Run all Experiment 3 checks and produce decomposition plots."""

    # ---- 1. Verify mu_eff decomposition ----
    print("\n=== Exp 3: Effective drift decomposition ===")
    for alpha in ALPHA_VALUES:
        decomp = alpha_correction_decomposition(
            X_GRID, default_sigma, default_dsigma, default_nu, alpha, VARRHO
        )
        mu_e = mu_eff_factor(
            X_GRID, default_mu, default_sigma, default_dsigma, default_nu, alpha, VARRHO
        )
        expected_correction = decomp["total_correction"]
        actual_correction = mu_e - default_mu(X_GRID)
        assert np.allclose(actual_correction, expected_correction, atol=1e-14), (
            f"Decomposition mismatch at alpha={alpha}"
        )
        print(
            f"  alpha={alpha:.1f}: max |correction| = "
            f"{np.max(np.abs(expected_correction)):.4f}, "
            f"max |own_var| = {np.max(np.abs(decomp['own_variance'])):.4f}, "
            f"max |covar| = {np.max(np.abs(decomp['covariation'])):.4f}"
        )

    # ---- 2. Verify factor drift conversion ----
    for alpha in ALPHA_VALUES:
        b_a = b_alpha_factor(X_GRID, default_b, default_nu, default_dnu, alpha)
        expected = default_b(X_GRID) + alpha * default_nu(X_GRID) * default_dnu(X_GRID)
        assert np.allclose(b_a, expected, atol=1e-14), (
            f"Factor drift conversion mismatch at alpha={alpha}"
        )

    # ---- 3. Optimal pi* ----
    print("\n=== Optimal pi* at centre of grid (x=0.275) ===")
    x_test = np.array([0.275])
    for alpha in ALPHA_VALUES:
        pi = pi_star_factor(
            x_test, R, default_mu, default_sigma, default_dsigma, default_nu, alpha, VARRHO
        )
        print(f"  alpha={alpha:.1f}: pi* = {pi[0]:.4f}")

    # ---- 4. Special-case checks ----
    print("\n=== Special-case covariation elimination ===")
    zero_fn = lambda x: np.zeros_like(x, dtype=np.float64)
    cases_df = special_case_checks(
        X_GRID, R, default_mu, default_sigma, default_dsigma, default_nu, alpha=0.7
    )
    print(cases_df.to_string(index=False))

    # All three cases: covariation must be zero (or near-machine-eps)
    for _, row in cases_df.iterrows():
        assert row["max_covariation"] < 1e-14, (
            f"Covariation not zero in {row['case']}: {row['max_covariation']}"
        )
    # Own-variance must remain non-zero
    for _, row in cases_df.iterrows():
        assert row["max_own_variance"] > 0.0, (
            f"Own-variance vanished unexpectedly in {row['case']}"
        )

    # ---- 5. Decomposition plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Plot 1: mu_eff(x) - mu(x) for different alpha
    ax = axes[0, 0]
    for alpha in ALPHA_VALUES:
        mu_e = mu_eff_factor(
            X_GRID, default_mu, default_sigma, default_dsigma, default_nu, alpha, VARRHO
        )
        correction = mu_e - default_mu(X_GRID)
        ax.plot(X_GRID, correction, label=f"α={alpha}")
    ax.set_title("Total correction μ_eff(x) − μ(x)")
    ax.set_xlabel("State x")
    ax.set_ylabel("Drift correction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Own-variance channel
    ax = axes[0, 1]
    for alpha in ALPHA_VALUES:
        decomp = alpha_correction_decomposition(
            X_GRID, default_sigma, default_dsigma, default_nu, alpha, VARRHO
        )
        ax.plot(X_GRID, decomp["own_variance"], label=f"α={alpha}")
    ax.set_title("Own-variance channel α·σ(x)²")
    ax.set_xlabel("State x")
    ax.set_ylabel("α·σ(x)²")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Covariation channel
    ax = axes[1, 0]
    for alpha in ALPHA_VALUES:
        decomp = alpha_correction_decomposition(
            X_GRID, default_sigma, default_dsigma, default_nu, alpha, VARRHO
        )
        ax.plot(X_GRID, decomp["covariation"], label=f"α={alpha}")
    ax.set_title(f"Covariation channel α·ρ̄·σ'(x)·ν(x)  (ρ̄={VARRHO})")
    ax.set_xlabel("State x")
    ax.set_ylabel("Covariation correction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Optimal pi*(x) for different alpha
    ax = axes[1, 1]
    for alpha in ALPHA_VALUES:
        pi = pi_star_factor(
            X_GRID, R, default_mu, default_sigma, default_dsigma, default_nu, alpha, VARRHO
        )
        ax.plot(X_GRID, pi, label=f"α={alpha}")
    ax.set_title("Optimal risky fraction π*(x)")
    ax.set_xlabel("State x")
    ax.set_ylabel("π*")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Experiment 3 – Factor model alpha correction decomposition\n"
        f"(σ=0.15+0.2x, ν=0.10+0.05x, μ=0.08+0.1x, b=mean-reverting, ρ̄={VARRHO})",
        fontsize=11,
    )
    fig.tight_layout()
    fig_path = RESULTS_DIR / "figure3_factor_decomposition.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure 3 saved → {fig_path}")

    # b_alpha plot
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for alpha in ALPHA_VALUES:
        b_a = b_alpha_factor(X_GRID, default_b, default_nu, default_dnu, alpha)
        ax2.plot(X_GRID, b_a, label=f"α={alpha}")
    ax2.set_title("Converted factor drift b_α(x) = b(x) + α·ν(x)·ν'(x)")
    ax2.set_xlabel("State x")
    ax2.set_ylabel("b_α(x)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig_path2 = RESULTS_DIR / "figure3b_factor_drift.png"
    fig2.savefig(fig_path2, dpi=150)
    plt.close(fig2)

    # Save special-case table
    cases_df.to_csv(RESULTS_DIR / "exp3_special_cases.csv", index=False)

    return {
        "cases_df": cases_df,
        "figure_path": str(fig_path),
        "figure_path2": str(fig_path2),
    }


if __name__ == "__main__":
    run_exp3()
