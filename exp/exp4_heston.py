"""
Experiment 4 – Heston specialisation: policy curves, sign change, Feller
admissibility, and stationary short-region probability.

Validates:
  1. theta_alpha (long-run mean of converted variance) for alpha in {0, 0.5, 1}
  2. K_alpha (policy coefficient) for alpha in {0, 0.5, 1}
  3. Figure 2 policy curves pi*(V) or pi*(sqrt(V))
  4. Sign-change threshold: V^* = 0.04 ↔ sqrt(V^*) = 0.20 for alpha=1
  5. Strict Feller condition satisfied for all three conventions
  6. Stationary P(V < 0.04) ≈ 23% for alpha=1
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

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALPHA_VALUES = [0.0, 0.5, 1.0]
KAPPA = PAPER_PARAMS["kappa"]
THETA = PAPER_PARAMS["theta"]
XI = PAPER_PARAMS["xi"]
VARRHO = PAPER_PARAMS["varrho"]
MU_MINUS_R = PAPER_PARAMS["mu_minus_r"]

# Volatility grid: approx 5% to 40%
VOL_GRID = np.linspace(0.05, 0.40, 400)
V_GRID = VOL_GRID ** 2


def run_exp4() -> dict:
    """Run all Experiment 4 checks and produce Figure 2 and summary table."""

    # ---- 1. Derived quantities ----
    print("\n=== Exp 4: Heston derived quantities ===")
    expected_ta = {0.0: 0.045, 0.5: 0.065, 1.0: 0.085}
    expected_Ka = {0.0: 0.060, 0.5: 0.010, 1.0: -0.040}
    for alpha in ALPHA_VALUES:
        ta = heston_theta_alpha(THETA, XI, KAPPA, alpha)
        Ka = heston_K_alpha(MU_MINUS_R, VARRHO, XI, alpha)
        print(
            f"  alpha={alpha:.1f}: theta_alpha={ta:.4f} (exp {expected_ta[alpha]:.3f}), "
            f"K_alpha={Ka:.4f} (exp {expected_Ka[alpha]:.3f})"
        )
        assert np.isclose(ta, expected_ta[alpha], atol=1e-10), (
            f"theta_alpha mismatch at alpha={alpha}: {ta} vs {expected_ta[alpha]}"
        )
        assert np.isclose(Ka, expected_Ka[alpha], atol=1e-10), (
            f"K_alpha mismatch at alpha={alpha}: {Ka} vs {expected_Ka[alpha]}"
        )

    # ---- 2. Sign-change threshold for alpha=1 ----
    Ka_1 = heston_K_alpha(MU_MINUS_R, VARRHO, XI, 1.0)
    V_star = sign_change_threshold(1.0, Ka_1)
    assert V_star is not None, "Expected a sign-change threshold for alpha=1"
    assert np.isclose(V_star, 0.04, atol=1e-10), (
        f"Sign-change threshold = {V_star}, expected 0.04"
    )
    assert np.isclose(np.sqrt(V_star), 0.20, atol=1e-8), (
        f"sqrt(V^*) = {np.sqrt(V_star)}, expected 0.20"
    )
    print(f"\n  Sign-change threshold (alpha=1): V^*={V_star}, sqrt(V^*)={np.sqrt(V_star):.3f}")

    # ---- 3. Policy at each long-run mean ----
    expected_pi_at_mean = {0.0: 1.33, 0.5: 0.65, 1.0: 0.53}
    print("\n=== Policy pi* at own long-run mean V=theta_alpha ===")
    for alpha in ALPHA_VALUES:
        ta = heston_theta_alpha(THETA, XI, KAPPA, alpha)
        Ka = heston_K_alpha(MU_MINUS_R, VARRHO, XI, alpha)
        pi_at_mean = heston_pi_star(np.array([ta]), alpha, Ka)[0]
        print(
            f"  alpha={alpha:.1f}: pi*(theta_alpha) = {pi_at_mean:.4f} "
            f"(paper approx {expected_pi_at_mean[alpha]})"
        )
        assert np.isclose(pi_at_mean, expected_pi_at_mean[alpha], atol=0.01), (
            f"pi* at mean mismatch: {pi_at_mean:.4f} vs {expected_pi_at_mean[alpha]}"
        )

    # ---- 4. Strict Feller condition ----
    print("\n=== Strict Feller condition (2*kappa*theta_alpha > xi^2 = 0.16) ===")
    expected_lhs = {0.0: 0.18, 0.5: 0.26, 1.0: 0.34}
    for alpha in ALPHA_VALUES:
        ta = heston_theta_alpha(THETA, XI, KAPPA, alpha)
        feller = strict_feller(KAPPA, ta, XI)
        print(
            f"  alpha={alpha:.1f}: lhs={feller['lhs']:.4f}, rhs={feller['rhs']:.4f}, "
            f"satisfied={feller['satisfied']}"
        )
        assert feller["satisfied"], f"Feller condition FAILED for alpha={alpha}"
        assert np.isclose(feller["lhs"], expected_lhs[alpha], atol=1e-10), (
            f"Feller lhs mismatch at alpha={alpha}"
        )

    # ---- 5. Stationary short-region probability ----
    print("\n=== Stationary P(V < 0.04) ===")
    for alpha in ALPHA_VALUES:
        ta = heston_theta_alpha(THETA, XI, KAPPA, alpha)
        gp = stationary_gamma_params(KAPPA, ta, XI)
        prob = stationary_short_prob(0.04, KAPPA, ta, XI)
        print(
            f"  alpha={alpha:.1f}: shape={gp['shape']:.4f}, scale={gp['scale']:.4f}, "
            f"P(V<0.04)={prob:.4f}"
        )
    # For alpha=1, probability should be ≈ 23%
    ta_1 = heston_theta_alpha(THETA, XI, KAPPA, 1.0)
    prob_1 = stationary_short_prob(0.04, KAPPA, ta_1, XI)
    assert np.isclose(prob_1, 0.23, atol=0.02), (
        f"P(V<0.04) for alpha=1 = {prob_1:.4f}, expected ≈ 0.23"
    )
    print(f"\n  Verified: P(V<0.04) for alpha=1 ≈ {prob_1:.4f} (paper: ~23%)")

    # ---- 6. Figure 2: policy curves vs sqrt(V) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {0.0: "navy", 0.5: "darkorange", 1.0: "darkgreen"}
    for alpha in ALPHA_VALUES:
        Ka = heston_K_alpha(MU_MINUS_R, VARRHO, XI, alpha)
        pi = heston_pi_star(V_GRID, alpha, Ka)
        ta = heston_theta_alpha(THETA, XI, KAPPA, alpha)
        ax.plot(
            VOL_GRID,
            pi,
            color=colors[alpha],
            linewidth=2,
            label=f"α={alpha} (K={Ka:.3f}, θ_α={ta:.3f})",
        )
        # Mark long-run mean
        ax.axvline(
            np.sqrt(ta),
            color=colors[alpha],
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

    # Sign-change threshold (alpha=1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axvline(0.20, color="darkgreen", linestyle=":", linewidth=1.5, label="V*=0.04 (20% vol)")
    ax.set_xlabel("Instantaneous volatility √V_t", fontsize=12)
    ax.set_ylabel("π* (optimal risky fraction)", fontsize=12)
    ax.set_title(
        "Figure 2 – Heston Optimal Risky Fraction vs. Instantaneous Volatility\n"
        f"(κ={KAPPA}, θ={THETA}, ξ={XI}, ρ̄={VARRHO}, μ−r={MU_MINUS_R})",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.04, 0.41)
    fig.tight_layout()
    fig_path = RESULTS_DIR / "figure2_heston_policy_curves.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure 2 saved → {fig_path}")

    # ---- 7. Full summary table ----
    summary_df = figure2_summary_table(ALPHA_VALUES)
    print("\n=== Figure 2 summary table ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(RESULTS_DIR / "exp4_heston_summary.csv", index=False)

    return {
        "summary_df": summary_df,
        "figure_path": str(fig_path),
        "prob_alpha1": prob_1,
    }


if __name__ == "__main__":
    run_exp4()
