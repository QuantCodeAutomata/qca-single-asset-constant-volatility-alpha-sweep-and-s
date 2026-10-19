"""
Experiment 5 – Identification and non-identification of alpha from return laws.

Validates:
  1. Multiple (mu, alpha) pairs with identical mu_eff = mu + alpha*sigma^2
  2. Identical return-law moments for all law-preserving parameterisations
  3. Invariant optimal theta under the law-preserving transformation
  4. Optional: multi-asset extension
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.identification import (
    construct_identification_pairs,
    equivalent_ito_drift_single,
    identification_table,
    identification_table_multi,
    logreturn_moments,
)
from src.multi_asset_alpha import MATRIX_B

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SIGMA = 0.20
M_BAR = 0.08
R = 0.03
T = 1.0
ALPHA_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]


def run_exp5() -> dict:
    """Run all Experiment 5 checks and produce identification table and plots."""

    # ---- 1. Identification pairs ----
    pairs = construct_identification_pairs(M_BAR, SIGMA, ALPHA_LIST)
    print("\n=== Exp 5: Identification pairs (sigma=0.20, m_bar=0.08) ===")
    print(pairs.to_string(index=False))
    # All mu_eff must equal M_BAR
    assert np.allclose(pairs["mu_eff"], M_BAR, atol=1e-14), (
        "Not all mu_eff equal m_bar"
    )

    # ---- 2. Return-law moments must be identical ----
    moments_list = [logreturn_moments(M_BAR, SIGMA, T) for _ in ALPHA_LIST]
    mean_0 = moments_list[0]["mean"]
    var_0 = moments_list[0]["variance"]
    for mom in moments_list[1:]:
        assert np.isclose(mom["mean"], mean_0, atol=1e-14), "Mean log-return differs"
        assert np.isclose(mom["variance"], var_0, atol=1e-14), "Variance differs"
    print(
        f"\nIdentical return law: mean={mean_0:.6f}, var={var_0:.6f} for all alpha in {ALPHA_LIST}"
    )

    # ---- 3. Optimal theta invariant under law-preserving reparametrisation ----
    id_table = identification_table(M_BAR, R, SIGMA, ALPHA_LIST, T)
    theta_invariant = (M_BAR - R) / SIGMA ** 2
    assert np.allclose(
        id_table["theta_star_law_preserving"], theta_invariant, atol=1e-14
    ), "theta_star not invariant"
    print(f"\nInvariant theta^* = {theta_invariant:.6f} (all alpha)")

    print("\n=== Full identification table ===")
    print(id_table.to_string(index=False))

    # ---- 4. Fixed-coefficient comparison ----
    # When mu is fixed (not adjusted), changing alpha DOES change mu_eff and theta
    mu_fixed = M_BAR  # same nominal mu for all alpha
    fixed_thetas = []
    fixed_mu_effs = []
    for alpha in ALPHA_LIST:
        mu_e = equivalent_ito_drift_single(mu_fixed, SIGMA, alpha)
        theta_fixed = (mu_e - R) / SIGMA ** 2
        fixed_thetas.append(theta_fixed)
        fixed_mu_effs.append(mu_e)
    fixed_df = pd.DataFrame(
        {
            "alpha": ALPHA_LIST,
            "mu_fixed": mu_fixed,
            "mu_eff_fixed": fixed_mu_effs,
            "theta_fixed": fixed_thetas,
        }
    )
    print("\n=== Fixed nominal mu: theta changes with alpha ===")
    print(fixed_df.to_string(index=False))

    # ---- 5. Multi-asset extension (optional) ----
    bar_mu2 = np.array([0.10, 0.14])
    multi_table = identification_table_multi(bar_mu2, R, MATRIX_B, ALPHA_LIST)
    print("\n=== Multi-asset identification table ===")
    print(multi_table[["alpha", "mu_eff_matches_bar_mu", "theta_star"]].to_string(index=False))
    assert multi_table["mu_eff_matches_bar_mu"].all(), (
        "Multi-asset mu_eff not equal to bar_mu for some alpha"
    )

    # ---- 6. Plot: identical densities under law-preserving parameterisation ----
    log_ret_grid = np.linspace(-0.8, 0.8, 500)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: all densities overlap (law-preserving)
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ALPHA_LIST)))
    for i, alpha in enumerate(ALPHA_LIST):
        mu_nom = M_BAR - alpha * SIGMA ** 2
        mu_e = equivalent_ito_drift_single(mu_nom, SIGMA, alpha)
        mom = logreturn_moments(mu_e, SIGMA, T)
        pdf = norm.pdf(log_ret_grid, loc=mom["mean"], scale=mom["std"])
        ax.plot(
            log_ret_grid, pdf, color=colors[i], linewidth=2.5 - 0.3 * i,
            label=f"α={alpha} (μ={mu_nom:.4f})",
        )
    ax.set_title(
        "Law-preserving: identical return densities\n(all α share same Itô law)",
        fontsize=10,
    )
    ax.set_xlabel("ln(S_T/S_0)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: fixed nominal mu – distinct densities
    ax = axes[1]
    for i, alpha in enumerate(ALPHA_LIST):
        mu_e = equivalent_ito_drift_single(mu_fixed, SIGMA, alpha)
        mom = logreturn_moments(mu_e, SIGMA, T)
        pdf = norm.pdf(log_ret_grid, loc=mom["mean"], scale=mom["std"])
        ax.plot(
            log_ret_grid, pdf, color=colors[i], linewidth=2.5 - 0.3 * i,
            label=f"α={alpha} (μ_eff={mu_e:.4f})",
        )
    ax.set_title(
        "Fixed nominal μ: distinct return densities\n(changing α changes Itô law)",
        fontsize=10,
    )
    ax.set_xlabel("ln(S_T/S_0)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Experiment 5 – Non-identification of α from Return Data\n"
        f"(σ={SIGMA}, m̄={M_BAR}, r={R}, T={T})",
        fontsize=12,
    )
    fig.tight_layout()
    fig_path = RESULTS_DIR / "figure5_identification.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure 5 saved → {fig_path}")

    # ---- Plot theta comparison ----
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(ALPHA_LIST, [theta_invariant] * len(ALPHA_LIST), "bo-", linewidth=2,
             label="Law-preserving θ* (invariant)")
    ax2.plot(ALPHA_LIST, fixed_thetas, "rs--", linewidth=2,
             label="Fixed nominal μ θ* (varies)")
    ax2.set_xlabel("α (convention parameter)")
    ax2.set_ylabel("θ* (optimal risky weight)")
    ax2.set_title("Optimal Portfolio: Law-preserving vs. Fixed-μ")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig_path2 = RESULTS_DIR / "figure5b_theta_comparison.png"
    fig2.savefig(fig_path2, dpi=150)
    plt.close(fig2)

    # Save tables
    id_table.to_csv(RESULTS_DIR / "exp5_identification_table.csv", index=False)
    fixed_df.to_csv(RESULTS_DIR / "exp5_fixed_mu_table.csv", index=False)

    return {
        "id_table": id_table,
        "fixed_df": fixed_df,
        "theta_invariant": theta_invariant,
        "figure_path": str(fig_path),
        "figure_path2": str(fig_path2),
    }


if __name__ == "__main__":
    run_exp5()
