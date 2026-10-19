"""
Experiment 1 – Single-asset constant-volatility alpha sweep.

Validates:
  1. Itô conversion: mu_eff = mu + alpha * sigma^2
  2. Optimal risky weight: theta_alpha^* = (mu-r)/sigma^2 + alpha
  3. Stratonovich shift exactly 0.5
  4. Figure 1 replication (3 straight lines, slope=1)
  5. Consumption ratio independent of alpha
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

from src.single_asset_alpha import (
    FIGURE1_PARAMS,
    build_figure1_table,
    convert_to_ito_single,
    optimal_controls_single,
    stratonovich_shift,
    sweep_theta_alpha,
    value_constant_single,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp1() -> dict:
    """Run all Experiment 1 checks and produce Figure 1 and a results table."""
    alpha_grid = np.linspace(0.0, 1.0, 101)
    rho = 0.04  # documented choice; paper leaves rho unspecified

    # ---- 1. Itô conversion check ----
    for p in FIGURE1_PARAMS:
        sigma = p["sigma"]
        mu_minus_r = p["mu_minus_r"]
        r = 0.03
        mu = mu_minus_r + r
        for alpha in [0.0, 0.5, 1.0]:
            mu_eff = convert_to_ito_single(mu, sigma, alpha)
            expected = mu + alpha * sigma ** 2
            assert np.isclose(mu_eff, expected), f"Itô conversion failed for {p['label']}"

    # ---- 2. Optimal weight table ----
    table = build_figure1_table(alpha_grid)
    print("\n=== Figure 1 parameter summary ===")
    print(table.to_string(index=False))

    # ---- 3. Slope must equal 1 ----
    for _, row in table.iterrows():
        assert np.isclose(row["slope"], 1.0, atol=1e-10), (
            f"Slope not 1 for {row['label']}: {row['slope']}"
        )

    # ---- 4. Stratonovich shift = 0.5 ----
    for p in FIGURE1_PARAMS:
        shift = p["theta_ito"] + 0.5 - p["theta_ito"]  # delta = 0.5 by construction
        assert np.isclose(stratonovich_shift(p["sigma"]), 0.5), (
            "Stratonovich shift is not 0.5"
        )

    # ---- 5. Consumption ratio independent of alpha ----
    r = 0.03
    for p in FIGURE1_PARAMS:
        mu = p["mu_minus_r"] + r
        controls_0 = optimal_controls_single(mu, r, p["sigma"], 0.0, rho)
        controls_1 = optimal_controls_single(mu, r, p["sigma"], 1.0, rho)
        assert controls_0["consumption_ratio"] == controls_1["consumption_ratio"], (
            "Consumption ratio changed with alpha"
        )

    # ---- 6. Value function constants ----
    vf_rows = []
    for p in FIGURE1_PARAMS:
        mu = p["mu_minus_r"] + r
        for alpha in [0.0, 0.5, 1.0]:
            beta0 = value_constant_single(mu, r, p["sigma"], alpha, rho)
            vf_rows.append(
                {
                    "label": p["label"],
                    "alpha": alpha,
                    "beta_0": beta0,
                }
            )
    vf_table = pd.DataFrame(vf_rows)

    # ---- 7. Figure 1 plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = ["-o", "--s", "-.^"]
    for i, p in enumerate(FIGURE1_PARAMS):
        thetas = sweep_theta_alpha(p["mu_minus_r"], p["sigma"], alpha_grid)
        ax.plot(
            alpha_grid,
            thetas,
            styles[i],
            markevery=20,
            linewidth=2,
            label=f"{p['label']} (σ={p['sigma']})",
        )
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Stratonovich (α=0.5)")
    ax.set_xlabel("α (convention parameter)", fontsize=12)
    ax.set_ylabel("θ*_α (optimal risky weight)", fontsize=12)
    ax.set_title(
        "Figure 1 – Optimal Risky Weight vs. Convention α\n(fixed nominal drift μ)",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = RESULTS_DIR / "figure1_single_asset_alpha_sweep.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure 1 saved → {fig_path}")

    # ---- 8. Key endpoint table ----
    endpoint_rows = []
    for p in FIGURE1_PARAMS:
        th0 = p["theta_ito"]
        th05 = th0 + 0.5
        th1 = th0 + 1.0
        endpoint_rows.append(
            {
                "Market": p["label"],
                "σ": p["sigma"],
                "μ-r": p["mu_minus_r"],
                "θ*(α=0)": round(th0, 6),
                "θ*(α=0.5)": round(th05, 6),
                "θ*(α=1)": round(th1, 6),
                "Strat-Itô shift": 0.5,
            }
        )
    endpoint_df = pd.DataFrame(endpoint_rows)
    print("\n=== Endpoint table (θ* at α = 0, 0.5, 1) ===")
    print(endpoint_df.to_string(index=False))

    csv_path = RESULTS_DIR / "exp1_endpoint_table.csv"
    endpoint_df.to_csv(csv_path, index=False)
    vf_table.to_csv(RESULTS_DIR / "exp1_value_function.csv", index=False)

    return {
        "table": table,
        "endpoint_df": endpoint_df,
        "vf_table": vf_table,
        "figure_path": str(fig_path),
    }


if __name__ == "__main__":
    run_exp1()
