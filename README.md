# Alpha-Sweep & Stochastic-Convention Effects in Merton Portfolio Theory

A deterministic, formula-driven replication of the stochastic integral convention
(Itô vs. Stratonovich vs. general-α) effects on optimal log-utility Merton
portfolios, covering single-asset, multi-asset, factor-driven, and Heston
specialisations, together with an identification analysis showing that return
data alone cannot pin down α without additional structural restrictions.

## Background

Many mathematical finance texts write the wealth SDE using a specific stochastic
integral convention without making the choice explicit. When the **nominal drift μ**
is treated as a fixed primitive rather than an artefact of the convention, switching
from Itô (α = 0) to Stratonovich (α = ½) to the anti-Itô endpoint (α = 1) changes
the equivalent Itô drift and therefore shifts the Merton optimal risky allocation.

The central formula in the one-risky-asset, constant-volatility case is:

```
θ*_α = (μ − r) / σ² + α           (Experiment 1)
```

This repository replicates five inter-related experiments that verify this
formula and its multi-asset, factor-driven, and Heston generalisations.

---

## Repository Structure

```
qca-single-asset-constant-volatility-alpha-sweep-and-s/
│
├── data/               # Placeholder – no empirical data used
│
├── src/                # Core formula implementations
│   ├── single_asset_alpha.py      # Exp 1: one-asset constant-vol
│   ├── multi_asset_alpha.py       # Exp 2: multi-asset constant-vol
│   ├── factor_driven_alpha.py     # Exp 3: general factor model
│   ├── heston_alpha.py            # Exp 4: Heston specialisation
│   └── identification.py         # Exp 5: identification/non-identification
│
├── exp/                # Experiment runner scripts
│   ├── exp1_single_asset.py
│   ├── exp2_multi_asset.py
│   ├── exp3_factor_driven.py
│   ├── exp4_heston.py
│   └── exp5_identification.py
│
├── tests/              # pytest test suite (91 tests)
│   ├── test_single_asset.py
│   ├── test_multi_asset.py
│   ├── test_factor_driven.py
│   ├── test_heston.py
│   └── test_identification.py
│
├── results/            # Generated figures, tables, and RESULTS.md
│   ├── RESULTS.md
│   ├── figure1_single_asset_alpha_sweep.png
│   ├── figure2_heston_policy_curves.png
│   ├── figure2a_remark42_sensitivity.png
│   ├── figure2b_law_preserving_invariance.png
│   ├── figure3_factor_decomposition.png
│   └── *.csv
│
├── README.md
└── .gitignore
```

---

## Experiments

### Experiment 1 – Single-Asset Alpha Sweep & Stratonovich Shift
- Converts α-interpreted dynamics to Itô form: `μ_eff = μ + α σ²`
- Optimal weight: `θ*_α = (μ−r)/σ² + α` (slope = 1 in α)
- Stratonovich shift: `θ*_½ − θ*_0 = 0.5` exactly
- Replicates Figure 1 for three benchmark parameter sets

### Experiment 2 – Multi-Asset Comparative Statics & Law-Preserving Invariance
- Multi-asset conversion: `μ_Ito = μ + α d_V`
- Optimal allocation: `θ*_α = V⁻¹(μ−r1) + α V⁻¹ d_V`
- Remark 4.2 covariance examples (Matrices A, B, C)
- Proposition 4.2: law-preserving invariance across α

### Experiment 3 – General Factor-Driven Model
- Two-channel decomposition: own-variance channel (α) + covariation channel
  (`α ρ̄ σ'(x) ν(x) / σ(x)²`)
- Special-case eliminations (ρ̄ = 0, σ' = 0, ν = 0)
- Optional ODE / Feynman–Kac verification for value function

### Experiment 4 – Heston Specialisation
- Long-run variance shift: `θ_α = θ + α ξ² / (2κ)`
- Policy: `π*_t = α + K_α / V_t`,  `K_α = (μ−r) + α ρ̄ ξ / 2`
- Sign-change threshold at 20% vol for α = 1 (negative correlation regime)
- Strict Feller admissibility check
- Stationary P(V < 0.04) ≈ 23% for α = 1

### Experiment 5 – Identification & Non-Identification of α
- Shows that (μ, α) pairs with equal `μ + α σ²` generate identical return laws
- Under law-preserving transformation, optimal θ* is invariant across α
- Confirms α is not separately identifiable from return data without structural
  restrictions on the drift

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scipy matplotlib

# Run all experiments
python exp/exp1_single_asset.py
python exp/exp2_multi_asset.py
python exp/exp3_factor_driven.py
python exp/exp4_heston.py
python exp/exp5_identification.py

# Run the full test suite
python -m pytest tests/ -v
```

---

## Key Results Summary

| Experiment | Key numerical result |
|------------|----------------------|
| Exp 1 | θ*(α) − θ*(0) = α exactly; Stratonovich shift = 0.5 |
| Exp 2 | V⁻¹d_V ≈ [1,1,1,1] (A); [−1.14, 1.43] (B); [0.80, 0.62, 0.73] (C) |
| Exp 3 | covariation channel vanishes when ρ̄=0, σ'=0, or ν=0 |
| Exp 4 | θ_1 = 0.085; K_1 = −0.04; P(V<0.04|α=1) ≈ 23.1% |
| Exp 5 | All (μ,α) pairs with same μ_eff give identical return law and θ* |

See `results/RESULTS.md` for detailed tables and figures.

---

## No Empirical Data

All experiments are entirely deterministic. No market data, securities, or dates
are used. The `MASSIVE_TOKEN` environment variable is not required.

---

## Testing

```
91 passed in 2.18s
```

All tests cover: input validation, mathematical properties, exact numeric
benchmarks from the paper, edge cases, and regression checks.
