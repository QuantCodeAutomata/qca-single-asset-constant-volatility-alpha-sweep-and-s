# Experiment Results

**Project:** Alpha-Sweep & Stochastic-Convention Effects in Merton Portfolio Theory  
**Date generated:** 2026-10-19  
**Tests:** 91 / 91 passed  

---

## Overview

These experiments replicate the core deterministic results of the paper's
alpha-parameterised stochastic integral convention study. All inputs are
closed-form formulas; no market data are used.

The central mechanism: when the nominal drift coefficient μ is held fixed while
the stochastic integral convention changes from Itô (α = 0) to Stratonovich
(α = ½) to the anti-Itô limit (α = 1), the equivalent Itô drift shifts as
**μ_eff = μ + α σ²**, and the log-utility Merton optimal risky weight shifts
one-for-one with α.

---

## Experiment 1 – Single-Asset Alpha Sweep & Stratonovich Shift

### Methodology
- One risky asset with constant volatility σ
- α-interpreted dynamics converted to Itô form: `μ_eff = μ + α σ²`
- Log-utility Merton solution: `θ*_α = (μ_eff − r) / σ² = (μ−r)/σ² + α`
- Consumption policy: `c*_t = ρ a_t` (unchanged by α)

### Figure 1 – Parameter Sets

| Market | σ | μ−r | θ*(α=0) | θ*(α=0.5) | θ*(α=1) | Strat−Itô shift |
|--------|------|------|---------|-----------|---------|-----------------|
| Low-vol defensive | 0.12 | 0.03 | 2.0833 | 2.5833 | 3.0833 | 0.5 |
| Broad-equity index | 0.18 | 0.05 | 1.5432 | 2.0432 | 2.5432 | 0.5 |
| High-vol growth | 0.35 | 0.08 | 0.6531 | 1.1531 | 1.6531 | 0.5 |

**Observations:**
- Each curve in Figure 1 is a straight line with slope = 1 in α
- The Stratonovich–Itô shift is exactly +0.5 across all parameter sets
- The optimal consumption ratio ρ is strictly invariant in α
- Endpoint values match the paper: approximately (2.08→3.08), (1.54→2.54), (0.65→1.65)

### Generated Files
- `figure1_single_asset_alpha_sweep.png` – Figure 1 replication
- `exp1_endpoint_table.csv` – θ* at α ∈ {0, 0.5, 1} for all three markets
- `exp1_value_function.csv` – β₀ values under chosen ρ = 0.04

---

## Experiment 2 – Multi-Asset Comparative Statics & Law-Preserving Invariance

### Methodology
- n risky assets; covariance matrix V = Σ Σᵀ, variance vector d_V = diag(V)
- Equivalent Itô drift: `μ_Ito = μ + α d_V`
- Optimal allocation: `θ*_α = V⁻¹(μ−r1) + α V⁻¹ d_V`
- Sensitivity: `∂θ*/∂α = V⁻¹ d_V`

### Remark 4.2 – Covariance Sensitivity Examples

| Matrix | d_V | V⁻¹ d_V (approx) | Interpretation |
|--------|-----|-------------------|----------------|
| A (4×4) | [4,4,4,4] | [1,1,1,1] | Uniform +α per asset despite off-diagonal elements |
| B (2×2) | [1,4] | [−1.143, 1.429] | **Negative first component** – increasing α lowers asset-1 weight |
| C (3×3) | [1,1,1] | [0.803, 0.619, 0.734] | Non-uniform positive corrections even with equal variances |

### Proposition 4.2 – Law-Preserving Invariance

Under the transformation `μ(α) = μ̄ − α d_V`, the equivalent Itô drift
is `μ(α) + α d_V = μ̄` for all α. The optimal allocation reduces to
`θ* = V⁻¹(μ̄ − r1)`, which is constant across α.

- Maximum numerical deviation across 101 α values: **3.89 × 10⁻¹⁶** (machine precision)

### Generated Files
- `figure2a_remark42_sensitivity.png` – Bar chart of V⁻¹ d_V for Matrices A, B, C
- `figure2b_law_preserving_invariance.png` – θ* invariance demonstration
- `exp2_remark42_table.csv` – Sensitivity table

---

## Experiment 3 – General Factor-Driven Model

### Methodology
- One risky asset, one state factor X_t with correlated Brownian motions
- Shock correlation: d⟨W^S, W^X⟩_t = ϱ dt
- Effective Itô drift: `μ_eff(x) = μ(x) + α[σ(x)² + ϱ σ'(x) ν(x)]`
- Optimal risky fraction:
  ```
  π*_t = [μ_eff(X_t) − r] / σ(X_t)²
       = [μ(X_t)−r]/σ(X_t)² + α  +  α ϱ σ'(x)ν(x)/σ(x)²
       = baseline  +  own-variance channel  +  covariation channel
  ```

### Test Functions Used (implementer choice; not paper-imposed)
- `σ(x) = 0.15 + 0.1x`,  `σ'(x) = 0.1`
- `ν(x) = 0.05 + 0.02x`,  `ν'(x) = 0.02`  
- `b(x) = 2(0.25 − x)`,  `μ(x) = 0.10 + 0.05x`,  `ϱ = −0.5`
- State grid: x ∈ [0, 0.55]

### Special-Case Elimination Table (α = 0.5)

| Case | max|covariation channel| | max|own-variance channel| |
|------|-------------------------------|--------------------------------|
| (i) ϱ = 0 | **0.0000** | 0.04375 |
| (ii) σ' = 0 (constant σ) | **0.0000** | 0.04375 |
| (iii) ν = 0 | **0.0000** | 0.04375 |
| Full model (ϱ = −0.5) | 0.0075 | 0.0312 |

The covariation channel vanishes in all three special cases while the
own-variance term α remains intact.

### Generated Files
- `figure3_factor_decomposition.png` – Decomposition of π*(x) into channels
- `figure3b_factor_drift.png` – μ_eff(x) decomposition
- `exp3_special_cases.csv` – Channel magnitudes for special cases

---

## Experiment 4 – Heston Specialisation

### Calibration (from the paper)
κ = 2, θ = 0.045, ξ = 0.4, ϱ = −0.5, μ−r = 0.06

### Derived Quantities

| α | θ_α | √θ_α | K_α | π*(θ_α) | Feller 2κθ_α > ξ²? |
|---|-----|------|------|---------|----------------------|
| 0.0 | 0.0450 | 0.2121 | +0.0600 | 1.3333 | ✅ (0.18 > 0.16) |
| 0.5 | 0.0650 | 0.2550 | +0.0100 | 0.6538 | ✅ (0.26 > 0.16) |
| 1.0 | 0.0850 | 0.2915 | −0.0400 | 0.5294 | ✅ (0.34 > 0.16) |

### Key Results

**Policy formula:** `π*_t = α + K_α / V_t`

**Sign-change threshold (α = 1, negative correlation):**
- V* = −K₁/α = 0.04/1 = **0.04**, i.e. **20% instantaneous volatility**
- Below 20% vol, the α = 1 strategy goes **short** the risky asset

**Stationary short-region probability P(V < 0.04):**

| α | Gamma shape a | Gamma scale b | P(V < 0.04) |
|---|---------------|---------------|--------------|
| 0.0 | 1.125 | 0.04 | 57.9% |
| 0.5 | 1.625 | 0.04 | 38.2% |
| 1.0 | 2.125 | 0.04 | **23.1%** |

The α = 1 stationary short-region probability ≈ **23.1%** reproduces the
paper's reported value of roughly 23%.

### Generated Files
- `figure2_heston_policy_curves.png` – π*(√V) for α ∈ {0, 0.5, 1}
- `exp4_heston_summary.csv` – All derived quantities

---

## Experiment 5 – Identification & Non-Identification of α

### Setup
- σ = 0.20, target μ_eff = m̄ = 0.08, horizon T = 1
- Five (μ, α) pairs all sharing the same μ_eff

### Identification Table (law-preserving parameterisation)

| α | Nominal μ | μ_eff | Log-return mean | Log-return var | θ* (invariant) |
|---|-----------|-------|-----------------|----------------|-----------------|
| 0.00 | 0.0800 | 0.08 | 0.06 | 0.04 | **1.25** |
| 0.25 | 0.0700 | 0.08 | 0.06 | 0.04 | **1.25** |
| 0.50 | 0.0600 | 0.08 | 0.06 | 0.04 | **1.25** |
| 0.75 | 0.0500 | 0.08 | 0.06 | 0.04 | **1.25** |
| 1.00 | 0.0400 | 0.08 | 0.06 | 0.04 | **1.25** |

All five parameterisations produce **exactly the same return law** (Normal
log-returns with mean 0.06, variance 0.04 per year) and the same optimal θ* = 1.25.

### Fixed Nominal μ Comparison

| α | μ_fixed | μ_eff | θ*(fixed μ) |
|---|---------|-------|-------------|
| 0.00 | 0.08 | 0.08 | 1.25 |
| 0.25 | 0.08 | 0.09 | 1.50 |
| 0.50 | 0.08 | 0.10 | 1.75 |
| 0.75 | 0.08 | 0.11 | 2.00 |
| 1.00 | 0.08 | 0.12 | 2.25 |

Under fixed μ, θ* changes because the **return law changes**. The portfolio
shifts arise from a changed Itô drift, not from α being independently identified.

### Conclusion

> **α is not separately identified from return data alone when μ is unrestricted.**
> Observed differences in optimal portfolios across convention parameters exist only
> in fixed-coefficient exercises where changing α simultaneously changes the
> effective return law.

### Generated Files
- `figure5_identification.png` – Log-return density overlay and θ* comparison
- `figure5b_theta_comparison.png` – Fixed-μ vs law-preserving θ* contrast
- `exp5_identification_table.csv` – Full identification table
- `exp5_fixed_mu_table.csv` – Fixed-μ comparison table

---

## Summary of All Key Numbers

| Quantity | Value | Source |
|----------|-------|--------|
| Stratonovich–Itô shift (single asset) | **0.5 exactly** | Exp 1 |
| Figure 1 slope (all three markets) | **1.0 exactly** | Exp 1 |
| V⁻¹d_V for Matrix A | [1, 1, 1, 1] | Exp 2 |
| V⁻¹d_V for Matrix B | [−1.143, 1.429] | Exp 2 |
| V⁻¹d_V for Matrix C | [0.803, 0.619, 0.734] | Exp 2 |
| Law-preserving max deviation | 3.89 × 10⁻¹⁶ | Exp 2 |
| Heston θ_α (α = 1) | 0.0850 | Exp 4 |
| Heston K_α (α = 1) | −0.0400 | Exp 4 |
| Sign-change vol (α = 1) | **20%** | Exp 4 |
| Stationary P(V < 0.04), α = 1 | **23.1%** | Exp 4 |
| Return-law invariance across α pairs | confirmed | Exp 5 |

---

## Test Suite

```
91 passed in 2.18s
```

Tests cover: Itô conversion exactness, slope-1 verification, Stratonovich shift,
Remark 4.2 matrix outputs, Proposition 4.2 invariance, factor-model decomposition,
Heston derived quantities, sign-change thresholds, Feller conditions, stationary
probabilities, and identification table properties.
