# Hierarchical Bayesian MMM across the roster (Iteration 4)

Partial-pooling marketing-mix model over the European label's **40 releases**. Every
release gets its own media-response parameters, drawn from a **label-level distribution**,
so small releases **borrow strength** from big ones. Builds on iterations 1–3 (releases →
streams → ad-spend causality).

## Files
- **`hierarchical_mmm.html`** — open in any browser; all plots/tables/model output embedded.
- **`hierarchical_mmm.ipynb`** — editable notebook.
- **`build_notebook.py`** — rebuilds + re-executes (`python build_notebook.py`; 4 PyMC/nutpie fits).
- `dev_h.py`, `dev_h2.py` — scratch validation (DGP calibration + shrinkage thesis).

## The model
For release `r`, week `w`, channel `c`:
```
streams = B_r·exp(-d·w)                       # organic decay baseline
        + Σ_c A_c·θ_r·sat(adstock(spend; α_c); λ_c)   # paid media
        + ε
log θ_r = μ + γ·tier_r + σ·z_r                # hierarchical responsiveness
```
Fit three ways that differ **only** in `θ_r`: **no pooling** (independent), **complete
pooling** (one θ), **partial pooling** (the hierarchy).

## What it shows
1. **Shrinkage** — per-release ROAS: partial pooling has the lowest error (RMSE ≈ 14 vs 16
   no-pool, 20 complete-pool) and recovers the tier structure.
2. **Calibration** — feeding iteration 3's geo-experiment lift in as a prior de-biases and
   tightens the MMM's absolute scale (experiments + MMM are complements).
3. **Budget optimization** — reallocate the same total budget across releases & channels
   using the fitted saturation curves → more incremental streams at zero extra cost.

## Reproduce
```bash
pip install numpy pandas matplotlib scipy pymc pymc-marketing nutpie arviz \
            nbformat nbconvert jupyter ipykernel
cd hierarchical-mmm && python build_notebook.py
```

## Next
Propagate posterior uncertainty into the optimizer (risk-aware allocation), time-varying/
seasonal baselines, multi-objective (streams vs listener LTV), and productionize with
`pymc-marketing`'s multidimensional MMM fed by a recurring GeoLift experiment calendar.
