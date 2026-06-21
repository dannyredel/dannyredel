# Vega — Counterpoint Identification Study

A synthetic-data **parameter-recovery / simulation-based-calibration** study. We
plant known channel effects and an organic→paid multiplier, simulate music-industry
data with its four pathologies (collinearity, scarcity, thin per-release data,
endogeneity), run a hierarchical Bayesian MMM + geo-lift calibration, and measure
whether the estimator gets the truth back.

> If recovery fails on data we built to be recoverable, real data has no chance.
> This study **gates the paid pilot** (design spec §11).

## What it tests (the falsifiable hypotheses, spec §2)

| # | Hypothesis | Falsifies if… |
|---|---|---|
| H1 | channel marginal-ROAS recovered with low bias | median \|rel bias\| > 25% |
| H2 | 80% credible intervals are honest | coverage outside [0.70, 0.90] |
| H3 | intervals tight enough to classify keep/cut vs the break-even bar | class accuracy < 80% |
| H4 | roster pooling beats single-release estimation | RMSE reduction < 20% |
| H5 | the multiplier κ is identifiable | κ-CI includes 0 at κ=0.8 |
| H6 | there is a viable collinearity operating region | H1–H3 fail at every ρ |
| H8 | the geo anchor tightens / de-biases the MMM | geo cuts neither width nor bias |

## Layout

```
vega/
  config.py   planted truth, scenario grid, decision thresholds (spec §5, §8)
  dgp.py      data-generating process: plant truth -> simulate the panel (§4)
  geo.py      geo-lift calibration loop: holdout -> meta-analysis -> prior (§6.1)
  model.py    the estimator under test: NumPyro hierarchical NB MMM (§6)
  metrics.py  recovery scoring: bias, coverage, width, classification, κ (§2)
  study.py    Monte-Carlo harness + GO/RESCOPE/KILL verdict (§7, §8)
  plots.py    the four headline recovery plots (§7)
  report.py   markdown recovery report (§11 deliverable #2)
  outputs/    results_*.json, verdict.json, report_*.md, recovery_*.png
```

## Run

```bash
pip install numpyro matplotlib            # jax comes with numpyro

python run_study.py --smoke               # plumbing check (~minutes)
python run_study.py --mode gating --M 8   # the decision-relevant grid (~40 min)
python run_study.py --mode full  --M 200  # the spec's ~144-cell grid (offline)
```

The harness runs NUTS chains **in parallel across cores**; set
`XLA_FLAGS=--xla_force_host_platform_device_count=N` and `OMP_NUM_THREADS=1` to
match your box.

## Modelling choices (and why)

The study's identification question is *which channel did the work* (β) and *is
the multiplier real* (κ), under collinearity / scarcity / endogeneity. So:

- **Estimated:** β (hierarchical, non-centered partial pooling), κ, the organic
  baseline (level, decay, editorial control), NB overdispersion φ.
- **Calibrated (treated as known):** adstock retention θ, Hill saturation k/s,
  spillover half-life λ — as in practice these are set by priors, not data. This
  isolates the identification claim and keeps NUTS fast and stable.
- **Cadence:** weekly × 12 periods (the "8–12 week campaign") rather than 84
  daily points — same structure, far fewer observations. Configurable in
  `config.N_PERIODS`.
- **DGP ≡ estimator** in functional form, so any recovery failure is
  *identification*, not misspecification.

The marginal-ROAS estimand reduces to `mROAS[r,c] = (1+κ/(1-λ)) · β[r,c] · C[r,c]`
where `C` is the fixed Hill-slope / carryover chain-rule constant — identical for
the planted truth and the recovered posterior, so "did we get it back?" is exact.
