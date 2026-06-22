# Ad Spend & Causal Inference — From OLS to MMM & GeoLift (Iteration 3)

Causal measurement of **paid-social impact on Spotify streams** for the European-label
scenario. Adds **ad spend** (TikTok, Instagram), frames the **causal questions**, and
climbs the full ladder of estimators — each scored against a **known ground truth** baked
into the data-generating process.

## Files
- **`causal_ad_spend.html`** — open in any browser; all plots, tables and model output embedded.
- **`causal_ad_spend.ipynb`** — editable notebook.
- **`build_notebook.py`** — rebuilds + re-executes (`python build_notebook.py`; ~6 min, runs 2 PyMC fits).
- `dev.py`, `dev2.py` — scratch validation harness used while developing the estimators.

## Data (assumed first-party)
- **Streams** — daily, per country, from Spotify for Artists.
- **Spend** — daily, per country, per channel, from TikTok / Meta Ads APIs.

## The DGP (with ground truth)
Country × day panel for one flagship single across **12 European markets**, 17 weeks:
`streams = organic(spike·decay·seasonality·common shock) + incremental(Hill(adstock(spend)))`.
Paid boost rolls out **staggered** across 8 treated countries (3 cohorts) with **4 held-out
controls**. Spend is **confounded** (timed into the decay phase) so naive correlation is
negative. True effect ≈ **+20% lift, ~13 streams/€**.

## The five causal questions
1. **Incrementality** (iROAS, cost-per-incremental-stream) · 2. **Channel MMM** (adstock +
saturation, marginal ROAS) · 3. **Per-campaign geo-lift** · 4. **Heterogeneity** · 5.
**Dynamics** (carryover, tail extension).

## The ladder (and the library used)
| Rung | Method | Library |
|---|---|---|
| 0 | Naive OLS / pre-post (biased, sign-flipped) | statsmodels |
| 1 | TWFE on spend | pyfixest |
| 2 | DiD 2×2 | pyfixest |
| 3 | Staggered DiD — Sun & Abraham event study | pyfixest |
| 4 | Synthetic Control (+ placebo inference) | pysyncon |
| 5 | Synthetic DiD | from scratch (cf. R `synthdid`) |
| 6 | Augmented SCM — the **GeoLift** engine | pysyncon |
| 7 | Bayesian Synthetic Control (credible intervals) | CausalPy |
| 8 | Bayesian MMM (adstock + Hill, contributions, ROAS) | pymc-marketing |

**Punchline:** relative bias vs ground truth collapses toward zero as you climb. Naive OLS
is wildly off (negative); every credible design recovers the truth within noise.

## Reproduce
```bash
pip install numpy pandas matplotlib scipy statsmodels linearmodels pyfixest \
            pysyncon causalpy pymc-marketing nbformat nbconvert jupyter ipykernel
cd causal-ad-spend && python build_notebook.py
```

## Next iteration — borrowing strength across releases
Pool the label's 40 releases: **hierarchical / partial-pooling Bayesian MMM** (per-release
adstock & saturation under label-level priors — the *pooled/Bayesian* idea), **multi-cell
GeoLift with power analysis**, **budget optimisation** from the saturation curves, and
**CUPED** variance reduction. `make_data()` exposes `organic`/`incremental` so any new
estimator can be scored against ground truth.

> Production notes: GeoLift and `synthdid` are R packages (we replicate their cores in
> Python and cite them); MMM in production calibrates its priors with the geo experiments —
> the two approaches are complements, not rivals.
