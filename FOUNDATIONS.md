# Foundations: Econometric & Causal Models for Music-Streaming Marketing Analytics

A reference document for the synthetic-data research in this repo — the methods, the
libraries that implement them, what we learned, and the literature behind them. It is the
distillation of a five-iteration arc that builds from descriptive simulation up to
causal incrementality, marketing-mix modelling, and a tool-calibration study.

> Scope: everything here was developed on **synthetic data** (known ground truth) so each
> method could be *scored*, not just run. All numbers below come from our own runs and are
> reproducible from the build scripts.

---

## 1 · Project map

| Iteration | Folder | Theme | Key artifact |
|---|---|---|---|
| 1 | `dgp-music-streams/` | DGP for releases & daily streams; descriptive stats | `music_streams_dgp.ipynb` |
| 2 | `label-campaign-2026/` | A concrete label: 10 artists, 40 releases, global | `label_campaign_2026.ipynb` |
| 3 | `causal-ad-spend/` | Ad spend → streams: the **causal ladder** (OLS→MMM/GeoLift) | `causal_ad_spend.ipynb` |
| 4 | `hierarchical-mmm/` | **Hierarchical Bayesian MMM** across 40 releases | `hierarchical_mmm.ipynb` |
| 5 | `geolift-calibration/` | **Are geo-lift tools interchangeable?** (Recast replication) | `geolift_calibration.ipynb` |

Each folder ships an executed notebook, a self-contained HTML export, a `build_notebook.py`,
and a README.

---

## 2 · The core modelling primitives

**Release-and-decay.** A track's daily streams spike at release and decay roughly
exponentially: `streams_t ≈ peak · exp(−ln2 · t / half_life)`. Half-lives in our DGPs are
~3–4 weeks. Catalogs are heavy-tailed: a few hits dominate (high Gini, steep Lorenz curve).

**Adstock (carryover).** Advertising effect persists and decays:
`adstock_t = spend_t + θ · adstock_{t−1}`, geometric decay rate `θ∈[0,1)`. Captures that a
TikTok push keeps paying off for days after the spend.

**Saturation (diminishing returns).** Response to spend is concave. Two common forms:
- **Hill**: `x^a / (x^a + κ^a)` — flexible S-curve, `κ` = half-saturation point.
- **Logistic**: `(1−e^{−λx}) / (1+e^{−λx})`.

**Media response = adstock → saturation → scale**, added on top of an organic baseline.
This is the backbone of both the causal DGP (iteration 3) and the MMM (iteration 4).

**Over-dispersed counts.** Streams are counts with variance > mean; we use Negative
Binomial / Poisson noise rather than Gaussian where appropriate.

---

## 3 · The causal-inference ladder (iteration 3)

The central problem: **ad spend is endogenous.** The label funds releases and moments it
already expects to do well, and pushes paid media into the post-launch *decay* phase. So
raw spend↔streams correlation is confounded — in our DGP it comes out **negative**
(corr ≈ −0.76): a naive analyst would conclude ads *hurt* streams. Each rung builds a more
credible counterfactual ("what would streams have been without the ads?").

| Rung | Method | Identifying assumption | Fixes | Failure mode | Library |
|---|---|---|---|---|---|
| 0 | **Naive OLS / pre-post** | none (correlation) | — | confounding; can flip sign | statsmodels |
| 1 | **TWFE** (unit+time FE) | selection on fixed effects | level differences | time-varying confounders | pyfixest |
| 2 | **DiD 2×2** | parallel trends | common shocks | trends differ in levels (use logs) | pyfixest |
| 3 | **Staggered DiD** (Sun–Abraham / Callaway–Sant'Anna) | parallel trends, no anticipation | **staggered-timing (negative-weight) bias** | needs clean controls | pyfixest, diff-diff |
| 4 | **Synthetic Control** | convex pre-fit of donors | one treated unit, weights | thin donor pool; no extrapolation | pysyncon |
| 5 | **Synthetic DiD** | unit **and** time weights | robustness vs SC/DiD | small donor pools | (scratch; R `synthdid`) |
| 6 | **Augmented SCM** (= GeoLift core) | ridge-relaxed fit | poor pre-fit / bias | extrapolation risk | pysyncon, GeoLift (R) |
| 7 | **Bayesian SC** | as SC + priors | **uncertainty quantification** | prior sensitivity | CausalPy |
| 8 | **Bayesian MMM** | adstock+Hill, calibrated | always-on budget, all channels | observational ⇒ needs calibration | pymc-marketing |

**Key results (our DGP, true lift ≈ +18–23%, iROAS ≈ 13 streams/€):**
- Naive OLS: **−154%** implied lift (sign-flipped, useless).
- DiD / staggered DiD / SC / synthdid / Augmented SCM / Bayesian SC: all land **near the
  truth** (≈ 18–25%). Relative bias collapses toward zero as you climb.
- **Design beats modelling**: the biggest credibility jump comes from the geo holdout +
  staggered rollout, not from any single estimator.
- Plain TWFE event studies are **biased under staggered adoption** (Goodman-Bacon) → use
  Sun–Abraham / Callaway–Sant'Anna.

---

## 4 · Marketing-Mix Modelling & hierarchical pooling (iteration 4)

**MMM** decomposes outcomes into baseline + each channel's adstocked, saturated
contribution; it answers the **always-on, all-channel budget** question and yields
response curves and marginal ROAS. It is **observational**, so it cannot by itself pin the
absolute scale of media effect — it must be **calibrated with experiments**.

**Why hierarchical / partial pooling.** With 40 releases, each has only a few weeks of
data:
- **No pooling** (a separate MMM per release) overfits — noisy, unstable ROAS.
- **Complete pooling** (one curve for all) ignores that a flagship and a debut respond
  differently.
- **Partial pooling** (hierarchical Bayes): each release gets its own parameters drawn from
  a label-level distribution, so small releases **borrow strength** from big ones. Estimates
  shrink toward the group — a lot when a release has little signal, little when it has lots.

`log θ_r = μ + γ·tier_r + σ·z_r` makes responsiveness vary by artist tier plus noise.

**Key results:**
- Per-release ROAS recovery (RMSE, lower=better): **no pooling 16.0**, complete pooling
  20.4, **partial pooling 14.0** (best; corr 0.76). Shrinkage wins.
- **Experiment calibration**: feeding iteration-3's geo lift in as a prior cut the MMM's
  total-incrementality bias from **+141% to +18%** — observational MMM over-attributes;
  experiments fix the level. *Run experiments **and** MMM — they are complements.*
- **Budget optimization**: reallocating the same budget toward responsive releases /
  unsaturated channels (within a ±50% band, equalizing marginal ROAS) lifted incremental
  streams **~+39%** at zero extra cost. (Point-estimate optimum; uncertainty-aware
  allocation is the next step.)

---

## 5 · Geo experiments & the tool-calibration study (iteration 5)

Geo experiments split regions into test/control and estimate lift with synthetic-control
logic — the cleanest way to measure incrementality when you can't randomize at the user
level. But **the tools are not interchangeable.**

**The Recast simulation study** (Robson Tigre / Recast,
[article](https://research.getrecast.com/geolift-sim-study/),
[code](https://github.com/getrecast/geolift-simulation-study)) ran **32,000 simulations**
benchmarking four tools — **Meta GeoLift** (augmented SC + conformal), **CausalPy**
(Bayesian SC), **Google Matched Markets** (time-based regression), **Google CausalImpact**
(BSTS) — across stress scenarios (textbook, 5× outlier geo, small donor pool, short
pre-period), measuring **coverage, false-positive rate, and power**. Conclusion: under
stress the tools diverge sharply and are **not drop-in replacements**.

**Our replication on the music-streaming case** (R unavailable, so GeoLift→its SC core,
CausalImpact cited not run; Matched Markets→a from-scratch TBR). DGP matches Recast's
features: log-normal geo baselines, shared trend, weekly seasonality, **AR(1)** noise.
Inject 0% (null) or +10% lift; measure FPR / power / coverage over many sims.

**Key findings (textbook scenario, our run):** all four roughly recover the +10% point
estimate, but their *inference* diverges wildly:

| Tool | FPR (want ~5%) | Power | Coverage (want ~95%) |
|---|---|---|---|
| DiD (cluster-robust, 1 treated geo) | **55%** | 100% | 44% |
| TBR (Gaussian PI, ignores AR(1)) | **46%** | 86% | 54% |
| Synthetic Control (placebo) | **11%** | 26% | 88% |
| CausalPy (naive credible interval) | **100%** | 100% | 8% |

- **Analytic-SE tools cry wolf.** DiD (a single treated cluster breaks cluster-robust SEs)
  and TBR (AR(1) noise violates the iid prediction interval) post false-positive rates of
  ~45–55% — they "find" lift that isn't there.
- **Design-based inference is honest but underpowered.** SC's placebo inference roughly
  holds its level (FPR ~5–11%, coverage ~90%) but has low power (~26%) with a modest donor
  pool — it misses real lifts.
- **Bayesian ≠ automatically calibrated.** Read naively (95% credible interval on the
  *latent* cumulative impact, few iterations), CausalPy was badly **over-confident**
  (~100% FPR, ~8% coverage). Judge significance against the **posterior-predictive**
  counterfactual (including observation noise) and set sensible priors.
- **Takeaway**: it's the **inference method**, not the point estimate, that decides trust.
  Validate calibration for *your* setup, and **design for power** (enough control geos,
  long pre-period, sufficient spend) before running a geo test.

**Inference matters more than the estimator.** Few-treated-cluster corrections
(Conley–Taber, wild-cluster bootstrap), placebo/permutation, and conformal inference exist
precisely to fix the over-rejection seen above.

---

## 6 · Library catalog

| Library | Lang | What it does | Used for | Maturity notes |
|---|---|---|---|---|
| **pyfixest** | Py | Fast fixed-effects regressions; DiD, event studies (Sun–Abraham, did2s, lpdid) | Rungs 1–3 | Active, validates vs R `fixest` |
| **linearmodels** | Py | Panel/IV estimators | panel utilities | Mature |
| **pysyncon** | Py | Synthetic Control, Augmented SC (AugSynth), penalized SC | Rungs 4 & 6 | Focused, reliable |
| **CausalPy** (pymc-labs) | Py | Bayesian causal inference: SC, DiD, ITS, RDD | Rung 7, geo study | Active, PyMC-backed |
| **PyMC** | Py | Probabilistic programming (NUTS) | Hierarchical MMM | Mature, core |
| **pymc-marketing** | Py | Bayesian MMM (adstock, Hill), multidim/hierarchical | Rung 8, iteration 4 | Active, production-grade |
| **nutpie** | Py/Rust | Fast NUTS sampler | speed (~6× vs default) | Active |
| **statsmodels / scipy** | Py | OLS, optimization | baselines, budget opt | Mature |
| **diff-diff** (igerber) | Py | Unified DiD suite: Callaway–Sant'Anna, Sun–Abraham, SyntheticDiD, **HonestDiD**, **Bacon decomposition**, ETWFE | candidate for robustness add-ons | v3.x, MIT, single-maintainer; validates vs R `did`/`synthdid`/`fixest` — convenient, cross-check for production |
| **Meta GeoLift** | R | Augmented SC + power analysis for geo tests | cited (R-only here) | Industry standard |
| **Google CausalImpact** | R / Py | BSTS counterfactual | cited (R-only; Py port breaks PyMC env) | Standard |
| **Google Matched Markets** | R / Py | Time-based regression (TBR/GBR) | re-implemented as TBR | Standard |
| **Robyn** (Meta) | R | Open-source MMM (ridge + evolutionary opt) | reference | Industry standard |

**Environment caution:** `tfcausalimpact` pulls TensorFlow and pins NumPy < 2, which breaks
PyMC/PyTensor (needs NumPy ≥ 2). Keep BSTS in a separate environment, or use R.

---

## 7 · References

**Difference-in-Differences & staggered designs**
- Card & Krueger (1994), minimum-wage DiD.
- Goodman-Bacon (2021), *DiD with variation in treatment timing*, J. Econometrics.
- Callaway & Sant'Anna (2021), *DiD with multiple time periods*, J. Econometrics.
- Sun & Abraham (2021), *Estimating dynamic effects with heterogeneous treatment timing*.
- de Chaisemartin & D'Haultfœuille (2020), *Two-way FE with heterogeneous effects*, AER.
- Roth, Sant'Anna, Bilinski, Poe (2023), *What's trending in DiD* (review).
- Rambachan & Roth (2023), *A more credible approach to parallel trends* (HonestDiD), ReStud.
- Conley & Taber (2011); Cameron, Gelbach & Miller (2008), few-treated-cluster inference.

**Synthetic control family**
- Abadie & Gardeazabal (2003); Abadie, Diamond, Hainmueller (2010, 2015).
- Abadie (2021), *Using synthetic controls*, J. Economic Literature.
- Ben-Michael, Feller & Rothstein (2021), *Augmented synthetic control*, JASA.
- Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021), *Synthetic DiD*, AER.
- Chernozhukov, Wüthrich & Zhu (2021), conformal inference for SC, JASA.

**Geo experiments & incrementality**
- Vaver & Koehler (2011), *Measuring ad effectiveness using geo experiments*, Google.
- Kerman, Wang & Vaver (2017), *Estimating ad effectiveness using GBR/TBR*, Google.
- Brodersen, Gallusser, Koehler, Remy & Scott (2015), *Inferring causal impact using BSTS*
  (CausalImpact), Annals of Applied Statistics.
- Meta GeoLift documentation: facebookincubator.github.io/GeoLift.
- Recast geo-lift simulation study: research.getrecast.com/geolift-sim-study ·
  github.com/getrecast/geolift-simulation-study.

**Marketing-Mix Modelling**
- Broadbent (1979), adstock.
- Jin, Wang, Sun, Chan & Koehler (2017), *Bayesian methods for MMM with carryover and
  shape effects*, Google.
- Chan & Perry (2017), *Challenges and opportunities in MMM*, Google.
- pymc-marketing & Meta Robyn documentation.

**Hierarchical / Bayesian modelling**
- Gelman & Hill (2007), *Data Analysis Using Regression and Multilevel/Hierarchical Models*.
- McElreath (2020), *Statistical Rethinking*.
- Betancourt (2017), *A conceptual introduction to HMC*.

---

## 8 · Reproducibility

```bash
pip install numpy==2.2.6 pandas scipy statsmodels matplotlib \
            pyfixest linearmodels pysyncon causalpy pymc pymc-marketing nutpie arviz \
            nbformat nbconvert jupyter ipykernel
# then, in any iteration folder:
python build_notebook.py            # regenerates the executed notebook
jupyter nbconvert --to html --embed-images <notebook>.ipynb
```

Every DGP exposes the ground-truth (`organic` / `incremental` / true lift), so any new
estimator can always be scored against the truth — the throughline of this whole project.
