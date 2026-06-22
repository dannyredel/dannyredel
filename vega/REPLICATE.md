# Replicating the Vega Identification Study — Notebook Guide

A step-by-step guide to rebuilding and running the study in a Jupyter notebook
(VS Code). The heavy lifting already lives in the `vega/` package — the notebook
is a thin **driver + inspection layer** on top of it. You do **not** rewrite the
model; you import it, run it at a power level your machine can afford, and read
the results.

> **Why a notebook?** The study is a Monte-Carlo loop of Bayesian fits. A notebook
> lets you (a) sanity-check the data-generating process and a single fit
> interactively, then (b) scale up cell-by-cell with checkpoints, watching
> convergence and timing as you go — instead of launching an opaque multi-hour job.

---

## 0. Prerequisites & environment

The repo layout (already on disk):

```
vega/
  config.py    scenario grid, planted-truth constants, decision thresholds, NUTS profiles
  dgp.py       data-generating process: plant truth -> simulate the 8 tables
  geo.py       geo-lift module: randomized holdout -> meta-analysis -> Meta prior
  model.py     the estimator under test (NumPyro hierarchical NB MMM) + mROAS posterior
  metrics.py   scoring: bias, coverage, width, keep/cut accuracy, kappa detection, convergence
  study.py     the Monte-Carlo harness (run_cell/run_study/run_misspec/run_lowspend) + verdict
  report.py    renders report_*.md
  plots.py     renders recovery_*.png
  outputs/     results_*.json, report_*.md, *.png land here
run_study.py   CLI entry point (mirror of what the notebook does)
```

### Set up in VS Code

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install numpyro jax arviz numpy matplotlib jupyter ipykernel
```

> **GPU (strongly recommended for the full grid):** install the CUDA build of JAX
> instead — `pip install -U "jax[cuda12]"`. NumPyro picks it up automatically; a
> GPU cuts the per-fit cost ~10–50× and turns the M≥100 grid from days into hours.

In VS Code: open the folder, create `replicate.ipynb` at the **repo root** (so
`from vega import ...` resolves), and select the `.venv` kernel.

---

## 1. Cell 1 — setup & imports

Put JAX/threading config **first**, before anything imports JAX.

```python
import os
# --- compute knobs (set BEFORE importing jax/numpyro) ---
# Do NOT force multiple host devices on CPU: it makes NumPyro replicate the model
# across devices (~Nx memory) and was the cause of OOM-kills in earlier runs.
os.environ.setdefault("XLA_FLAGS", "")                 # leave host-device count at default (1)
os.environ.setdefault("OMP_NUM_THREADS", "4")          # match your physical cores
# os.environ["VEGA_CHAIN_METHOD"] = "sequential"       # uncomment on a tight-memory box

import numpy as np
import jax
print("JAX devices:", jax.devices())                   # CPU -> [CpuDevice]; GPU -> [cuda:0]

from vega import config as cfg, dgp, geo, model, metrics, study
from vega import report, plots
```

---

## 2. Cell 2 — sanity-check the data-generating process

Plant the truth and simulate one roster; confirm the tables look like a real
label's data before trusting any recovery number.

```python
scn = cfg.Scenario(N=50, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True)
data, truth = dgp.simulate_roster(scn, seed=0)

print("scenario:", scn.label())
print("planted channel mROAS:", truth["mroas"])        # {meta:9, tiktok:14, spotify:4, plugger:2.5}
print("planted kappa (organic->paid multiplier):", truth.get("kappa"))
print("tables/keys:", list(data.keys()))
print("streams matrix:", np.asarray(data["streams"]).shape, "(releases x weeks)")
```

What to check: high-spend channels (Meta, TikTok) dominate the spend mix; the
break-even bar is €6 mROAS so Spotify (4) and Plugger (2.5) are *below* the bar and
Meta (9) / TikTok (14) are *above* — those are the keep/cut ground-truth labels.

---

## 3. Cell 3 — the geo-lift prior

The `pooled_geo` estimator anchors Meta's mROAS with a randomized-holdout lift
meta-analysis. Inspect it:

```python
gp = geo.geo_prior(data)            # None if scn.geo_anchor is False
print("geo prior on Meta mROAS:  mean=%.2f  sd=%.2f  (from %d experiments)"
      % (gp["mean"], gp["sd"], gp["n_experiments"]))
```

---

## 4. Cell 4 — a single fit (the inner loop, unpacked)

This is exactly what `study._fit_one` does, spelled out so you can watch one fit.
Start at the **fast profile** (`cfg.FAST_NUTS`: 600+600 × 4 @ depth 7) — it's the
"knee" where the estimand parameters converge without the full 1000+1000 cost.

```python
import time
nuts = cfg.FAST_NUTS                                    # {'warmup':600,'samples':600,'chains':4,...}

t = time.time()
mcmc, design = model.fit(
    data, estimator="pooled_geo", geo_prior=gp,
    num_warmup=nuts["warmup"], num_samples=nuts["samples"], num_chains=nuts["chains"],
    target_accept=nuts["target_accept"], max_tree_depth=nuts["max_tree_depth"],
    seed=0,
)
print(f"fit took {time.time()-t:.0f}s")                 # CPU: ~40-140s; GPU: a few s

diag  = model.convergence_diagnostics(mcmc)             # per-param R-hat, divergences, ESS
roster, kappa, _ = model.mroas_posterior(mcmc, data, design)
score = metrics.score_fit(roster, kappa, truth, data, design)

print("converged?", metrics.is_converged(diag), "| max R-hat=%.3f" % diag["max_rhat"],
      "| divergences=", diag["n_divergent"], "| div rate=%.4f" % diag["div_rate"])
print("keep/cut accuracy:", score["class_accuracy"],
      "| median |rel bias|:", round(score["median_abs_rel_bias"], 3),
      "| kappa detected:", score["kappa_detected"])
```

> **First-fit slowness is XLA compilation, not sampling** — the second fit of the
> same shape is much faster. This is why the harness clears the compile cache per
> fit inside `run_cell` (XLA caches but does not reuse fresh NUTS executables, so an
> un-cleared many-fit loop OOMs mid-compile).

---

## 5. Cell 5 — one Monte-Carlo cell (M replicates + convergence gating)

`study.run_cell` runs M simulations, fits each estimator, **discards sims that fail
the convergence gate** (estimand R-hat ≤ 1.01 and divergence-rate ≤ 0.5%), and
aggregates the survivors. Start tiny to gauge wall-time, then scale M.

```python
M = 6                                                   # start small; raise to 100+ on a GPU
cell = study.run_cell(scn, M, estimators=["pooled", "pooled_geo"], nuts=cfg.FAST_NUTS)

pg = cell["pooled_geo"]
print("n converged sims:", pg["n_sims"])
print("class accuracy:   ", round(pg["class_accuracy"], 3))      # H3  (>0.80 = GO)
print("median |rel bias|:", round(pg["median_abs_rel_bias"], 3)) # H1  (<0.25 = GO)
print("80% CI coverage:  ", round(pg["coverage"], 3))            # H2  (0.70-0.90)
print("kappa detected:   ", round(pg["kappa_detected_frac"], 3)) # H5
conv = pg["convergence"]                                         # convergence yield
print("converged / attempted:", conv["n_converged"], "/", conv["n_attempted"],
      "| frac dropped:", round(conv["frac_dropped"], 2),
      "| strict zero-div rate:", round(conv["frac_strict_zero_div"], 2))
print("per-channel bias: ", {c: round(pg["channels"][c]["median_rel_bias"], 2)
                             for c in cfg.CHANNELS})              # watch Spotify/Plugger drift up
```

---

## 6. Cell 6 — the κ = 0 null (false-positive test, H5)

The most important falsification: when there is **no** organic→paid multiplier
(κ=0), the estimator must **not** claim to detect one. Run the GO cell with κ=0.8
(power) and a twin with κ=0 (false-positive rate).

```python
go_cell = study.run_cell(cfg.Scenario(N=50, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),
                         M, ["pooled_geo"], cfg.FAST_NUTS)
k0_cell = study.run_cell(cfg.Scenario(N=50, rho=0.6, kappa=0.0, gamma=0.5, geo_anchor=True),
                         M, ["pooled_geo"], cfg.FAST_NUTS)

print("power  @ kappa=0.8:", go_cell["pooled_geo"]["kappa_detected_frac"])  # want high (>=0.80)
print("FP rate @ kappa=0.0:", k0_cell["pooled_geo"]["kappa_detected_frac"])  # want low  (<=0.20)
```

`kappa_detected` = the 80% posterior CI for κ clears a ROPE of ±0.05 around zero
(`cfg.KAPPA_ROPE`). A high FP rate here would sink the GO verdict regardless of the
recovery numbers.

---

## 7. Cell 7 — the verdict grid (H1–H8) and the GO/KILL call

The full grid sweeps ρ (collinearity), κ (multiplier), N (roster size), and γ
(endogeneity). Use the helpers in `config`:

```python
grid = cfg.verdict_grid()                # ~12 cells
for s in grid: print(s.label())

# Run the whole grid (writes vega/outputs/results_verdict.json, checkpointed per cell).
# Keep M small on CPU; M>=100 needs a GPU.
study.run_study(grid, M=6, nuts=cfg.FAST_NUTS, tag="verdict")

import json
verdict = json.loads(open("vega/outputs/results_verdict.json").read())
summary = study.conclude(verdict)        # applies the H1-H8 ledger + GO/RESCOPE/KILL rules
print("VERDICT:", summary["verdict"])
for h, r in summary["ledger"].items():
    print(f"  {h:14s} {'PASS' if r['pass'] else 'FAIL'}  {r['detail']}")
```

Decision thresholds live in `config.py` (`CLASS_ACCURACY_GO=0.80`, `REL_BIAS_GO=0.25`,
`COVERAGE_LO/HI`, `KAPPA_FALSE_POSITIVE_MAX=0.20`, `POOLING_RMSE_REDUCTION=0.20`).

---

## 8. Cell 8 — misspecification battery (does the verdict survive a wrong model?)

Refit under deliberately wrong specs and compare. This is what separates a
"correct-spec GO" from a "robust GO".

```python
specs = ["correct", "wrong_adstock", "wrong_saturation", "omit_editorial", "poisson"]
study.run_misspec(cfg.misspec_cells(), specs, M=6, nuts=cfg.FAST_NUTS, tag="misspec")

misspec = json.loads(open("vega/outputs/results_misspec.json").read())
summary = study.conclude(verdict, misspec=misspec)
print("correct-spec verdict:", summary["verdict_correct_spec"])
print("misspec-robust verdict:", summary["verdict_misspec_robust"])
```

---

## 9. Cell 9 — low-spend prior mitigation (the Spotify/Plugger fix)

The low-spend channels bias upward because the channel-symmetric prior — centred on
the (high-spend-dominated) roster average — pulls them up. Compare a global prior vs
a per-channel prior:

```python
go = next(s for s in cfg.verdict_grid() if s.label() == cfg.GO_LABEL)
study.run_lowspend(go, M=6, nuts=cfg.FAST_NUTS, tag="lowspend")
lowspend = json.loads(open("vega/outputs/results_lowspend.json").read())
print(json.dumps(study.lowspend_floor(lowspend), indent=2, default=float))
```

---

## 10. Cell 10 — render the report + figures

```python
report.render()       # -> vega/outputs/report_verdict.md
plots.plot_all()      # -> vega/outputs/recovery_verdict.png
```

---

## 11. Scaling to decision power (the only real cost)

Everything above runs at exploratory power (`M=6`, `FAST_NUTS`). The **decision-grade**
result needs:

| knob | exploratory | decision-grade |
|---|---|---|
| `M` (sims/cell) | 6 | **≥100** |
| NUTS | `cfg.FAST_NUTS` (600+600, depth 7, accept 0.88) | 1000+1000, depth 8–10, accept 0.9 (`cfg.NUTS_*` defaults) |

Decision-grade NUTS in any cell:

```python
nuts = dict(warmup=cfg.NUTS_WARMUP, samples=cfg.NUTS_SAMPLES, chains=cfg.NUTS_CHAINS,
            target_accept=cfg.NUTS_TARGET_ACCEPT, max_tree_depth=cfg.NUTS_MAX_TREE_DEPTH)
```

**Cost reality (measured on a 4-CPU box):** ~40–140 s per converged fit. A 12-cell
grid × 100 sims × ~2 estimators ≈ thousands of fits ⇒ **multi-day on CPU**. On a GPU
it's a few hours. Run the full grid headless rather than in the notebook:

```bash
python run_study.py --mode all --M 100 \
    --warmup 1000 --samples 1000 --chains 4 --target-accept 0.9 --tree-depth 10
```

`run_study.py` is the exact CLI mirror of cells 5–10; `--mode {verdict,misspec,lowspend,all}`.

---

## 12. Troubleshooting (lessons from the CPU runs)

| Symptom | Cause | Fix |
|---|---|---|
| Process **OOM-killed** silently early | `XLA_FLAGS=--xla_force_host_platform_device_count=N` replicates the model N× | Don't set it (default 1 device); chains vectorize on one device |
| OOM **mid-way** through a long cell | XLA compile-cache accretion across fits | already handled — `run_cell` calls `jax.clear_caches()` + `gc.collect()` per fit |
| Tight-memory box still OOMs | 4 vectorized chains held at once | `os.environ["VEGA_CHAIN_METHOD"]="sequential"` (≈4× slower, ≈4× less peak memory) |
| `frac_dropped` high / many discards | collinearity → slow mixing, a few divergences per sim | expected; the gate uses estimand R-hat + a 0.5% div-rate tolerance, and reports the *strict* zero-divergence rate as a result. Reparameterising σ / the baseline↔paid ridge would reduce divergences |
| R-hat ~1.2–1.4, fits "fast" | `max_tree_depth` too low (5) | use depth ≥7; depth-7 is the convergence knee |
| First fit slow, rest fast | XLA JIT compilation | normal — compilation amortizes within a same-shape loop |

---

## 13. What a passing run looks like (reference, from the smoke-grade run)

On the GO cell (`N50_rho0.6_k0.8_g0.5_geo`, `pooled_geo`): keep/cut accuracy ≈ 0.88,
median |rel bias| ≈ 0.13, 80% coverage ≈ 0.88, κ detected in ~100% of sims at κ=0.8,
pooling cuts RMSE ~85% vs no-pool, geo tightens intervals ~20%. Per-channel: Meta/TikTok
recover clean (bias ≈ ±0.04); **Spotify/Plugger bias upward** (+0.27 / +0.44), with
Spotify straddling the €6 bar ~75% of the time — the known weak point. See
`vega/FINDINGS.md` for the full write-up and the pending decision-grade items.
