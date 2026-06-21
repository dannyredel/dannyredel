# Synthetic Music Streams — DGP playground

A reproducible **data-generating process (DGP)** for music releases and their
**daily streams**, plus time-series plots and descriptive statistics.

## Files
- **`music_streams_dgp.ipynb`** — the notebook with all outputs and figures embedded
  (open and read top-to-bottom; no need to re-run).
- **`build_notebook.py`** — script that programmatically builds *and executes* the
  notebook. Edit the model here, then `python build_notebook.py` to regenerate.

## The model
For track *i* on a day with weekday *w*, *a* days after release:

```
mu[i,a] = (P_artist * Q_track)         # log-normal peak level (beta_i)
          * exp(-ln2/h_i * a)          # exponential decay, half-life h_i
          * s_w                        # day-of-week seasonality (Fri/weekend lift)
          * (1 + viral_bump[i,a])      # rare transient shocks (playlist/TikTok)
streams[i,a] ~ NegBinomial(mean=mu, r) # over-dispersed counts
```

Heterogeneous artist popularity × track quality (both log-normal) plus rare viral
shocks produce realistic **catalog concentration** (a few hits dominate).

## What the notebook covers
1. Setup & parameters · 2. Generate releases · 3. Generate daily streams ·
4. Panel summary · 5. Time series (catalog + example tracks) · 6. Decay in event
time (with fitted half-life) · 7. Day-of-week seasonality · 8. Distribution of
daily streams · 9. Concentration (Lorenz curve + Gini) · 10. Per-track table ·
11. Takeaways.

## Reproduce / tweak
```bash
pip install numpy pandas matplotlib nbformat nbconvert jupyter ipykernel
cd dgp-music-streams
python build_notebook.py        # rebuilds music_streams_dgp.ipynb with fresh outputs
```
Everything is driven by `SEED` and the `P` config block in §1 of the notebook
(also in `build_notebook.py`): `pop_sigma`/`qual_sigma` (spread ⇒ Gini),
`halflife_med`, `dow_raw`, the `viral_*` knobs, and `nb_r` (noise).
