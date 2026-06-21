# 2026 Release Campaign — European Label

Scenario analytics for a mid-size **European independent label**: **10 artists**,
**40 single releases** across calendar year **2026**, released **globally**. This
iteration is **descriptive only** — no ad spend yet.

## Files
- **`label_campaign_2026.html`** — open in any browser; all plots + tables embedded.
- **`label_campaign_2026.ipynb`** — editable notebook.
- **`build_notebook.py`** — rebuilds and re-executes the notebook (`python build_notebook.py`).

## Roster (proposed, realistic for a mid-size indie)
| Tier | Artists | Typical single peak (streams/day) |
|---|---|---|
| Flagship | 1 | ~30–45k |
| Mid-tier | 3 | ~7–13k |
| Emerging | 4 | ~1–3k |
| Developing | 2 | ~0.2–0.5k |

Singles drop on **Fridays** ("New Music Friday"), spread Jan–early Dec so each
release has runway before year-end.

## DGP
`mu = peak · exp(-ln2·a/half_life) · day_of_week · (1 + viral_bump)`, observed as
`NegBinomial(mu, r)`. `peak = artist_tier_level × release_quality`.

## What's inside
Release calendar · total & cumulative daily streams · streams by artist (stacked +
ranked) · event-time spike/decay with fitted half-life · day-of-week seasonality ·
daily-stream distribution + Lorenz/Gini concentration · per-release & per-artist tables.

## Next iteration — ad spend
Every release carries a latent `peak` (β) that ad spend will scale:
`peak_effective = β · (1 + κ·log(1 + spend/s₀))` (diminishing returns). Compare
streams with vs without spend to estimate ROI / cost-per-stream. Inputs are the
`panel` (track-day) and `releases` (one row per single) tables.
