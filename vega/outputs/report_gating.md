# Vega — Identification Study Recovery Report

*Generated from `results_gating.json`. M=4 sims/cell, weekly T=12, 300+250 NUTS draws x 2 chains. Elapsed ~14min (crashed on final N25 cell; verdict cells intact)s.*

## Verdict: **GO**

Decision rules per study §8 (GO needs class acc >0.80, |rel bias| <0.25, coverage in [0.70,0.90], kappa-CI excludes 0, and pooling to beat no-pool).

### Hypothesis ledger (H1-H8)

| Hypothesis | Result | Detail |
|---|---|---|
| H1_recovery | PASS ✅ | |rel bias|=0.13 (<0.25) |
| H2_coverage | PASS ✅ | coverage=0.88 in [0.70,0.90] |
| H3_decision | PASS ✅ | class acc=0.88 (>0.80) |
| H4_pooling | PASS ✅ | RMSE reduction=0.85 (>=0.20) |
| H5_kappa | PASS ✅ | kappa excl 0 in 1.00 of sims |
| H8_geo_value | PASS ✅ | geo cuts width 0.20, bias 0.03 |

- Pooling RMSE reduction (H4): **0.85** (threshold ≥0.20)
- Geo anchor width reduction (H8): **0.197**
- Geo anchor bias reduction (H8): **0.026**
- RMSE vs roster size (H4): N10=2.91, N25=None, N50=1.85

### Per-cell recovery

| Scenario | Estimator | class acc | \|rel bias\| | coverage | 80% width | RMSE | kappa excl 0 |
|---|---|---|---|---|---|---|---|
| N50_rho0.3_k0.8_g0.5_geo | pooled | 0.81 | 0.24 | 0.81 | 4.3 | 2.01 | 1.00 |
| N50_rho0.3_k0.8_g0.5_geo | pooled_geo | 0.81 | 0.26 | 0.81 | 5.0 | 2.04 | 1.00 |
| N50_rho0.6_k0.8_g0.5_geo | nopool | 0.50 | 1.39 | 0.12 | 5.2 | 12.35 | 1.00 |
| N50_rho0.6_k0.8_g0.5_geo | pooled | 0.88 | 0.17 | 0.81 | 4.0 | 1.85 | 1.00 |
| N50_rho0.6_k0.8_g0.5_geo | pooled_geo | 0.88 | 0.13 | 0.88 | 3.8 | 1.76 | 1.00 |
| N50_rho0.9_k0.8_g0.5_geo | pooled | 0.88 | 0.16 | 0.81 | 4.0 | 1.73 | 1.00 |
| N50_rho0.9_k0.8_g0.5_geo | pooled_geo | 0.88 | 0.14 | 0.75 | 3.7 | 1.84 | 1.00 |
| N50_rho0.3_k0.8_g0.5_nogeo | pooled | 0.81 | 0.16 | 0.81 | 4.9 | 1.83 | 1.00 |
| N50_rho0.6_k0.8_g0.5_nogeo | pooled | 0.88 | 0.13 | 0.75 | 4.7 | 1.86 | 1.00 |
| N50_rho0.9_k0.8_g0.5_nogeo | pooled | 0.88 | 0.20 | 0.69 | 4.6 | 1.74 | 1.00 |
| N10_rho0.6_k0.8_g0.5_geo | nopool | 0.44 | 1.43 | 0.31 | 10.1 | 11.18 | 1.00 |
| N10_rho0.6_k0.8_g0.5_geo | pooled | 0.75 | 0.30 | 0.88 | 8.1 | 2.91 | 1.00 |
| N10_rho0.6_k0.8_g0.5_geo | pooled_geo | 0.75 | 0.29 | 0.88 | 6.9 | 1.96 | 1.00 |

### Per-channel recovery — GO cell (N=50, rho=0.6, kappa=0.8, geo on, pooled+geo)

| Channel | true mROAS | median rel bias | coverage | class acc | straddle bar | above bar? |
|---|---|---|---|---|---|---|
| meta | 9.0 | -0.04 | 1.00 | 1.00 | 0.00 | yes |
| tiktok | 14.0 | +0.04 | 0.75 | 1.00 | 0.00 | yes |
| spotify | 4.0 | +0.27 | 0.75 | 0.75 | 0.75 | no |
| plugger | 2.5 | +0.44 | 1.00 | 0.75 | 0.50 | no |

> Break-even bar = 6.0 streams/€. Spotify (coarse-geo, low-spend) is expected to be the chronically hard-to-classify channel (study §9).
