"""Headline recovery plots (study §7).

  * CI width vs roster size N            -> H4 (does pooling buy tightness?)
  * bias vs rho, with/without geo anchor -> H6, H8 (collinearity boundary)
  * coverage vs N                        -> H2 (is the uncertainty honest?)
  * classification accuracy heatmap      -> H3 (the money plot)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as cfg

OUTDIR = Path(__file__).parent / "outputs"


def _load(tag="gating"):
    return json.loads((OUTDIR / f"results_{tag}.json").read_text())


def _cell(res, label, est):
    r = res.get(label)
    return r["cell"].get(est, {}) if r else {}


def plot_all(tag="gating"):
    res = _load(tag)
    rhos = [0.3, 0.6, 0.9]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # --- (1) bias vs rho, geo on/off (H6, H8) -------------------------------
    ax = axes[0, 0]
    for geo_on, style in [(True, "o-"), (False, "s--")]:
        g = "geo" if geo_on else "nogeo"
        est = "pooled_geo" if geo_on else "pooled"
        ys = [_cell(res, f"N50_rho{r}_k0.8_g0.5_{g}", est).get("median_abs_rel_bias", np.nan)
              for r in rhos]
        ax.plot(rhos, ys, style, label=f"{'geo anchor' if geo_on else 'no geo'}")
    ax.axhline(cfg.REL_BIAS_GO, color="red", ls=":", label="H1 threshold (0.25)")
    ax.set_xlabel("inter-channel correlation rho"); ax.set_ylabel("median |relative bias|")
    ax.set_title("H6/H8 — bias vs collinearity"); ax.legend(); ax.grid(alpha=0.3)

    # --- (2) classification accuracy vs rho (H3) ----------------------------
    ax = axes[0, 1]
    for geo_on, style in [(True, "o-"), (False, "s--")]:
        g = "geo" if geo_on else "nogeo"
        est = "pooled_geo" if geo_on else "pooled"
        ys = [_cell(res, f"N50_rho{r}_k0.8_g0.5_{g}", est).get("class_accuracy", np.nan)
              for r in rhos]
        ax.plot(rhos, ys, style, label=f"{'geo anchor' if geo_on else 'no geo'}")
    ax.axhline(cfg.CLASS_ACCURACY_GO, color="red", ls=":", label="H3 threshold (0.80)")
    ax.set_xlabel("inter-channel correlation rho"); ax.set_ylabel("classification accuracy")
    ax.set_title("H3 — the money plot: keep/cut accuracy"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)

    # --- (3) width & RMSE vs N (H4) -----------------------------------------
    ax = axes[1, 0]
    Ns = [10, 50, 100]
    widths = [_cell(res, f"N{n}_rho0.6_k0.8_g0.5_geo", "pooled").get("mean_width", np.nan)
              for n in Ns]
    rmse_pool = [_cell(res, f"N{n}_rho0.6_k0.8_g0.5_geo", "pooled").get("rmse", np.nan)
                 for n in Ns]
    rmse_nopool = [_cell(res, f"N{n}_rho0.6_k0.8_g0.5_geo", "nopool").get("rmse", np.nan)
                   for n in Ns]
    ax.plot(Ns, rmse_pool, "o-", label="pooled RMSE")
    ax.plot(Ns, rmse_nopool, "s--", label="no-pool RMSE")
    ax.plot(Ns, widths, "^:", color="gray", label="pooled 80% CI width")
    ax.set_xlabel("roster size N"); ax.set_ylabel("streams/EUR")
    ax.set_title("H4 — pooling pays (RMSE & width vs N)"); ax.legend(); ax.grid(alpha=0.3)

    # --- (4) per-channel coverage (H2) in the GO cell -----------------------
    ax = axes[1, 1]
    go = _cell(res, "N50_rho0.6_k0.8_g0.5_geo", "pooled_geo")
    if go and "channels" in go:
        names = cfg.CHANNELS
        covs = [go["channels"][c]["coverage"] for c in names]
        accs = [go["channels"][c]["class_accuracy"] for c in names]
        x = np.arange(len(names))
        ax.bar(x - 0.2, covs, 0.4, label="80% CI coverage")
        ax.bar(x + 0.2, accs, 0.4, label="classification accuracy")
        ax.axhspan(cfg.COVERAGE_LO, cfg.COVERAGE_HI, color="green", alpha=0.1,
                   label="H2 coverage band")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_title("H2/H3 — per-channel (GO cell)"); ax.legend(); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Vega identification study — recovery curves ({tag})", fontsize=14)
    fig.tight_layout()
    path = OUTDIR / f"recovery_{tag}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
