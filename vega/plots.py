"""Headline recovery plots (study §7), decision-grade.

  * bias vs rho, geo on/off              -> H6, H8 (collinearity boundary)
  * classification accuracy vs rho        -> H3 (the money plot)
  * RMSE vs N (pooled vs no-pool)         -> H4 (pooling pays)
  * kappa recovery vs true kappa + null   -> H5 (incl. the false-positive test)
  * geo bias-gain under misspecification  -> Task 4 (geo understated?)
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


def _load(name):
    p = OUTDIR / name
    return json.loads(p.read_text()) if p.exists() else {}


def _cell(res, label, est):
    r = res.get(label)
    return r["cell"].get(est, {}) if r else {}


def plot_all(tag="verdict"):
    res = _load(f"results_{tag}.json")
    misspec = _load("results_misspec.json")
    rhos = [0.3, 0.6, 0.9]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) bias vs rho, geo on/off
    ax = axes[0, 0]
    for geo_on, st in [(True, "o-"), (False, "s--")]:
        g = "geo" if geo_on else "nogeo"
        est = "pooled_geo" if geo_on else "pooled"
        ys = [_cell(res, f"N50_rho{r}_k0.8_g0.5_{g}", est).get("median_abs_rel_bias", np.nan) for r in rhos]
        ax.plot(rhos, ys, st, label="geo anchor" if geo_on else "no geo")
    ax.axhline(cfg.REL_BIAS_GO, color="red", ls=":", label="H1 (0.25)")
    ax.set_xlabel("ρ"); ax.set_ylabel("median |rel bias|"); ax.set_title("H6/H8 — bias vs collinearity")
    ax.legend(); ax.grid(alpha=0.3)

    # (2) classification accuracy vs rho
    ax = axes[0, 1]
    for geo_on, st in [(True, "o-"), (False, "s--")]:
        g = "geo" if geo_on else "nogeo"
        est = "pooled_geo" if geo_on else "pooled"
        ys = [_cell(res, f"N50_rho{r}_k0.8_g0.5_{g}", est).get("class_accuracy", np.nan) for r in rhos]
        ax.plot(rhos, ys, st, label="geo anchor" if geo_on else "no geo")
    ax.axhline(cfg.CLASS_ACCURACY_GO, color="red", ls=":", label="H3 (0.80)")
    ax.set_xlabel("ρ"); ax.set_ylabel("classification accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("H3 — keep/cut accuracy"); ax.legend(); ax.grid(alpha=0.3)

    # (3) RMSE vs N
    ax = axes[0, 2]
    Ns = [10, 25, 50, 100]
    rp = [_cell(res, f"N{n}_rho0.6_k0.8_g0.5_geo", "pooled").get("rmse", np.nan) for n in Ns]
    rn = [_cell(res, f"N{n}_rho0.6_k0.8_g0.5_geo", "nopool").get("rmse", np.nan) for n in Ns]
    ax.plot(Ns, rp, "o-", label="pooled"); ax.plot(Ns, rn, "s--", label="no-pool")
    ax.set_xlabel("roster size N"); ax.set_ylabel("RMSE (streams/€)")
    ax.set_title("H4 — pooling pays"); ax.legend(); ax.grid(alpha=0.3)

    # (4) kappa recovery + null
    ax = axes[1, 0]
    ks = [0.0, 0.3, 0.8]
    med = [_cell(res, f"N50_rho0.6_k{k}_g0.5_geo", "pooled_geo").get("kappa_median", np.nan) for k in ks]
    det = [_cell(res, f"N50_rho0.6_k{k}_g0.5_geo", "pooled_geo").get("kappa_detected_frac", np.nan) for k in ks]
    ax.plot(ks, ks, "k:", label="truth")
    ax.plot(ks, med, "o-", label="posterior median κ")
    ax2 = ax.twinx(); ax2.plot(ks, det, "^--", color="purple", label="detection frac")
    ax2.axhline(cfg.KAPPA_FALSE_POSITIVE_MAX, color="red", ls=":")
    ax.set_xlabel("true κ"); ax.set_ylabel("estimated κ"); ax2.set_ylabel("frac CI clears ROPE")
    ax.set_title("H5 — κ recovery + κ=0 null"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)

    # (5) per-channel coverage/acc, GO cell
    ax = axes[1, 1]
    go = _cell(res, cfg.GO_LABEL, "pooled_geo")
    if go and "channels" in go:
        names = cfg.CHANNELS
        covs = [go["channels"][c]["coverage"] for c in names]
        accs = [go["channels"][c]["class_accuracy"] for c in names]
        x = np.arange(len(names))
        ax.bar(x - 0.2, covs, 0.4, label="coverage"); ax.bar(x + 0.2, accs, 0.4, label="class acc")
        ax.axhspan(cfg.COVERAGE_LO, cfg.COVERAGE_HI, color="green", alpha=0.1)
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.02)
        ax.set_title("H2/H3 — per-channel (GO)"); ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (6) geo bias-gain under misspecification
    ax = axes[1, 2]
    if misspec:
        specs = ["correct", "wrong_adstock", "wrong_saturation", "omit_editorial", "poisson"]
        gains = []
        for sp in specs:
            e = misspec.get(f"{cfg.GO_LABEL}__{sp}", {}).get("cell", {})
            p, g = e.get("pooled", {}), e.get("pooled_geo", {})
            gains.append((p.get("median_abs_rel_bias", np.nan) - g.get("median_abs_rel_bias", np.nan))
                         if p and g else np.nan)
        x = np.arange(len(specs))
        ax.bar(x, gains, color=["gray"] + ["steelblue"] * (len(specs) - 1))
        ax.set_xticks(x); ax.set_xticklabels(specs, rotation=30, ha="right")
        ax.set_ylabel("geo bias gain (pooled − +geo)")
        ax.set_title("Task 4 — geo value under misspec")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Vega decision-grade — recovery curves ({tag})", fontsize=15)
    fig.tight_layout()
    path = OUTDIR / f"recovery_{tag}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    return path
