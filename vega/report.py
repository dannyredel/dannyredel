"""Render the recovery report (study §11 deliverable #2).

Turns the saved results + verdict into a human-readable markdown summary: the
H1-H8 ledger, the per-cell recovery table, and the GO/RESCOPE/KILL call.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config as cfg

OUTDIR = Path(__file__).parent / "outputs"


def render(tag="gating") -> Path:
    res = json.loads((OUTDIR / f"results_{tag}.json").read_text())
    verdict = json.loads((OUTDIR / "verdict.json").read_text())
    meta = res.get("_meta", {})

    L = []
    L.append("# Vega — Identification Study Recovery Report\n")
    L.append(f"*Generated from `results_{tag}.json`. "
             f"M={meta.get('M')} sims/cell, weekly T={meta.get('n_periods')}, "
             f"{meta.get('warmup')}+{meta.get('samples')} NUTS draws x "
             f"{meta.get('chains')} chains. Elapsed {meta.get('elapsed_s')}s.*\n")

    L.append(f"## Verdict: **{verdict['verdict']}**\n")
    L.append("Decision rules per study §8 (GO needs class acc >0.80, "
             "|rel bias| <0.25, coverage in [0.70,0.90], kappa-CI excludes 0, "
             "and pooling to beat no-pool).\n")

    L.append("### Hypothesis ledger (H1-H8)\n")
    L.append("| Hypothesis | Result | Detail |")
    L.append("|---|---|---|")
    for h, r in verdict["ledger"].items():
        mark = "PASS ✅" if r["pass"] else "FAIL ❌"
        L.append(f"| {h} | {mark} | {r['detail']} |")
    L.append("")
    L.append(f"- Pooling RMSE reduction (H4): **{verdict['pooling_rmse_reduction']}** "
             f"(threshold ≥0.20)")
    L.append(f"- Geo anchor width reduction (H8): **{verdict['geo_width_reduction']}**")
    L.append(f"- Geo anchor bias reduction (H8): **{verdict['geo_bias_reduction']}**")
    ns = verdict.get("n_sweep", {})
    def _r(x):
        return round(x, 2) if isinstance(x, (int, float)) else x
    L.append(f"- RMSE vs roster size (H4): N10={_r(ns.get('N10'))}, "
             f"N25={_r(ns.get('N25'))}, N50={_r(ns.get('N50'))}\n")

    # --- per-cell recovery table -------------------------------------------
    L.append("### Per-cell recovery\n")
    L.append("| Scenario | Estimator | class acc | \\|rel bias\\| | coverage | "
             "80% width | RMSE | kappa excl 0 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for label, entry in res.items():
        if label.startswith("_"):
            continue
        for est in cfg.ESTIMATORS:
            agg = entry["cell"].get(est)
            if not agg:
                continue
            L.append(f"| {label} | {est} | {agg['class_accuracy']:.2f} | "
                     f"{agg['median_abs_rel_bias']:.2f} | {agg['coverage']:.2f} | "
                     f"{agg['mean_width']:.1f} | {agg['rmse']:.2f} | "
                     f"{agg['kappa_excludes_zero_frac']:.2f} |")
    L.append("")

    # --- per-channel detail for the GO cell (the H3 money detail) ----------
    go = res.get("N50_rho0.6_k0.8_g0.5_geo", {}).get("cell", {}).get("pooled_geo")
    if go and "channels" in go:
        L.append("### Per-channel recovery — GO cell "
                 "(N=50, rho=0.6, kappa=0.8, geo on, pooled+geo)\n")
        L.append("| Channel | true mROAS | median rel bias | coverage | "
                 "class acc | straddle bar | above bar? |")
        L.append("|---|---|---|---|---|---|---|")
        for c in cfg.CHANNELS:
            ch = go["channels"][c]
            above = "yes" if cfg.TRUE_MROAS[c] > cfg.BREAK_EVEN_BAR else "no"
            L.append(f"| {c} | {cfg.TRUE_MROAS[c]} | {ch['median_rel_bias']:+.2f} | "
                     f"{ch['coverage']:.2f} | {ch['class_accuracy']:.2f} | "
                     f"{ch['straddle_frac']:.2f} | {above} |")
        L.append("")
        L.append(f"> Break-even bar = {cfg.BREAK_EVEN_BAR} streams/€. "
                 "Spotify (coarse-geo, low-spend) is expected to be the "
                 "chronically hard-to-classify channel (study §9).\n")

    path = OUTDIR / f"report_{tag}.md"
    path.write_text("\n".join(L))
    return path
