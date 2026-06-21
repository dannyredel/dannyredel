"""Render the decision-grade recovery report (study §11 deliverable #2).

Adds, vs the M=4 smoke report: H6/H7 in the ledger; the kappa=0 false-positive
result; per-cell convergence columns; the misspecified-variant comparison
(incl. pooled vs pooled+geo); the low-spend bias diagnosis + minimum-spend
floor; the completed RMSE-vs-N curve; and a "what changed vs M=4" section.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config as cfg
from . import study

OUTDIR = Path(__file__).parent / "outputs"


def _load(name):
    p = OUTDIR / name
    return json.loads(p.read_text()) if p.exists() else None


def _f(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def render(tag="verdict") -> Path:
    res = _load(f"results_{tag}.json") or {}
    verdict = _load("verdict.json") or {}
    misspec = _load("results_misspec.json")
    lowspend = _load("results_lowspend.json")
    meta = res.get("_meta", {})
    nuts = meta.get("nuts", {})

    L = []
    L.append("# Vega — Decision-Grade Identification Study\n")
    L.append(f"*M={meta.get('M')} sims/cell · NUTS {nuts.get('warmup')}+{nuts.get('samples')} "
             f"× {nuts.get('chains')} chains @ target_accept {nuts.get('target_accept')} "
             f"(max_tree_depth {nuts.get('max_tree_depth')}) · weekly T={meta.get('n_periods')}.*")
    L.append(f"*Convergence gate: keep sim iff max R-hat ≤ {meta.get('rhat_max')} and "
             f"divergences ≤ {meta.get('max_div')}. κ detection ROPE = {meta.get('rope')}. "
             f"Elapsed {meta.get('elapsed_s')}s.*\n")

    # --- verdict ------------------------------------------------------------
    L.append(f"## Verdict — correct-spec: **{verdict.get('verdict_correct_spec','—')}** · "
             f"misspec-robust: **{verdict.get('verdict_misspec_robust','—')}**\n")

    L.append("### Hypothesis ledger (H1–H8)\n")
    L.append("| Hypothesis | Result | Detail |")
    L.append("|---|---|---|")
    for h, r in verdict.get("ledger", {}).items():
        L.append(f"| {h} | {'PASS ✅' if r['pass'] else 'FAIL ❌'} | {r['detail']} |")
    L.append("")
    L.append(f"- **H4 pooling** RMSE reduction: **{verdict.get('pooling_rmse_reduction')}** (≥0.20)")
    L.append(f"- **H8 geo** width reduction {verdict.get('geo_width_reduction')}, "
             f"bias reduction {verdict.get('geo_bias_reduction')}")
    L.append(f"- **H6** collinearity breakpoint: ρ={verdict.get('collinearity_breakpoint')} "
             f"(None ⇒ H1–H3 hold through ρ=0.9)")
    L.append(f"- **H7** endogeneity bias ratio (on/off): {verdict.get('endogeneity_bias_ratio')} (≤2)\n")

    # --- kappa null (H5) ----------------------------------------------------
    L.append("### H5 — multiplier identifiability incl. the κ=0 null\n")
    L.append("| true κ | metric | value | target |")
    L.append("|---|---|---|---|")
    L.append(f"| 0.0 | false-positive rate (CI clears ROPE) | "
             f"**{verdict.get('kappa_false_positive_rate')}** | ≤ {cfg.KAPPA_FALSE_POSITIVE_MAX} |")
    L.append(f"| 0.3 | detection power | {verdict.get('kappa_power_0.3')} | high |")
    L.append(f"| 0.8 | detection power | {verdict.get('kappa_power_0.8')} | ≥ {cfg.KAPPA_EXCLUDES_ZERO_GO} |")
    L.append("")

    # --- per-cell recovery with convergence ---------------------------------
    L.append("### Per-cell recovery (with convergence)\n")
    L.append("| Scenario | Est | acc | \\|rel bias\\| | cov | width | RMSE | κ-det | "
             "n(kept/att) | %drop | maxR̂ | min ESS |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, entry in res.items():
        if label.startswith("_") or "cell" not in entry:
            continue
        for est in cfg.ESTIMATORS:
            agg = entry["cell"].get(est)
            if not agg or "class_accuracy" not in agg:
                continue
            cv = agg.get("convergence", {})
            L.append(f"| {label} | {est} | {_f(agg['class_accuracy'])} | "
                     f"{_f(agg['median_abs_rel_bias'])} | {_f(agg['coverage'])} | "
                     f"{_f(agg['mean_width'],1)} | {_f(agg['rmse'])} | "
                     f"{_f(agg['kappa_detected_frac'])} | "
                     f"{agg.get('n_sims',0)}/{cv.get('n_attempted',0)} | "
                     f"{_f(cv.get('frac_dropped',0)*100,0)}% | {_f(cv.get('max_rhat'),3)} | "
                     f"{_f(cv.get('min_ess_bulk'),0)} |")
    L.append("")

    # --- RMSE vs N (H4) -----------------------------------------------------
    ns = verdict.get("n_sweep_rmse", {})
    L.append("### H4 — pooling pays: RMSE vs roster size N (pooled)\n")
    L.append("| N | 10 | 25 | 50 | 100 |")
    L.append("|---|---|---|---|---|")
    L.append(f"| RMSE | {_f(ns.get('N10'))} | {_f(ns.get('N25'))} | "
             f"{_f(ns.get('N50'))} | {_f(ns.get('N100'))} |")
    L.append("")

    # --- misspecification battery -------------------------------------------
    if misspec:
        L.append("### Misspecified-estimator battery — GO cell (pooled vs pooled+geo)\n")
        L.append("The estimator's assumed response is deliberately wrong vs the DGP. "
                 "`geo gain` = pooled |bias| − pooled+geo |bias| (higher ⇒ the anchor "
                 "rescues more).\n")
        L.append("| spec | pooled \\|bias\\| | +geo \\|bias\\| | pooled cov | +geo cov | "
                 "pooled acc | +geo acc | geo gain |")
        L.append("|---|---|---|---|---|---|---|---|")
        ms = (verdict.get("misspec") or {}).get("rows", {})
        for spec in ["correct", "wrong_adstock", "wrong_saturation", "omit_editorial", "poisson"]:
            r = ms.get(spec)
            if not r:
                continue
            L.append(f"| {spec} | {_f(r['pooled_bias'])} | {_f(r['geo_bias'])} | "
                     f"{_f(r['pooled_cov'])} | {_f(r['geo_cov'])} | {_f(r['pooled_acc'])} | "
                     f"{_f(r['geo_acc'])} | {_f(r['geo_bias_gain'])} |")
        L.append("")
        gm = (verdict.get("misspec") or {}).get("geo_helps_more_under_misspec")
        L.append(f"> Geo anchor helps **more** under misspecification than correct-spec: "
                 f"**{gm}**. This tests whether geo's value was understated in the "
                 f"correct-spec run (where it was marginal).\n")

    # --- low-spend bias + floor ---------------------------------------------
    go = res.get(cfg.GO_LABEL, {}).get("cell", {}).get("pooled_geo")
    if go and "channels" in go:
        L.append("### Low-spend channel bias — diagnosis & minimum-spend floor\n")
        L.append("| channel | true mROAS | median rel bias | coverage | straddle | "
                 "adstocked-spend exposure |")
        L.append("|---|---|---|---|---|---|")
        for c in cfg.CHANNELS:
            ch = go["channels"][c]
            L.append(f"| {c} | {cfg.TRUE_MROAS[c]} | {ch['median_rel_bias']:+.2f} | "
                     f"{_f(ch['coverage'])} | {_f(ch['straddle_frac'])} | "
                     f"{ch['exposure']:.3g} |")
        L.append("")
    if lowspend:
        L.append("**Mitigation — global vs per-channel β-prior centre** (pooled+geo, GO cell):\n")
        L.append("| prior centre | low-spend channels: median rel bias | "
                 "spotify straddle | plugger straddle |")
        L.append("|---|---|---|---|")
        for mode in ["global", "per_channel"]:
            cellm = lowspend.get(mode, {}).get("pooled_geo", {})
            ch = cellm.get("channels", {})
            if not ch:
                continue
            lb = (ch.get("spotify", {}).get("median_rel_bias", 0)
                  + ch.get("plugger", {}).get("median_rel_bias", 0)) / 2
            L.append(f"| {mode} | {lb:+.2f} | "
                     f"{_f(ch.get('spotify',{}).get('straddle_frac'))} | "
                     f"{_f(ch.get('plugger',{}).get('straddle_frac'))} |")
        L.append("")
    floor = (_load("verdict.json") or {}).get("lowspend_floor") if _load("verdict.json") else None
    fl = study.lowspend_floor(res) if res else {}
    if fl.get("floor") is not None:
        L.append(f"> **Recommended minimum-spend floor:** report per-channel ROAS only "
                 f"for channels with cumulative adstocked-spend exposure ≥ "
                 f"**{fl['floor']:.3g}** (below this, |bias|>{fl['bias_thresh']} or "
                 f"bar-straddle>{fl['straddle_thresh']}). Feeds the §11 confidence-"
                 f"labelling rulebook.\n")

    # --- what changed vs M=4 ------------------------------------------------
    L.append("### What changed vs the M=4 smoke run\n")
    L.append(_what_changed(verdict))

    path = OUTDIR / f"report_{tag}.md"
    path.write_text("\n".join(L))
    return path


def _what_changed(verdict) -> str:
    prior = _load("verdict.json")  # current; the M=4 figures are quoted inline
    lines = [
        "- **Power:** M=4 → M≥100 with convergence gating; cell metrics now exclude "
        "non-converged sims (see %drop / max R̂ columns).",
        "- **κ=0 null added:** the M=4 run only showed the easy κ=0.8; the false-positive "
        f"rate at κ=0 is now reported (**{verdict.get('kappa_false_positive_rate')}**, "
        f"target ≤{cfg.KAPPA_FALSE_POSITIVE_MAX}), with power at κ=0.3.",
        "- **Misspecification:** the M=4 run fit a model identical to the DGP; the "
        "battery now degrades the estimator (adstock/saturation/confounder/likelihood) "
        "and compares pooled vs pooled+geo under each.",
        "- **Low-spend bias** (M=4: spotify +0.27, plugger +0.44) is confirmed/diagnosed "
        "at M≥100 and a minimum-spend floor is derived.",
        "- **N=25 crash fixed** (jax.clear_caches between fits caps the compiled-executable "
        "leak) and the RMSE-vs-N curve completed through N=100.",
        "- **H6/H7 added** to the ledger (collinearity breakpoint; endogeneity on/off).",
    ]
    return "\n".join(lines)
