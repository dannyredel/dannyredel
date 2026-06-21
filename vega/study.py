"""Monte-Carlo harness (study §7, §8) — decision-grade.

Per sim we fit the estimator, record convergence diagnostics, gate on them, and
score only converged sims. Drivers:
  run_study      — the verdict grid (rho/kappa/N/gamma sweeps).
  run_misspec    — the misspecification battery (correct vs wrong specs) x
                   (pooled vs pooled+geo) on the GO cell + neighbours.
  run_lowspend   — the prior-mitigation comparison feeding the spend floor.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import jax
import numpy as np

from . import config as cfg
from . import dgp, geo, metrics, model

OUTDIR = Path(__file__).parent / "outputs"

DEFAULT_NUTS = dict(warmup=cfg.NUTS_WARMUP, samples=cfg.NUTS_SAMPLES,
                    chains=cfg.NUTS_CHAINS, target_accept=cfg.NUTS_TARGET_ACCEPT,
                    max_tree_depth=cfg.NUTS_MAX_TREE_DEPTH)


def _fit_one(data, truth, est, gp, nuts, spec, prior_center, prior_scale, seed):
    """One fit -> (score, diag) or (None, diag) if it errored/diverged hard."""
    mcmc, design = model.fit(
        data, est, geo_prior=gp, num_warmup=nuts["warmup"],
        num_samples=nuts["samples"], num_chains=nuts["chains"], seed=seed,
        target_accept=nuts["target_accept"], max_tree_depth=nuts["max_tree_depth"],
        spec=spec, prior_center=prior_center, prior_scale=prior_scale)
    diag = model.convergence_diagnostics(mcmc)
    roster, kappa, _ = model.mroas_posterior(mcmc, data, design)
    score = metrics.score_fit(roster, kappa, truth, data, design)
    del mcmc, roster, kappa
    return score, diag


def run_cell(scn, M, estimators, nuts=None, spec=None, prior_center="global",
             prior_scale=2.0, base_seed=0, verbose=True) -> dict:
    """Run M replicates for one cell; gate on convergence; score converged sims."""
    nuts = nuts or DEFAULT_NUTS
    scores = {e: [] for e in estimators}
    diags = {e: [] for e in estimators}
    geo_info = []
    for m in range(M):
        seed = base_seed + 1000 * m
        data, truth = dgp.simulate_roster(scn, seed=seed)
        gp = geo.geo_prior(data) if scn.geo_anchor else None
        if gp is not None:
            geo_info.append(gp)
        for est in estimators:
            if est == "pooled_geo" and not scn.geo_anchor:
                continue
            try:
                score, diag = _fit_one(data, truth, est, gp, nuts, spec,
                                       prior_center, prior_scale, seed)
                diags[est].append(diag)
                if metrics.is_converged(diag):
                    scores[est].append(score)
            except Exception as exc:
                if verbose:
                    print(f"      ! {est} sim {m} failed: {exc}")
            finally:
                # Clear per fit: each model.fit compiles a fresh NUTS executable
                # that XLA caches but does NOT reuse across fits, so without this
                # a many-fit cell accumulates compiled artifacts until the box
                # OOMs mid-compile. (Clearing only at cell boundaries crashed a
                # 60-fit cell ~halfway through.)
                jax.clear_caches()
                gc.collect()
    cell = {est: metrics.aggregate_cell(scores[est], diags[est]) for est in estimators}
    cell["_geo"] = {"n": len(geo_info),
                    "mean_experiments": float(np.mean([g["n_experiments"] for g in geo_info]))
                    if geo_info else 0.0}
    return cell


def estimators_for(scn) -> list[str]:
    """No-pool is only needed for the H4 pooling contrast (N sweep + GO)."""
    if not scn.geo_anchor:
        return ["pooled"]
    needs_nopool = scn.N in (10, 25, 100) or (scn.N == 50 and scn.rho == 0.6
                                              and scn.kappa == 0.8 and scn.gamma == 0.5)
    base = ["pooled", "pooled_geo"]
    return (["nopool"] + base) if needs_nopool else base


def run_study(grid, M=100, nuts=None, tag="verdict", estimators=None) -> dict:
    nuts = nuts or DEFAULT_NUTS
    OUTDIR.mkdir(exist_ok=True)
    t0 = time.time()
    results = {"_meta": {"M": M, "nuts": nuts, "tag": tag, "n_periods": cfg.N_PERIODS,
                         "rope": cfg.KAPPA_ROPE, "rhat_max": cfg.RHAT_MAX,
                         "max_div": cfg.MAX_DIVERGENCES, "elapsed_s": None}}
    for i, scn in enumerate(grid):
        lbl = scn.label()
        ests = estimators or estimators_for(scn)
        print(f"[{i+1}/{len(grid)}] {lbl}  (M={M}, {ests})", flush=True)
        tc = time.time()
        results[lbl] = {"scenario": scn.__dict__,
                        "cell": run_cell(scn, M, ests, nuts)}
        _print_cell(lbl, results[lbl]["cell"])
        print(f"    ({time.time()-tc:.0f}s)", flush=True)
        results["_meta"]["elapsed_s"] = round(time.time() - t0, 1)
        (OUTDIR / f"results_{tag}.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved -> {OUTDIR / f'results_{tag}.json'}")
    return results


def run_misspec(cells, specs, M=100, nuts=None, tag="misspec") -> dict:
    """Misspecification battery (Task 4): for each cell x spec, fit pooled and
    pooled+geo, so the geo anchor's value can be compared under each wrong model."""
    nuts = nuts or DEFAULT_NUTS
    OUTDIR.mkdir(exist_ok=True)
    t0 = time.time()
    results = {"_meta": {"M": M, "nuts": nuts, "tag": tag, "specs": list(specs),
                         "elapsed_s": None}}
    for scn in cells:
        for spec_name in specs:
            spec = cfg.SPECS[spec_name]
            key = f"{scn.label()}__{spec_name}"
            print(f"[{key}]  (M={M})", flush=True)
            tc = time.time()
            results[key] = {"scenario": scn.__dict__, "spec": spec_name,
                            "cell": run_cell(scn, M, ["pooled", "pooled_geo"], nuts,
                                             spec=spec)}
            _print_cell(key, results[key]["cell"])
            print(f"    ({time.time()-tc:.0f}s)", flush=True)
            results["_meta"]["elapsed_s"] = round(time.time() - t0, 1)
            (OUTDIR / f"results_{tag}.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved -> {OUTDIR / f'results_{tag}.json'}")
    return results


def run_lowspend(scn, M=100, nuts=None, tag="lowspend") -> dict:
    """Low-spend-bias mitigation comparison (Task 5): the GO cell under the
    baseline global prior vs the per-channel prior, pooled+geo."""
    nuts = nuts or DEFAULT_NUTS
    OUTDIR.mkdir(exist_ok=True)
    results = {"_meta": {"M": M, "nuts": nuts, "tag": tag}}
    for mode in ["global", "per_channel"]:
        print(f"[lowspend prior_center={mode}] (M={M})", flush=True)
        results[mode] = run_cell(scn, M, ["pooled_geo"], nuts, prior_center=mode)
        _print_cell(mode, results[mode])
        (OUTDIR / f"results_{tag}.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def _print_cell(lbl, cell):
    for est, agg in cell.items():
        if est.startswith("_") or not agg or "class_accuracy" not in agg:
            continue
        conv = agg.get("convergence", {})
        print(f"    {est:11s} acc={agg['class_accuracy']:.2f} "
              f"|relbias|={agg['median_abs_rel_bias']:.2f} cov={agg['coverage']:.2f} "
              f"width={agg['mean_width']:.1f} rmse={agg['rmse']:.2f} "
              f"kDet={agg['kappa_detected_frac']:.2f} "
              f"[n={agg.get('n_sims',0)}/{conv.get('n_attempted',0)} "
              f"drop={conv.get('frac_dropped',0):.0%} rhat={conv.get('max_rhat',float('nan')):.3f}]")


# --------------------------------------------------------------------------- #
# Low-spend minimum-spend floor (Task 5d)                                     #
# --------------------------------------------------------------------------- #
def lowspend_floor(results: dict, bias_thresh=0.25, straddle_thresh=0.5) -> dict:
    """Derive the minimum cumulative-adstocked-spend floor below which per-channel
    ROAS should not be reported. Pools per-channel (exposure, |bias|, straddle)
    points across all cells and finds the exposure above which channels are
    reliably recoverable."""
    pts = []
    for lbl, entry in results.items():
        if lbl.startswith("_") or "cell" not in entry:
            continue
        for est, agg in entry["cell"].items():
            if est.startswith("_") or not agg or "channels" not in agg:
                continue
            for name, ch in agg["channels"].items():
                pts.append(dict(cell=lbl, est=est, channel=name,
                                exposure=ch["exposure"], abs_bias=abs(ch["median_rel_bias"]),
                                straddle=ch["straddle_frac"], coverage=ch["coverage"]))
    if not pts:
        return {}
    pts.sort(key=lambda p: p["exposure"])
    expo = np.array([p["exposure"] for p in pts])
    bias = np.array([p["abs_bias"] for p in pts])
    strd = np.array([p["straddle"] for p in pts])
    # smallest exposure such that ALL channels at >= that exposure satisfy both
    # thresholds (bias acceptable AND not chronically bar-straddling)
    floor = None
    for i in range(len(expo)):
        ok = (bias[i:] <= bias_thresh) & (strd[i:] <= straddle_thresh)
        if ok.all():
            floor = float(expo[i])
            break
    return dict(points=pts, floor=floor, bias_thresh=bias_thresh,
                straddle_thresh=straddle_thresh)


# --------------------------------------------------------------------------- #
# Verdict (Tasks 6,7,8 + correct-spec vs misspec)                             #
# --------------------------------------------------------------------------- #
def conclude(results: dict, misspec: dict | None = None) -> dict:
    def cell(label, est):
        r = results.get(label)
        return r["cell"].get(est, {}) if r else {}

    GO = cfg.GO_LABEL
    go = cell(GO, "pooled_geo")
    go_pooled = cell(GO, "pooled")
    go_nogeo = cell("N50_rho0.6_k0.8_g0.5_nogeo", "pooled")
    kill = cell("N50_rho0.9_k0.8_g0.5_nogeo", "pooled")
    nopool_go = cell(GO, "nopool")

    red = (1.0 - go_pooled["rmse"] / max(nopool_go["rmse"], 1e-9)
           if nopool_go and go_pooled else 0.0)
    if go and go_nogeo:
        width_red = 1.0 - go["mean_width"] / max(go_nogeo["mean_width"], 1e-9)
        bias_red = 1.0 - go["median_abs_rel_bias"] / max(go_nogeo["median_abs_rel_bias"], 1e-9)
    else:
        width_red = bias_red = 0.0

    # H5 null — false-positive rate at kappa=0
    k0 = cell("N50_rho0.6_k0.0_g0.5_geo", "pooled_geo")
    k03 = cell("N50_rho0.6_k0.3_g0.5_geo", "pooled_geo")
    fp_rate = k0.get("kappa_detected_frac", None)
    power_03 = k03.get("kappa_detected_frac", None)

    # H6 — collinearity breakpoint (first rho where H1-H3 fail, geo on)
    breakpoint = None
    for rho in [0.3, 0.6, 0.9]:
        c = cell(f"N50_rho{rho}_k0.8_g0.5_geo", "pooled_geo")
        if not c:
            continue
        fails = (c.get("class_accuracy", 0) <= cfg.CLASS_ACCURACY_GO
                 or c.get("median_abs_rel_bias", 1) >= cfg.REL_BIAS_GO
                 or not (cfg.COVERAGE_LO <= c.get("coverage", 0) <= cfg.COVERAGE_HI))
        if fails and breakpoint is None:
            breakpoint = rho

    # H7 — endogeneity on vs off (matched cell)
    endo_off = cell("N50_rho0.6_k0.8_g0.0_geo", "pooled_geo")
    if go and endo_off:
        bias_on = go.get("median_abs_rel_bias", np.nan)
        bias_off = endo_off.get("median_abs_rel_bias", np.nan)
        endo_ratio = bias_on / max(bias_off, 1e-9)
    else:
        endo_ratio = None

    v = metrics.verdict(go, kill, red)

    ledger = {
        "H1_recovery": _pf(go.get("median_abs_rel_bias", 1) < cfg.REL_BIAS_GO,
                           f"|rel bias|={_g(go,'median_abs_rel_bias')} (<0.25)"),
        "H2_coverage": _pf(cfg.COVERAGE_LO <= go.get("coverage", 0) <= cfg.COVERAGE_HI,
                           f"coverage={_g(go,'coverage')} in [0.70,0.90]"),
        "H3_decision": _pf(go.get("class_accuracy", 0) > cfg.CLASS_ACCURACY_GO,
                           f"class acc={_g(go,'class_accuracy')} (>0.80)"),
        "H4_pooling": _pf(red >= cfg.POOLING_RMSE_REDUCTION,
                          f"RMSE reduction={red:.2f} (>=0.20)"),
        "H5_kappa": _pf(go.get("kappa_detected_frac", 0) >= cfg.KAPPA_EXCLUDES_ZERO_GO
                        and (fp_rate is None or fp_rate <= cfg.KAPPA_FALSE_POSITIVE_MAX),
                        f"power@0.8={_g(go,'kappa_detected_frac')}, "
                        f"FP@0={fp_rate}, power@0.3={power_03}"),
        "H6_collinearity": _pf(breakpoint is None or breakpoint == 0.9,
                               f"breakpoint rho={breakpoint} (None=holds through 0.9)"),
        "H7_endogeneity": _pf(endo_ratio is not None and endo_ratio <= 2.0,
                              f"bias(endo on)/bias(off)={endo_ratio if endo_ratio is None else round(endo_ratio,2)} (<=2)"),
        "H8_geo_value": _pf(width_red >= 0.15 or bias_red >= 0.15,
                            f"geo cuts width {width_red:.2f}, bias {bias_red:.2f}"),
    }

    # correct-spec vs misspec-robust verdict
    misspec_summary = _misspec_summary(misspec) if misspec else None

    summary = {
        "verdict_correct_spec": v,
        "verdict_misspec_robust": (misspec_summary or {}).get("verdict") if misspec_summary else None,
        "ledger": ledger,
        "pooling_rmse_reduction": round(red, 3),
        "geo_width_reduction": round(width_red, 3),
        "geo_bias_reduction": round(bias_red, 3),
        "kappa_false_positive_rate": fp_rate,
        "kappa_power_0.3": power_03, "kappa_power_0.8": go.get("kappa_detected_frac"),
        "collinearity_breakpoint": breakpoint,
        "endogeneity_bias_ratio": None if endo_ratio is None else round(endo_ratio, 2),
        "n_sweep_rmse": {f"N{n}": cell(f"N{n}_rho0.6_k0.8_g0.5_geo", "pooled").get("rmse")
                         for n in [10, 25, 50, 100]},
        "go_cell": go, "kill_cell": kill,
        "misspec": misspec_summary,
    }
    (OUTDIR / "verdict.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _misspec_summary(misspec: dict) -> dict:
    """For the GO cell: correct vs each wrong spec, and the geo delta under each."""
    rows = {}
    geo_helps_more = False
    go = cfg.GO_LABEL
    correct = misspec.get(f"{go}__correct", {}).get("cell", {})
    base_geo_gain = None
    if correct:
        cp = correct.get("pooled", {}); cg = correct.get("pooled_geo", {})
        if cp and cg:
            base_geo_gain = cp.get("median_abs_rel_bias", 0) - cg.get("median_abs_rel_bias", 0)
    for key, entry in misspec.items():
        if key.startswith("_") or "cell" not in entry or not key.startswith(go):
            continue
        spec = entry.get("spec")
        c = entry["cell"]
        p, g = c.get("pooled", {}), c.get("pooled_geo", {})
        if not p or not g:
            continue
        geo_gain = p.get("median_abs_rel_bias", 0) - g.get("median_abs_rel_bias", 0)
        rows[spec] = dict(
            pooled_bias=p.get("median_abs_rel_bias"), geo_bias=g.get("median_abs_rel_bias"),
            pooled_cov=p.get("coverage"), geo_cov=g.get("coverage"),
            pooled_acc=p.get("class_accuracy"), geo_acc=g.get("class_accuracy"),
            geo_bias_gain=round(geo_gain, 3))
        if spec != "correct" and base_geo_gain is not None and geo_gain > base_geo_gain:
            geo_helps_more = True
    # misspec-robust GO: every spec's pooled+geo still clears H1/H3
    robust = all(r["geo_bias"] is not None and r["geo_bias"] < cfg.REL_BIAS_GO
                 and r["geo_acc"] > cfg.CLASS_ACCURACY_GO for r in rows.values()) if rows else None
    return dict(rows=rows, geo_helps_more_under_misspec=geo_helps_more,
                verdict="GO" if robust else ("RESCOPE" if rows else None))


def _g(d, key):
    v = d.get(key)
    return f"{v:.2f}" if isinstance(v, (int, float)) else "NA"


def _pf(passed, detail):
    return {"pass": bool(passed), "detail": detail}
