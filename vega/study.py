"""Monte-Carlo harness (study §7, §8).

for scenario in grid:
    for m in 1..M:
        truth = draw_truth(scenario); data = simulate_DGP(truth, scenario)
        for estimator in {nopool, pooled, pooled_geo}:
            post = fit(estimator, data); record(recovery metrics)
aggregate -> recovery curves -> GO / RESCOPE / KILL verdict.
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


def run_cell(scn: cfg.Scenario, M: int, estimators: list[str],
             warmup: int, samples: int, chains: int, base_seed: int = 0,
             verbose: bool = True) -> dict:
    """Run M Monte-Carlo replicates for one scenario cell, all estimators."""
    scores = {e: [] for e in estimators}
    geo_info = []
    for m in range(M):
        seed = base_seed + 1000 * m
        data, truth = dgp.simulate_roster(scn, seed=seed)
        gp = geo.geo_prior(data) if scn.geo_anchor else None
        if gp is not None:
            geo_info.append(gp)
        for est in estimators:
            if est == "pooled_geo" and not scn.geo_anchor:
                continue                                  # no anchor -> no geo fit
            try:
                mcmc = model.fit(data, est, geo_prior=gp, num_warmup=warmup,
                                 num_samples=samples, num_chains=chains, seed=seed)
                roster, kappa, _ = model.mroas_posterior(mcmc, data)
                s = metrics.score_fit(roster, kappa, truth, data)
                scores[est].append(s)
                del mcmc, roster, kappa
            except Exception as exc:                      # keep the sweep alive
                if verbose:
                    print(f"      ! {est} sim {m} failed: {exc}")
            finally:
                # Each fit builds a fresh NUTS kernel -> a new compiled XLA
                # executable. Without clearing, these accumulate across the
                # Monte Carlo until the box thrashes and finally OOMs (and the
                # per-fit time creeps up). Clearing caps memory at a flat
                # baseline at the cost of one recompile per fit.
                jax.clear_caches()
                gc.collect()
    cell = {est: metrics.aggregate_cell(sc) for est, sc in scores.items()}
    cell["_geo"] = {
        "n": len(geo_info),
        "mean_experiments": float(np.mean([g["n_experiments"] for g in geo_info]))
        if geo_info else 0.0,
    }
    return cell


def estimators_for(scn: cfg.Scenario) -> list[str]:
    """Run only the estimators each cell needs (the no-pool fit is expensive and
    only required for the H4 pooling contrast: the N-sweep + the GO cell)."""
    if not scn.geo_anchor:
        return ["pooled"]                       # H8 contrast only needs pooled
    needs_nopool = scn.N in (10, 25) or (scn.N == 50 and scn.rho == 0.6)
    base = ["pooled", "pooled_geo"]
    return (["nopool"] + base) if needs_nopool else base


def run_study(grid, M=10, estimators=None, warmup=400, samples=400, chains=4,
              tag="gating") -> dict:
    OUTDIR.mkdir(exist_ok=True)
    t0 = time.time()
    # seed _meta up front so every checkpoint carries the run config (a crash
    # mid-run then still yields a self-describing results file)
    results = {"_meta": {"M": M, "estimators": estimators or "per-cell",
                         "warmup": warmup, "samples": samples, "chains": chains,
                         "tag": tag, "n_periods": cfg.N_PERIODS, "elapsed_s": None}}
    for i, scn in enumerate(grid):
        lbl = scn.label()
        ests = estimators or estimators_for(scn)
        print(f"[{i+1}/{len(grid)}] {lbl}  (M={M}, {ests})  ... ", flush=True)
        tc = time.time()
        results[lbl] = {"scenario": scn.__dict__, "cell": run_cell(
            scn, M, ests, warmup, samples, chains)}
        dt = time.time() - tc
        _print_cell(lbl, results[lbl]["cell"])
        print(f"    ({dt:.0f}s)", flush=True)
        # checkpoint after every cell so a later crash never wipes progress
        results["_meta"]["elapsed_s"] = round(time.time() - t0, 1)
        (OUTDIR / f"results_{tag}.json").write_text(
            json.dumps(results, indent=2, default=str))
    out = OUTDIR / f"results_{tag}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved -> {out}")
    return results


def _print_cell(lbl, cell):
    for est, agg in cell.items():
        if est.startswith("_") or not agg:
            continue
        print(f"    {est:11s} acc={agg['class_accuracy']:.2f} "
              f"|relbias|={agg['median_abs_rel_bias']:.2f} "
              f"cov={agg['coverage']:.2f} width={agg['mean_width']:.1f} "
              f"rmse={agg['rmse']:.2f} kExcl0={agg['kappa_excludes_zero_frac']:.2f}")


def conclude(results: dict) -> dict:
    """Derive the GO / RESCOPE / KILL verdict + the H1-H8 ledger (study §2, §8)."""
    def cell(label, est):
        r = results.get(label)
        return r["cell"].get(est, {}) if r else {}

    go = cell("N50_rho0.6_k0.8_g0.5_geo", "pooled_geo")
    go_nogeo = cell("N50_rho0.6_k0.8_g0.5_nogeo", "pooled")
    kill = cell("N50_rho0.9_k0.8_g0.5_nogeo", "pooled")
    small = cell("N10_rho0.6_k0.8_g0.5_geo", "pooled")
    big = cell("N25_rho0.6_k0.8_g0.5_geo", "pooled")
    nopool_go = cell("N50_rho0.6_k0.8_g0.5_geo", "nopool")

    # H4 — pooling RMSE reduction vs no-pool, in the GO cell
    pooled_go = cell("N50_rho0.6_k0.8_g0.5_geo", "pooled")
    if nopool_go and pooled_go:
        red = 1.0 - pooled_go["rmse"] / max(nopool_go["rmse"], 1e-9)
    else:
        red = 0.0

    # H8 — geo anchor: bias & width reduction (geo vs nogeo at rho=0.6)
    if go and go_nogeo:
        width_red = 1.0 - go["mean_width"] / max(go_nogeo["mean_width"], 1e-9)
        bias_red = 1.0 - go["median_abs_rel_bias"] / max(go_nogeo["median_abs_rel_bias"], 1e-9)
    else:
        width_red = bias_red = 0.0

    v = metrics.verdict(go, kill, red)

    ledger = {
        "H1_recovery": _pf(go.get("median_abs_rel_bias", 1) < cfg.REL_BIAS_GO,
                           f"|rel bias|={go.get('median_abs_rel_bias'):.2f} (<0.25)"),
        "H2_coverage": _pf(cfg.COVERAGE_LO <= go.get("coverage", 0) <= cfg.COVERAGE_HI,
                           f"coverage={go.get('coverage'):.2f} in [0.70,0.90]"),
        "H3_decision": _pf(go.get("class_accuracy", 0) > cfg.CLASS_ACCURACY_GO,
                           f"class acc={go.get('class_accuracy'):.2f} (>0.80)"),
        "H4_pooling": _pf(red >= cfg.POOLING_RMSE_REDUCTION,
                          f"RMSE reduction={red:.2f} (>=0.20)"),
        "H5_kappa": _pf(go.get("kappa_excludes_zero_frac", 0) >= cfg.KAPPA_EXCLUDES_ZERO_GO,
                        f"kappa excl 0 in {go.get('kappa_excludes_zero_frac'):.2f} of sims"),
        "H8_geo_value": _pf(width_red >= 0.15 or bias_red >= 0.15,
                            f"geo cuts width {width_red:.2f}, bias {bias_red:.2f}"),
    }
    summary = {
        "verdict": v,
        "ledger": ledger,
        "pooling_rmse_reduction": round(red, 3),
        "geo_width_reduction": round(width_red, 3),
        "geo_bias_reduction": round(bias_red, 3),
        "go_cell": go, "kill_cell": kill,
        "n_sweep": {"N10": small.get("rmse"), "N25": big.get("rmse"),
                    "N50": pooled_go.get("rmse")},
    }
    (OUTDIR / "verdict.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _pf(passed, detail):
    return {"pass": bool(passed), "detail": detail}
