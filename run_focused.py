#!/usr/bin/env python3
"""Focused in-session run: the highest-value decision-grade cells at reduced M.

At ~50-60s/fit (converged sampling, 4 cores) the full M>=100 grid is multi-day,
so this runs the cells that carry the new findings -- the GO cell, the kappa=0
null + kappa=0.3 power, and one misspecification variant with the pooled-vs-geo
comparison -- checkpointing each so the most critical results land first.
Reproduce at full power offline with run_study.py --M 100.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from vega import config as cfg
from vega import study

OUT = Path("vega/outputs")
NUTS = cfg.FAST_NUTS
S = lambda **k: cfg.Scenario(**k)


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str))


def main():
    t0 = time.time()
    M_go, M_k0, M_k3, M_ms = 20, 20, 15, 12
    print(f"=== focused | NUTS={NUTS} | M(go={M_go},k0={M_k0},k3={M_k3},misspec={M_ms}) ===\n",
          flush=True)

    verdict = {"_meta": {"M": f"go={M_go},k0={M_k0},k3={M_k3}", "nuts": NUTS,
                         "tag": "verdict", "n_periods": cfg.N_PERIODS,
                         "rope": cfg.KAPPA_ROPE, "rhat_max": cfg.RHAT_MAX,
                         "max_div": cfg.MAX_DIVERGENCES, "div_rate_max": cfg.DIV_RATE_MAX,
                         "note": "reduced-M in-session run; full M>=100 is offline"}}

    plan = [
        ("N50_rho0.6_k0.8_g0.5_geo", S(N=50, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),
         ["nopool", "pooled", "pooled_geo"], M_go),                       # GO (H1-H5,H8)
        ("N50_rho0.6_k0.0_g0.5_geo", S(N=50, rho=0.6, kappa=0.0, gamma=0.5, geo_anchor=True),
         ["pooled_geo"], M_k0),                                          # H5 null (FP)
        ("N50_rho0.6_k0.3_g0.5_geo", S(N=50, rho=0.6, kappa=0.3, gamma=0.5, geo_anchor=True),
         ["pooled_geo"], M_k3),                                          # H5 power
    ]
    for lbl, scn, ests, M in plan:
        print(f"[verdict {lbl}] M={M} {ests}", flush=True)
        tc = time.time()
        verdict[lbl] = {"scenario": scn.__dict__, "cell": study.run_cell(scn, M, ests, NUTS)}
        study._print_cell(lbl, verdict[lbl]["cell"])
        print(f"   ({time.time()-tc:.0f}s)", flush=True)
        verdict["_meta"]["elapsed_s"] = round(time.time() - t0, 1)
        save("results_verdict.json", verdict)

    # misspecification battery on the GO cell: correct vs wrong_adstock, pooled vs +geo
    misspec = {"_meta": {"M": M_ms, "nuts": NUTS, "tag": "misspec",
                         "specs": ["correct", "wrong_adstock"]}}
    go = S(N=50, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True)
    for spec_name in ["correct", "wrong_adstock"]:
        key = f"{cfg.GO_LABEL}__{spec_name}"
        print(f"[misspec {key}] M={M_ms}", flush=True)
        tc = time.time()
        misspec[key] = {"scenario": go.__dict__, "spec": spec_name,
                        "cell": study.run_cell(go, M_ms, ["pooled", "pooled_geo"], NUTS,
                                               spec=cfg.SPECS[spec_name])}
        study._print_cell(key, misspec[key]["cell"])
        print(f"   ({time.time()-tc:.0f}s)", flush=True)
        save("results_misspec.json", misspec)

    # finalize
    summary = study.conclude(verdict, misspec=misspec)
    from vega import report, plots
    report.render(tag="verdict")
    plots.plot_all(tag="verdict")
    print(f"\nVERDICT correct-spec={summary['verdict_correct_spec']} "
          f"misspec-robust={summary['verdict_misspec_robust']} "
          f"| total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
