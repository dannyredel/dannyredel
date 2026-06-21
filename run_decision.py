#!/usr/bin/env python3
"""Prioritized decision-grade driver.

Runs the study in strict value order with per-cell checkpointing, so under the
container's ~40-50s/fit ceiling the most decision-critical results (GO cell, the
kappa=0 null, the misspecification geo-comparison) land first even if the long
tail (full rho/N sweeps) does not finish in-session. Each phase writes its own
results_*.json; `conclude` + `report` read whatever is present.

Usage:
  python run_decision.py --Mcore 40 --Mmisspec 25 --Msweep 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vega import config as cfg
from vega import plots, report, study

OUT = Path("vega/outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Mcore", type=int, default=40)
    ap.add_argument("--Mmisspec", type=int, default=25)
    ap.add_argument("--Msweep", type=int, default=30)
    ap.add_argument("--nuts", default="fast", choices=["fast", "spec"])
    args = ap.parse_args()
    nuts = cfg.FAST_NUTS if args.nuts == "fast" else dict(
        warmup=cfg.NUTS_WARMUP, samples=cfg.NUTS_SAMPLES, chains=cfg.NUTS_CHAINS,
        target_accept=cfg.NUTS_TARGET_ACCEPT, max_tree_depth=cfg.NUTS_MAX_TREE_DEPTH)
    print(f"=== decision-grade | Mcore={args.Mcore} Mmisspec={args.Mmisspec} "
          f"Msweep={args.Msweep} | NUTS={nuts} ===\n", flush=True)

    S = lambda **k: cfg.Scenario(**k)
    # Phase 1 — the verdict-critical cells, value-ordered.
    core = [
        S(N=50, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),    # GO (H1-H5,H8)
        S(N=50, rho=0.6, kappa=0.0, gamma=0.5, geo_anchor=True),    # H5 null (FP)
        S(N=50, rho=0.6, kappa=0.3, gamma=0.5, geo_anchor=True),    # H5 power
        S(N=50, rho=0.6, kappa=0.8, gamma=0.0, geo_anchor=True),    # H7
        S(N=50, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=False),   # H8 contrast
        S(N=50, rho=0.9, kappa=0.8, gamma=0.5, geo_anchor=False),   # KILL
        S(N=50, rho=0.9, kappa=0.8, gamma=0.5, geo_anchor=True),    # H6 high-rho
    ]
    study.run_study(core, M=args.Mcore, nuts=nuts, tag="verdict")

    # Phase 2 — misspecification battery on the GO cell + one neighbour.
    study.run_misspec(
        [S(N=50, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),
         S(N=50, rho=0.9, kappa=0.8, gamma=0.5, geo_anchor=True)],
        ["correct", "wrong_adstock", "wrong_saturation", "omit_editorial", "poisson"],
        M=args.Mmisspec, nuts=nuts, tag="misspec")

    # Phase 3 — low-spend prior mitigation (global vs per-channel).
    study.run_lowspend(S(N=50, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),
                       M=args.Mcore, nuts=nuts, tag="lowspend")

    # Phase 4 — the rest of the rho/N sweeps (lower priority, may not finish).
    sweep = [
        S(N=50, rho=0.3, kappa=0.8, gamma=0.5, geo_anchor=True),
        S(N=50, rho=0.3, kappa=0.8, gamma=0.5, geo_anchor=False),
        S(N=10, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),
        S(N=25, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),
        S(N=100, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),
    ]
    # append to the same verdict file
    existing = json.loads((OUT / "results_verdict.json").read_text())
    more = study.run_study(sweep, M=args.Msweep, nuts=nuts, tag="verdict_sweep")
    existing.update({k: v for k, v in more.items() if not k.startswith("_")})
    (OUT / "results_verdict.json").write_text(json.dumps(existing, indent=2, default=str))

    finalize()


def finalize():
    verdict = json.loads((OUT / "results_verdict.json").read_text())
    misspec = json.loads((OUT / "results_misspec.json").read_text()) \
        if (OUT / "results_misspec.json").exists() else None
    summary = study.conclude(verdict, misspec=misspec)
    report.render(tag="verdict")
    plots.plot_all(tag="verdict")
    print(f"\nVERDICT correct-spec={summary['verdict_correct_spec']} "
          f"misspec-robust={summary['verdict_misspec_robust']}", flush=True)


if __name__ == "__main__":
    main()
