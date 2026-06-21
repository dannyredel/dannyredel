#!/usr/bin/env python3
"""Run the Counterpoint identification study (Vega).

Examples
--------
  python run_study.py --smoke                 # 1-2 cells, M=2, ~minutes (sanity)
  python run_study.py --mode gating --M 12    # the decision-relevant grid
  python run_study.py --mode full  --M 200    # the spec's ~144-cell grid (offline)

The full grid is ~144 cells x M sims x 3 estimators of NUTS — that is an offline
job. `--mode gating` runs only the cells that drive the GO / RESCOPE / KILL
verdict (the rho-sweep with/without the geo anchor, the KILL stress cell, and an
N-sweep for the pooling claim).
"""
from __future__ import annotations

import argparse

from vega import config as cfg
from vega import plots, report, study


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["gating", "full"], default="gating")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check plumbing")
    ap.add_argument("--M", type=int, default=12, help="Monte-Carlo replicates per cell")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.smoke:
        grid = [
            cfg.Scenario(N=20, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),
            cfg.Scenario(N=20, rho=0.9, kappa=0.8, gamma=0.5, geo_anchor=False),
        ]
        M = args.M if args.M != 12 else 2
        tag = args.tag or "smoke"
    elif args.mode == "full":
        grid = cfg.full_grid()
        M, tag = args.M, args.tag or "full"
    else:
        grid = cfg.gating_grid()
        M, tag = args.M, args.tag or "gating"

    print(f"=== Vega identification study | mode={args.mode} smoke={args.smoke} "
          f"| cells={len(grid)} M={M} | weekly T={cfg.N_PERIODS} ===\n")

    results = study.run_study(grid, M=M, warmup=args.warmup, samples=args.samples,
                              chains=args.chains, tag=tag)

    if not args.smoke:
        summary = study.conclude(results)
        print("\n" + "=" * 64)
        print(f"VERDICT: {summary['verdict']}")
        print("=" * 64)
        for h, res in summary["ledger"].items():
            mark = "PASS" if res["pass"] else "FAIL"
            print(f"  [{mark}] {h:14s} {res['detail']}")
        print(f"\n  pooling RMSE reduction : {summary['pooling_rmse_reduction']}")
        print(f"  geo width reduction    : {summary['geo_width_reduction']}")
        print(f"  geo bias reduction     : {summary['geo_bias_reduction']}")

        rep = report.render(tag=tag)
        fig = plots.plot_all(tag=tag)
        print(f"\n  report -> {rep}\n  plots  -> {fig}")


if __name__ == "__main__":
    main()
