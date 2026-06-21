#!/usr/bin/env python3
"""Run the Counterpoint identification study (Vega) — decision-grade.

Modes
-----
  verdict   : the rho/kappa/N/gamma verdict grid (H1-H8 + the kappa=0 null).
  misspec   : the misspecification battery (correct vs wrong specs) x (pooled vs
              pooled+geo) on the GO cell + 2 neighbours.
  lowspend  : the low-spend prior-mitigation comparison (global vs per-channel).
  all       : verdict, then misspec, then lowspend, then conclude + report.

Everything is configurable:
  python run_study.py --mode verdict  --M 100
  python run_study.py --mode misspec  --M 100 --specs correct,wrong_adstock,poisson
  python run_study.py --mode all      --M 100 --warmup 1000 --samples 1000 --chains 4
  python run_study.py --mode verdict  --M 50  --tree-depth 8   # faster, exploratory

The default NUTS is the decision-grade 1000+1000 x 4 chains @ target_accept 0.9.
"""
from __future__ import annotations

import argparse

from vega import config as cfg
from vega import study


def nuts_from_args(a):
    return dict(warmup=a.warmup, samples=a.samples, chains=a.chains,
               target_accept=a.target_accept, max_tree_depth=a.tree_depth)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["verdict", "misspec", "lowspend", "all"],
                    default="verdict")
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=cfg.NUTS_WARMUP)
    ap.add_argument("--samples", type=int, default=cfg.NUTS_SAMPLES)
    ap.add_argument("--chains", type=int, default=cfg.NUTS_CHAINS)
    ap.add_argument("--target-accept", type=float, default=cfg.NUTS_TARGET_ACCEPT)
    ap.add_argument("--tree-depth", type=int, default=cfg.NUTS_MAX_TREE_DEPTH)
    ap.add_argument("--specs", default="correct,wrong_adstock,wrong_saturation,omit_editorial,poisson")
    ap.add_argument("--cells", default=None, help="optional: subset of verdict cells by label")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    nuts = nuts_from_args(args)

    print(f"=== Vega decision-grade | mode={args.mode} M={args.M} "
          f"NUTS={nuts} ROPE={cfg.KAPPA_ROPE} gate(rhat<={cfg.RHAT_MAX},div<={cfg.MAX_DIVERGENCES}) ===\n")

    if args.mode in ("verdict", "all"):
        grid = cfg.verdict_grid()
        if args.cells:
            keep = set(args.cells.split(","))
            grid = [s for s in grid if s.label() in keep]
        study.run_study(grid, M=args.M, nuts=nuts, tag=args.tag or "verdict")

    if args.mode in ("misspec", "all"):
        specs = args.specs.split(",")
        study.run_misspec(cfg.misspec_cells(), specs, M=args.M, nuts=nuts,
                          tag="misspec")

    if args.mode in ("lowspend", "all"):
        from vega import config as c
        go = next(s for s in cfg.verdict_grid() if s.label() == c.GO_LABEL)
        study.run_lowspend(go, M=args.M, nuts=nuts, tag="lowspend")

    if args.mode == "all":
        import json
        from pathlib import Path
        out = Path("vega/outputs")
        verdict = json.loads((out / "results_verdict.json").read_text())
        misspec = json.loads((out / "results_misspec.json").read_text())
        summary = study.conclude(verdict, misspec=misspec)
        from vega import report, plots
        report.render()
        plots.plot_all()
        print(f"\nVERDICT (correct-spec): {summary['verdict_correct_spec']}")
        print(f"VERDICT (misspec-robust): {summary['verdict_misspec_robust']}")


if __name__ == "__main__":
    main()
