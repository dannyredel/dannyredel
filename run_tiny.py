import time, json, numpy as np
from vega import study, config as cfg
S=lambda **k: cfg.Scenario(**k)
NUTS=dict(warmup=600,samples=600,chains=4,target_accept=0.88,max_tree_depth=7)
t0=time.time(); out={}
plan=[("GO_pooled_geo",S(N=50,rho=0.6,kappa=0.8,gamma=0.5,geo_anchor=True),["pooled_geo"],6),
      ("k0_pooled_geo",S(N=50,rho=0.6,kappa=0.0,gamma=0.5,geo_anchor=True),["pooled_geo"],6)]
for lbl,scn,ests,M in plan:
    tc=time.time(); print(f"[{lbl}] M={M}",flush=True)
    cell=study.run_cell(scn,M,ests,NUTS); out[lbl]=cell
    a=cell.get("pooled_geo",{}); cv=a.get("convergence",{})
    print(f"  -> n={a.get('n_sims')} kDet={a.get('kappa_detected_frac')} "
          f"acc={a.get('class_accuracy')} relbias={a.get('median_abs_rel_bias')} "
          f"cov={a.get('coverage')} kept={cv.get('kept_frac')} maxR={cv.get('max_rhat')} "
          f"({time.time()-tc:.0f}s)",flush=True)
    json.dump(out,open("vega/outputs/results_tiny.json","w"),indent=1,default=float)
print(f"TOTAL {time.time()-t0:.0f}s",flush=True)
