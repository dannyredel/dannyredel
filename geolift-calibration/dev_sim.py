"""Dev: geo-lift estimator CALIBRATION study (Recast-style) on our music-streaming case.
Validate that estimators give sane FPR (~5% under null) / power, and time the loop."""
import warnings; warnings.filterwarnings("ignore"); import logging; logging.disable(logging.WARNING)
import numpy as np, pandas as pd, time

# ---------------- DGP: geo panel of daily streams for one flagship single ----------------
def sim_panel(rng, G=12, Tpre=60, Tpost=30, tau=0.0, outlier=False):
    T=Tpre+Tpost; t=np.arange(T)
    base=np.exp(rng.normal(np.log(40000),0.6,G))            # log-normal geo baselines
    if outlier: base[0]*=5                                   # convex-hull violation
    decay=np.exp(-np.log(2)/45*t)                            # shared release decay
    weekly=np.array([.95,.93,.95,1,1.18,1.14,1.06]); weekly=weekly/weekly.mean()
    shared=np.cumsum(rng.normal(0,0.02,T))                   # shared trend (AR-ish)
    Y=np.zeros((G,T))
    for g in range(G):
        ar=np.zeros(T)
        for k in range(1,T): ar[k]=0.6*ar[k-1]+rng.normal(0,0.05)   # AR(1) noise
        Y[g]=base[g]*decay*weekly[t%7]*np.exp(shared+ar)
    Y[0,Tpre:]*=(1+tau)                                      # treat geo 0 post-period
    cols=[f"g{g}" for g in range(G)]
    df=pd.DataFrame(Y.T,columns=cols); df["t"]=t; df["post"]=(t>=Tpre).astype(int)
    true_lift=tau                                            # ground-truth % lift in treated geo, post
    return df, cols, Tpre, true_lift

# ---------------- estimators: each returns (lift%, ci_lo, ci_hi) ----------------
import pyfixest as pf
def est_did(df,cols,Tpre):
    long=df.melt(id_vars=["t","post"],value_vars=cols,var_name="geo",value_name="y")
    long["ly"]=np.log(long.y); long["treat"]=(long.geo=="g0").astype(int)
    long["tp"]=long.treat*long.post
    m=pf.feols("ly ~ tp | geo + t", long, vcov={"CRV1":"geo"})
    b=m.coef()["tp"]; se=m.se()["tp"]
    return np.expm1(b), np.expm1(b-1.96*se), np.expm1(b+1.96*se)

def est_tbr(df,cols,Tpre):
    y=df["g0"].values; x=df[cols[1:]].sum(1).values
    pre=slice(0,Tpre); post=slice(Tpre,None)
    A=np.vstack([np.ones(Tpre),x[pre]]).T
    coef,_,_,_=np.linalg.lstsq(A,y[pre],rcond=None)
    resid=y[pre]-A@coef; sig=resid.std(ddof=2)
    cf=coef[0]+coef[1]*x[post]; eff=y[post]-cf
    cum=eff.sum(); cf_sum=cf.sum(); n=len(eff)
    se=sig*np.sqrt(n)                                        # cumulative effect SE (deterministic TBR)
    return cum/cf_sum, (cum-1.96*se)/cf_sum, (cum+1.96*se)/cf_sum

from pysyncon import Dataprep, AugSynth
def _augsynth_gap(df,treat,donors,Tpre):
    long=df.melt(id_vars=["t"],value_vars=[treat]+donors,var_name="geo",value_name="y")
    dp=Dataprep(foo=long,predictors=["y"],predictors_op="mean",dependent="y",
        unit_variable="geo",time_variable="t",treatment_identifier=treat,
        controls_identifier=donors,time_predictors_prior=list(range(Tpre)),
        time_optimize_ssr=list(range(Tpre)))
    a=AugSynth(); a.fit(dataprep=dp)
    wide=df.set_index("t"); synth=(wide[donors]@a.W)
    gap=wide[treat]-synth
    post=gap.iloc[Tpre:].sum(); cf=synth.iloc[Tpre:].sum()
    return post/cf
def est_augscm(df,cols,Tpre):
    eff=_augsynth_gap(df,"g0",cols[1:],Tpre)
    plac=[]
    for d in cols[1:]:
        try: plac.append(_augsynth_gap(df,d,[c for c in cols if c!=d],Tpre))
        except Exception: pass
    sd=np.std(plac) if len(plac)>1 else abs(eff)
    return eff, eff-1.96*sd, eff+1.96*sd

import causalpy as cp
def est_causalpy(df,cols,Tpre,draws=300):
    cpdf=df.set_index("t")[cols]
    r=cp.SyntheticControl(cpdf,Tpre,control_units=cols[1:],treated_units=["g0"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=dict(
            draws=draws,tune=draws,chains=2,cores=1,progressbar=False,random_seed=int(df["g0"].iloc[0])%9999)))
    post=r.post_impact.sum(dim="obs_ind").stack(s=("chain","draw")).values.ravel()
    cf=r.post_pred["posterior_predictive"].mu.sum(dim="obs_ind").stack(s=("chain","draw")).values.ravel()
    lift=post/cf
    return float(np.mean(lift)), float(np.percentile(lift,2.5)), float(np.percentile(lift,97.5))

# ---------------- quick validation ----------------
if __name__=="__main__":
    ESTS={"DiD":est_did,"TBR":est_tbr,"AugSCM":est_augscm}
    for tau in [0.0,0.10]:
        print(f"\n=== tau={tau:.0%} (textbook, 30 sims) ===")
        t0=time.time(); rows=[]
        for s in range(30):
            rng=np.random.default_rng(1000+s)
            df,cols,Tpre,true=sim_panel(rng,tau=tau)
            for nm,fn in ESTS.items():
                try:
                    e,lo,hi=fn(df,cols,Tpre)
                    rows.append((nm,e,lo,hi,(lo>0 or hi<0),(lo<=true<=hi)))
                except Exception as ex: print(nm,"FAIL",repr(ex)[:80])
        R=pd.DataFrame(rows,columns=["tool","est","lo","hi","reject","cover"])
        for nm in ESTS:
            g=R[R.tool==nm]
            print(f"  {nm:8s} mean est {g.est.mean():+.1%} | reject {g.reject.mean():.0%} | coverage {g.cover.mean():.0%}")
        print(f"  ({time.time()-t0:.0f}s)")
    # one CausalPy timing
    t0=time.time(); rng=np.random.default_rng(7); df,cols,Tpre,true=sim_panel(rng,tau=0.10)
    print("\nCausalPy one fit:", est_causalpy(df,cols,Tpre), f"({time.time()-t0:.0f}s)")
    print("DONE")
