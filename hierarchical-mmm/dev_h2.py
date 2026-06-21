"""Validate the shrinkage thesis: partial pooling beats no-pooling & complete pooling
at recovering per-release ROAS. Uses nutpie for speed."""
import warnings; warnings.filterwarnings("ignore"); import logging; logging.disable(logging.WARNING)
import numpy as np, pandas as pd, pytensor.tensor as pt, pymc as pm, time
from dev_h import make_data
df,truth,meta=make_data()
R,W=meta["R"],meta["W"]; Lmax=8; chans=meta["channels"]
def lagmat(col):
    M=np.zeros((len(df),Lmax+1))
    for rid,g in df.groupby("rid"):
        idx=g.index.values; x=g[col].values/1000.0
        for l in range(Lmax+1): M[idx[l:],l]=x[:len(x)-l] if l>0 else x
    return M
LAGS={c:lagmat(c) for c in chans}
rid=df.rid.values; tier_z=df.groupby("rid").tier_z.first().values
y=df.streams.values; wk=df.week.values
spend_by_rel=df.groupby("rid")[chans].sum().sum(1).values

def build(pooling):
    with pm.Model() as m:
        d_=pm.HalfNormal("d",0.3)
        logB=pm.Normal("logB",11,3,shape=R); B=pt.exp(logB)
        base=B[rid]*pt.exp(-d_*wk)
        if pooling=="complete":
            logtheta=pm.Normal("mu_theta",0,0.5)*pt.ones(R)
        elif pooling=="none":
            logtheta=pm.Normal("logtheta",0,1.0,shape=R)         # independent, no sharing
        else:  # partial
            mu_t=pm.Normal("mu_theta",0,0.5); gam=pm.Normal("gamma",0,0.5)
            sig_t=pm.HalfNormal("sig_theta",0.5); z=pm.Normal("z",0,1,shape=R)
            logtheta=mu_t+gam*tier_z+sig_t*z
        theta=pt.exp(logtheta)
        media=0.0
        for c in chans:
            a=pm.Beta(f"alpha_{c}",2,4); lam=pm.HalfNormal(f"lam_{c}",1.0); A=pm.HalfNormal(f"A_{c}",8000)
            ad=pt.dot(LAGS[c], a**np.arange(Lmax+1))
            media=media+A*theta[rid]*(1-pt.exp(-lam*ad))/(1+pt.exp(-lam*ad))
        pm.Deterministic("media", media)
        pm.Normal("obs", base+media, pm.HalfNormal("sigma",0.05*y.mean()), observed=y)
        t0=time.time()
        idata=pm.sample(300,tune=300,chains=2,cores=1,nuts_sampler="nutpie",
                        progressbar=False,random_seed=1)
        print(f"  [{pooling}] sampled in {time.time()-t0:.0f}s")
    return idata

def rel_roas(idata):
    med=idata.posterior["media"].mean(("chain","draw")).values
    s=pd.Series(med,index=df.index); tot=s.groupby(df.rid).sum().values
    return tot/spend_by_rel

print(f"DGP: avg lift {truth.lift.mean():.1%}, median ROAS {truth.roas.median():.1f}")
res={}
for p in ["none","complete","partial"]:
    id_=build(p); est=rel_roas(id_)
    rmse=np.sqrt(np.mean((est-truth.roas.values)**2))
    res[p]=(est,rmse,id_)
    print(f"  [{p}] per-release ROAS RMSE vs truth = {rmse:.2f}  corr={np.corrcoef(est,truth.roas)[0,1]:.3f}")
print("\nSHRINKAGE THESIS:", "PASS" if res['partial'][1]<res['none'][1] and res['partial'][1]<res['complete'][1] else "CHECK")
ip=res['partial'][2]
print("partial recovered gamma", float(ip.posterior['gamma'].mean()).__round__(2), "true", meta['gamma'],
      "| sig_theta", float(ip.posterior['sig_theta'].mean()).__round__(2), "true", meta['sig_theta'])
print("DONE")
