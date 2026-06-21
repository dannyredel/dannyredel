"""Dev: hierarchical partial-pooling Bayesian MMM across 40 releases.
Validate DGP calibration + that the hierarchical model recovers per-release ROAS.
"""
import warnings; warnings.filterwarnings("ignore"); import logging; logging.disable(logging.WARNING)
import numpy as np, pandas as pd, pytensor.tensor as pt, pymc as pm

# ---------------- DGP ----------------
def adstock_np(x, a, lmax=8):
    out = np.zeros_like(x, float);
    for l in range(lmax+1): out[l:] += (a**l)*x[:len(x)-l] if l>0 else x.copy()*1.0
    # simpler exact recursion:
    out = np.zeros_like(x, float); acc=0.0
    for i,v in enumerate(x): acc=v+a*acc; out[i]=acc
    return out
def logsat(x, lam): return (1-np.exp(-lam*x))/(1+np.exp(-lam*x))

def make_data(seed=11):
    rng = np.random.default_rng(seed)
    roster = pd.DataFrame([
        ("Aurora Vance","Flagship",250000,4),("Mans Eklund","Mid",70000,5),
        ("Lena Brandt","Mid",60000,4),("Theo Mercier","Mid",48000,4),
        ("Niamh OConnor","Emerging",16000,4),("Dario Conti","Emerging",13000,4),
        ("Sofia Almeida","Emerging",10000,4),("Kai Lindqvist","Emerging",8000,4),
        ("Emile Rousseau","Developing",3000,4),("Petra Novak","Developing",2000,3)],
        columns=["artist","tier","peak","n"])
    rel=[]
    for _,r in roster.iterrows():
        for _ in range(r.n): rel.append((r.artist,r.tier,r.peak))
    rel=pd.DataFrame(rel,columns=["artist","tier","peak"]); rel["rid"]=range(len(rel))
    R=len(rel); W=16; channels=["tiktok","instagram"]
    # true hierarchical params
    d=0.18                                   # weekly organic decay
    ALPHA={"tiktok":0.5,"instagram":0.4}     # weekly adstock
    LAM={"tiktok":0.8,"instagram":0.9}       # logistic saturation (spend in k EUR)
    A={"tiktok":9000.0,"instagram":5000.0}   # channel max weekly effect (x theta)
    tier_z=(np.log(rel.peak)-np.log(rel.peak).mean())/np.log(rel.peak).std()
    gamma=0.5; sig_theta=0.35
    u=rng.normal(0,sig_theta,R)
    theta=np.exp(gamma*tier_z.values+u)      # release responsiveness (mean~1)
    rows=[]; truth=[]
    for i in range(R):
        peak=rel.peak[i]; wk=np.arange(W)
        base=peak*np.exp(-d*wk)*np.exp(rng.normal(0,0.05,W))
        budget=peak*0.04*np.exp(rng.normal(0,0.3))      # total EUR
        wprof=np.exp(-0.35*wk); wprof/=wprof.sum()       # front-loaded
        incr=np.zeros(W); spend={}
        for c in channels:
            share={"tiktok":0.6,"instagram":0.4}[c]
            sp=budget*share*wprof*np.exp(rng.normal(0,0.1,W))
            spend[c]=sp
            eff=A[c]*theta[i]*logsat(adstock_np(sp/1000.0,ALPHA[c]),LAM[c])  # spend in kEUR
            incr+=eff
        y=base+incr+rng.normal(0,0.03*base.mean(),W)
        tot_sp=sum(spend[c].sum() for c in channels)
        truth.append(dict(rid=i,tier=rel.tier[i],theta=theta[i],
                          roas=incr.sum()/tot_sp, lift=incr.sum()/base.sum(),
                          spend=tot_sp, incr=incr.sum()))
        for w in range(W):
            rows.append(dict(rid=i,tier=rel.tier[i],peak=peak,week=w,
                tier_z=tier_z.values[i], streams=max(y[w],1.0),
                tiktok=spend["tiktok"][w], instagram=spend["instagram"][w]))
    df=pd.DataFrame(rows); truth=pd.DataFrame(truth)
    meta=dict(R=R,W=W,channels=channels,ALPHA=ALPHA,LAM=LAM,A=A,d=d,gamma=gamma,
              sig_theta=sig_theta, roster=roster, rel=rel)
    return df, truth, meta

df,truth,meta=make_data()
print("panel",df.shape,"releases",meta["R"])
print(f"avg lift {truth.lift.mean():.1%} | median ROAS {truth.roas.median():.1f} streams/EUR | total spend EUR {truth.spend.sum():,.0f}")
print(truth.groupby("tier").agg(theta=("theta","mean"),roas=("roas","mean"),lift=("lift","mean")).round(2))

# ---------------- precompute adstock lag matrices (per release) ----------------
R,W=meta["R"],meta["W"]; Lmax=8
def lagmat(col):
    M=np.zeros((len(df),Lmax+1))
    for rid,g in df.groupby("rid"):
        idx=g.index.values; x=(g[col].values)/1000.0
        for l in range(Lmax+1):
            M[idx[l:],l]=x[:len(x)-l] if l>0 else x
    return M
LAGS={c:lagmat(c) for c in meta["channels"]}
rid_idx=df.rid.values; tier_z=df.groupby("rid").tier_z.first().values
y=df.streams.values; wk=df.week.values

# ---------------- hierarchical (partial pooling) MMM ----------------
import time; t0=time.time()
with pm.Model() as hmodel:
    d_=pm.HalfNormal("d",0.3)
    logB=pm.Normal("logB",np.log(df.peak.values[::1].mean()),2,shape=R)  # rough
    B=pm.Deterministic("B",pt.exp(logB))
    base=B[rid_idx]*pt.exp(-d_*wk)
    # hierarchical theta
    mu_t=pm.Normal("mu_theta",0,0.5); gam=pm.Normal("gamma",0,0.5); sig_t=pm.HalfNormal("sig_theta",0.5)
    z=pm.Normal("z",0,1,shape=R); logtheta=mu_t+gam*tier_z+sig_t*z
    theta=pm.Deterministic("theta",pt.exp(logtheta))
    media=0.0
    for c in meta["channels"]:
        a=pm.Beta(f"alpha_{c}",2,2); lam=pm.HalfNormal(f"lam_{c}",1.0); A=pm.HalfNormal(f"A_{c}",8000)
        w=a**np.arange(Lmax+1)
        ad=pt.dot(LAGS[c],w)
        sat=(1-pt.exp(-lam*ad))/(1+pt.exp(-lam*ad))
        media=media+A*theta[rid_idx]*sat
    mu=base+media
    sig=pm.HalfNormal("sigma",0.05*y.mean())
    pm.Normal("obs",mu,sig,observed=y)
    idata=pm.sample(300,tune=300,chains=2,cores=1,progressbar=False,target_accept=0.9,random_seed=1)
print(f"sampled in {time.time()-t0:.0f}s")

# recover per-release ROAS estimate
th=idata.posterior["theta"].mean(("chain","draw")).values
# estimated ROAS proportional to theta * channel response / spend -> compare ranking/scale via theta corr
import numpy as np
print("theta corr (est vs true):", np.corrcoef(th, truth.theta.values)[0,1].round(3))
print("recovered gamma:", float(idata.posterior["gamma"].mean()).__round__(2), "true", meta["gamma"])
print("recovered sig_theta:", float(idata.posterior["sig_theta"].mean()).__round__(2), "true", meta["sig_theta"])
for c in meta["channels"]:
    print(f"  alpha_{c} est {float(idata.posterior['alpha_'+c].mean()):.2f} true {meta['ALPHA'][c]}")
print("DONE")
