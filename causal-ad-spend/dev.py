"""Dev/validation: DGP for ad-spend causal inference + smoke-test every estimator.
Run: python dev.py   (must print OK for each rung before we build the notebook)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# =====================================================================
# DGP : country x day panel of streams for a FLAGSHIP single campaign,
# with multi-channel paid-social spend, adstock + Hill saturation,
# a STAGGERED geo "boost" rollout, holdout controls, and KNOWN ground truth.
# =====================================================================
def adstock(x, theta):
    """Geometric adstock (carryover)."""
    out = np.empty_like(x, dtype=float); acc = 0.0
    for i, v in enumerate(x):
        acc = v + theta * acc
        out[i] = acc
    return out

def hill(x, alpha, kappa):
    """Hill saturation: x^a / (x^a + k^a) in [0,1) (diminishing returns)."""
    xa = np.power(np.maximum(x, 0), alpha)
    return xa / (xa + np.power(kappa, alpha))

def make_data(seed=7):
    rng = np.random.default_rng(seed)
    # --- 12 European markets (relative Spotify market size) ---
    geo = pd.DataFrame({
        "country": ["DE","FR","ES","IT","NL","SE","PL","BE","AT","PT","IE","DK"],
        "size":    [1.00,0.85,0.70,0.65,0.45,0.40,0.50,0.30,0.28,0.25,0.20,0.22],
    })
    C = len(geo)
    # campaign timeline: release on a Friday, observe 119 days (17 weeks)
    t0 = pd.Timestamp("2026-03-06")
    days = pd.date_range(t0, periods=119, freq="D")
    T = len(days); a = np.arange(T)

    # --- staggered geo design: 8 treated (3 cohorts), 4 holdout controls ---
    # control countries chosen across sizes so a good donor pool exists
    control = ["FR","NL","PT","DK"]
    cohorts = {28: ["DE","ES"], 42: ["IT","PL","BE"], 56: ["SE","AT","IE"]}
    start_day = {}
    for g, cs in cohorts.items():
        for c in cs: start_day[c] = g
    for c in control: start_day[c] = None     # never treated

    # --- shared structure so geos co-move (needed for DiD/SC to work) ---
    lam = np.log(2)/30.0                                   # 30-day half-life
    weekly = np.array([0.95,0.93,0.95,1.00,1.18,1.14,1.06]); weekly/=weekly.mean()
    nat_shock = np.exp(np.cumsum(rng.normal(0,0.03,T))     # common AR-ish trend
                       + rng.normal(0,0.02,T))
    base_peak = 60000.0                                    # flagship per-unit-size peak

    # --- true media-response params (GROUND TRUTH) ---
    THETA = {"tiktok":0.55, "instagram":0.45}             # adstock carryover
    VMAX  = {"tiktok":3500.0, "instagram":1800.0}         # max incremental streams/day (x size)
    ALPHA, KAPPA = 1.0, 600.0                             # Hill: kappa in EUR/day (half-saturation)

    rows = []
    for _, gr in geo.iterrows():
        c, size = gr.country, gr.size
        organic = (base_peak*size*np.exp(-lam*a)*weekly[days.weekday]*nat_shock
                   * np.exp(rng.normal(0,0.05,T)))         # idiosyncratic noise
        # ----- spend (EUR/day, modest geo-test budgets, scaled by market) -----
        sd = start_day[c]
        spend = {ch: np.zeros(T) for ch in THETA}
        if sd is not None:
            # boost: ramp over 1 wk then sustain w/ mild decay, scaled by market size
            prof = np.where(a>=sd, np.minimum((a-sd+1)/7,1)*np.exp(-(np.maximum(a-sd-7,0))/70),0)
            spend["tiktok"]    = prof * (120*size) * np.exp(rng.normal(0,0.1,T))
            spend["instagram"] = prof * (80*size)  * np.exp(rng.normal(0,0.1,T))
        # ----- incremental streams from spend (adstock -> Hill saturation) -----
        incr = np.zeros(T)
        for ch in THETA:
            eff = hill(adstock(spend[ch], THETA[ch]), ALPHA, KAPPA)   # in [0,1)
            incr += VMAX[ch]*size*eff                                  # extra streams/day
        streams = np.maximum(organic + incr, 0)
        # observed = Poisson count noise on top
        streams = rng.poisson(np.maximum(streams,0.1)).astype(float)
        for i in range(T):
            rows.append(dict(country=c, size=size, date=days[i], t=i,
                             treated=int(sd is not None), start_day=(sd if sd else -1),
                             post=int(sd is not None and i>=sd),
                             tiktok=spend["tiktok"][i], instagram=spend["instagram"][i],
                             organic=organic[i], incremental=incr[i], streams=streams[i]))
    df = pd.DataFrame(rows)
    df["spend"] = df.tiktok + df.instagram
    meta = dict(days=days, geo=geo, control=control, cohorts=cohorts,
                start_day=start_day, THETA=THETA, VMAX=VMAX, ALPHA=ALPHA, KAPPA=KAPPA)
    return df, meta

df, meta = make_data()
print("panel:", df.shape, "| countries:", df.country.nunique(), "| days:", df.t.nunique())

# ---- ground-truth estimands ----
treated_post = df[(df.treated==1) & (df.post==1)]
true_incr_total = treated_post.incremental.sum()
true_lift_pct = treated_post.incremental.sum() / treated_post.organic.sum()
true_iroas = treated_post.incremental.sum() / treated_post.spend.sum()
print(f"GROUND TRUTH: incr_total={true_incr_total:,.0f}  lift={true_lift_pct:.1%}  iROAS={true_iroas:.2f} streams/$")

results = {}

# =====================================================================
# RUNG 1 — naive OLS (streams on spend)  [expected: biased]
# =====================================================================
import statsmodels.formula.api as smf
m = smf.ols("streams ~ spend", df).fit()
coef = m.params["spend"]
ols_incr = coef*treated_post.spend.sum()
results["1 OLS (naive)"] = ols_incr/treated_post.organic.sum()
print(f"[OK] OLS coef={coef:.3f} -> implied lift {results['1 OLS (naive)']:.1%}")

# =====================================================================
# RUNG 2 — TWFE (country + day FE) on spend
# =====================================================================
import pyfixest as pf
fe = pf.feols("streams ~ spend | country + date", df)
coef2 = fe.coef()["spend"]
results["2 TWFE (FE on $)"] = coef2*treated_post.spend.sum()/treated_post.organic.sum()
print(f"[OK] TWFE coef={coef2:.3f} -> lift {results['2 TWFE (FE on $)']:.1%}")

# =====================================================================
# RUNG 3 — DiD 2x2 (first cohort vs controls, on the post dummy)
# =====================================================================
sub = df[df.country.isin(meta["cohorts"][28] + meta["control"])].copy()
sub["postX"] = ((sub.country.isin(meta["cohorts"][28])) & (sub.t>=28)).astype(int)
did = pf.feols("streams ~ postX | country + date", sub)
b = did.coef()["postX"]
# convert ATT (per country-day extra streams) to % lift vs that group's organic
g28 = df[(df.country.isin(meta["cohorts"][28])) & (df.t>=28)]
results["3 DiD 2x2"] = b*len(g28)/g28.organic.sum()
print(f"[OK] DiD 2x2 ATT/day={b:.1f} -> lift {results['3 DiD 2x2']:.1%}")

# =====================================================================
# RUNG 4 — staggered DiD (Gardner did2s, robust to staggered timing)
# =====================================================================
es = df.copy()
es["id"] = es.country.astype("category").cat.codes
es["gname"] = np.where(es.treated==1, es.start_day, 0)   # 0 = never treated
es2s = pf.event_study(es, yname="streams", idname="id", tname="t",
                      gname="gname", estimator="did2s", att=True)
s2_att = es2s.coef().iloc[0]
results["4 Staggered DiD (did2s)"] = s2_att*len(treated_post)/treated_post.organic.sum()
print(f"[OK] did2s ATT/day={s2_att:.1f} -> lift {results['4 Staggered DiD (did2s)']:.1%}")

# =====================================================================
# RUNG 5 — Synthetic Control (one treated country, pysyncon)
# =====================================================================
from pysyncon import Dataprep, Synth
focal = "DE"; adopt = meta["start_day"][focal]
donors = meta["control"]
wide = df.pivot_table(index="t", columns="country", values="streams")
sc_df = df[df.country.isin([focal]+donors)].copy()
dp = Dataprep(foo=sc_df, predictors=["streams"], predictors_op="mean",
              dependent="streams", unit_variable="country", time_variable="t",
              treatment_identifier=focal, controls_identifier=donors,
              time_predictors_prior=list(range(adopt)),
              time_optimize_ssr=list(range(adopt)))
synth = Synth(); synth.fit(dataprep=dp)
synth_path = synth._synthetic(Z0=wide[donors].loc[:])
de_post = df[(df.country==focal)&(df.t>=adopt)]
de_synth = (wide[donors] @ synth.W).loc[adopt:]
sc_att = (wide[focal].loc[adopt:] - de_synth).sum()
results["5 Synthetic Control"] = sc_att/de_post.organic.sum()
print(f"[OK] SC att_total={sc_att:,.0f} -> DE lift {results['5 Synthetic Control']:.1%} (DE true {de_post.incremental.sum()/de_post.organic.sum():.1%})")

# =====================================================================
# RUNG 6 — Synthetic DiD (Arkhangelsky et al 2021), from scratch
# =====================================================================
def synthdid_att(Y, treated_idx, t_pre):
    """Y: (units x time) numpy; treated_idx: list of treated rows;
    t_pre: # of pre-treatment columns. Returns ATT (avg over treated post)."""
    ctrl = [i for i in range(Y.shape[0]) if i not in treated_idx]
    Y0, Y1 = Y[ctrl], Y[treated_idx]
    Tpre, Tpost = t_pre, Y.shape[1]-t_pre
    # unit weights: control pre -> treated pre average (ridge-regularized simplex)
    from scipy.optimize import nnls
    ytarget = Y1[:, :Tpre].mean(0)
    A = Y0[:, :Tpre].T
    zeta = 1e-6*np.std(A)
    Aa = np.vstack([A, np.sqrt(zeta)*np.ones(A.shape[1])])
    ba = np.concatenate([ytarget, [np.sqrt(zeta)*1.0]])
    w,_ = nnls(Aa, ba); w = w/ (w.sum() if w.sum()>0 else 1)
    # time weights: pre periods -> post average across controls
    xt = Y0[:, Tpre:].mean(1)
    B = Y0[:, :Tpre]
    lt,_ = nnls(np.vstack([B, np.sqrt(zeta)*np.ones(B.shape[1])]),
                np.concatenate([xt,[np.sqrt(zeta)]])); lt = lt/(lt.sum() if lt.sum()>0 else 1)
    # weighted DiD
    treated_post = Y1[:, Tpre:].mean(0).mean()
    treated_pre  = (Y1[:, :Tpre].mean(0)*lt).sum()
    ctrl_post    = (w*Y0[:, Tpre:].mean(1)).sum()
    ctrl_pre     = (w[:,None]*Y0[:, :Tpre]).sum(0); ctrl_pre=(ctrl_pre*lt).sum()
    return (treated_post-treated_pre) - (ctrl_post-ctrl_pre)

# single treated country DE vs control donors, adoption = DE start
Y = wide[[focal]+donors].T.values          # units x time
sdid = synthdid_att(Y, [0], adopt)
results["6 Synthetic DiD"] = sdid*1/1  # ATT/day
# express as DE lift
results["6 Synthetic DiD"] = sdid*len(de_post)/de_post.organic.sum()
print(f"[OK] synthdid ATT/day={sdid:.1f} -> DE lift {results['6 Synthetic DiD']:.1%}")

# =====================================================================
# RUNG 7 — Augmented Synthetic Control (ridge-augmented; GeoLift core)
# =====================================================================
from pysyncon import AugSynth
aug = AugSynth(); aug.fit(dataprep=dp)
aug_synth = (wide[donors] @ aug.W).loc[adopt:]
aug_att = (wide[focal].loc[adopt:] - aug_synth).sum()
results["7 Augmented SCM"] = aug_att/de_post.organic.sum()
print(f"[OK] AugSCM att_total={aug_att:,.0f} -> DE lift {results['7 Augmented SCM']:.1%}")

print("\nResults so far:")
for k,v in results.items(): print(f"  {k:26s} {v:.1%}")
print(f"  {'TRUTH (all treated)':26s} {true_lift_pct:.1%}")
print(f"  {'TRUTH (DE only)':26s} {de_post.incremental.sum()/de_post.organic.sum():.1%}")
