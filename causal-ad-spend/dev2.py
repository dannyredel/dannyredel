import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
import numpy as np, pandas as pd
from dev import make_data
df, meta = make_data()

# ---- national daily aggregate for MMM ----
nat = df.groupby("date").agg(streams=("streams","sum"), tiktok=("tiktok","sum"),
                             instagram=("instagram","sum")).reset_index()
nat["t"] = np.arange(len(nat))
true_nat_incr = df.incremental.sum()
true_nat_lift = df.incremental.sum()/df.organic.sum()
print(f"TRUE national incr={true_nat_incr:,.0f}  lift={true_nat_lift:.1%}")

# ===== RUNG 8: Bayesian MMM (pymc-marketing) =====
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
mmm = MMM(date_column="date", channel_columns=["tiktok","instagram"],
          adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(),
          control_columns=["t"], yearly_seasonality=None)
mmm.fit(nat[["date","tiktok","instagram","t"]], nat["streams"],
        draws=300, tune=300, chains=2, cores=1, progressbar=False, target_accept=0.9, random_seed=1)
contrib = mmm.compute_channel_contribution_original_scale()   # (chain,draw,date,channel)
mmm_incr = contrib.mean(dim=("chain","draw")).sum().item()
print(f"[OK] MMM total incr={mmm_incr:,.0f}  implied lift={mmm_incr/df.organic.sum():.1%}  (truth {true_nat_lift:.1%})")
# per-channel share
ch = contrib.mean(dim=("chain","draw")).sum(dim="date")
print("   per-channel incr:", {c: round(float(ch.sel(channel=c)),0) for c in ["tiktok","instagram"]})

# ===== RUNG 9: CausalPy Bayesian Synthetic Control (DE) =====
import causalpy as cp
focal="DE"; adopt=meta["start_day"][focal]; donors=meta["control"]
wide = df.pivot_table(index="t", columns="country", values="streams")
cpdf = wide[[focal]+donors].copy(); cpdf.columns = [str(c) for c in cpdf.columns]
print("causalpy attrs:", [a for a in dir(cp) if a[0].isupper()][:12])
try:
    formula = f"DE ~ 0 + " + " + ".join(donors)
    result = cp.SyntheticControl(cpdf, adopt, formula=formula,
                model=cp.pymc_models.WeightedSumFitter(sample_kwargs={"draws":300,"tune":300,"chains":2,"cores":1,"progressbar":False,"random_seed":1}))
    # post-period effect = sum of (actual - predicted)
    impact = result.post_impact.mean(dim=("chain","draw")).values
    cp_att = float(np.sum(impact))
    de_post = df[(df.country==focal)&(df.t>=adopt)]
    print(f"[OK] CausalPy SC att_total={cp_att:,.0f}  DE lift={cp_att/de_post.organic.sum():.1%}  (DE truth {de_post.incremental.sum()/de_post.organic.sum():.1%})")
except Exception as e:
    import traceback; traceback.print_exc()
    print("CAUSALPY API needs adjustment:", repr(e))
print("DONE")
