"""Geo-lift module — the identification lever (study §6.1).

The MMM is under-identified under collinearity. The fix is exogenous variation:
a randomized Meta holdout zeroes one clean-geo channel in a random subset of
(release, territory) cells. We then run the §6.1 calibration loop:

  1. per-release randomized holdout on Meta (done in the DGP);
  2. analyze each experiment with a few-geo-tolerant estimator -> release-level
     lift with an interval (here: an OLS contrast, since Meta's spend is
     exogenously zeroed it is identified despite collinearity);
  3. hierarchically pool the experiment-level lifts (random-effects
     meta-analysis, DerSimonian-Laird);
  4. feed that posterior as the informative MMM prior on beta0_meta.

This "many small experiments, meta-analyzed" frame is far more robust with 5
geos than a single staggered-DiD panel.
"""
from __future__ import annotations

import numpy as np

from . import config as cfg
from .dgp import carryover

# Plug-in spillover multiplier used only to *shape the geo feature* (the true
# kappa is unknown to the experiment; this is a domain-knowledge calibration
# constant, and the prior carries an honest width to absorb the mismatch).
KAPPA_PLUGIN = 0.6


def _release_lift(data, r) -> tuple[float, float] | None:
    """OLS estimate of beta[r, meta] from one release's geo experiment.

    Regress unit-period streams on every channel's Hill response plus a baseline
    proxy. Meta's randomized zeroing supplies exogenous within-release variation,
    so its coefficient is identified even when the paid channels are collinear.
    Returns (beta_meta_hat, se) or None if Meta has no usable variation.
    """
    T, U, Cn = data["X"].shape
    rel = data["rel_of_unit"]
    sel = rel == r
    units = np.where(sel)[0]
    if units.size == 0:
        return None

    # stack this release's (territory x week) observations. Build the feature
    # exactly as beta enters the mean: paid + its share of the spillover
    # carryover, g_c = Hill_c + kappa * carryover_lam(Hill_c). Without the
    # carryover term Meta's coefficient absorbs spillover and biases high.
    lam = data["lam"]
    Xr = data["X"][:, units, :]                                     # (T, u, C)
    g = Xr + KAPPA_PLUGIN * carryover(Xr, lam)
    y = data["streams"][:, units].reshape(-1)                       # (T*u,)
    Xc = g.reshape(-1, Cn)                                          # feature matrix
    # need real exogenous variation in Meta to identify it
    if Xc[:, cfg.META].std() < 1e-8:
        return None

    # baseline proxy: fanbase * territory_share * post-launch decay
    terr = data["terr_of_unit"][units]
    weeks = np.arange(T)
    decay = np.exp(-weeks / cfg.BASELINE_TAU)
    tshare = np.array([cfg.TERRITORY_SHARE[g] for g in cfg.TERRITORIES])
    base = (data["fanbase"][r] * tshare[terr][None, :] * decay[:, None]).reshape(-1)

    # design: [baseline, intercept, channel Hill responses]
    Z = np.column_stack([base, np.ones_like(base), Xc])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ coef
    dof = max(len(y) - Z.shape[1], 1)
    sigma2 = (resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(Z.T @ Z)
    except np.linalg.LinAlgError:
        return None
    j = 2 + cfg.META                                               # meta coef position
    beta_hat = coef[j]
    se = float(np.sqrt(max(cov[j, j], 1e-12)))
    return float(beta_hat), se


def _dersimonian_laird(est, se):
    """Random-effects meta-analysis -> pooled mean and its standard error."""
    est = np.asarray(est)
    se = np.asarray(se)
    v = se ** 2
    w = 1.0 / v
    fixed = np.sum(w * est) / np.sum(w)
    Q = np.sum(w * (est - fixed) ** 2)
    df = len(est) - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_re = 1.0 / (v + tau2)
    pooled = np.sum(w_re * est) / np.sum(w_re)
    se_pooled = float(np.sqrt(1.0 / np.sum(w_re)))
    return float(pooled), se_pooled


def geo_prior(data) -> dict | None:
    """Run the calibration loop -> informative Normal prior on log beta0_meta.

    Returns {'mean': ..., 'sd': ...} on the log scale, or None if the geo anchor
    is absent / yields no usable experiments.
    """
    if not data.get("geo_anchor", False):
        return None
    held = np.where(data["held"])[0]
    ests, ses = [], []
    for r in held:
        res = _release_lift(data, r)
        if res is None:
            continue
        b, se = res
        if b > 0 and np.isfinite(se):           # positive, finite effects only
            ests.append(b)
            ses.append(se)
    if len(ests) < 2:
        return None

    pooled, se_pooled = _dersimonian_laird(ests, ses)
    if pooled <= 0:
        return None

    # delta-method transfer to the log scale (where beta0 lives)
    mean_log = float(np.log(pooled))
    # honest width: a geo-lift point estimate is informative but not dogmatic,
    # so floor the log-scale sd well above the (tiny) sampling SE.
    sd_log = float(np.clip(se_pooled / pooled, 0.25, 0.60))
    return {"mean": mean_log, "sd": sd_log, "n_experiments": len(ests),
            "pooled_beta_meta": pooled}
