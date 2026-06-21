"""The estimator under test — NumPyro hierarchical NB MMM (study §6).

Structurally identical to the DGP in the *correct-spec* configuration (so any
failure is identification, not misspecification). We estimate what the product
claim depends on:

  * beta[r, c]  — channel effects, non-centered partial pooling (H4);
  * kappa       — the organic->paid multiplier (H5);
  * baseline    — organic level, decay, (optionally) editorial control;
  * phi         — Negative-Binomial overdispersion.

Estimators (pooling axis):
  nopool      — independent diffuse beta per release;
  pooled      — hierarchical partial pooling, generic prior;
  pooled_geo  — hierarchical + informative geo prior on beta0_meta (H8).

Specifications (misspecification axis, §6): the response design the estimator
*assumes* can be deliberately wrong vs the DGP — wrong adstock retention, wrong
saturation (linear/log instead of Hill), an omitted editorial confounder, or a
Poisson likelihood against NB data. `build_design` produces the design matrix
and the matching marginal-ROAS chain-rule constant for whatever spec is asked.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from . import config as cfg
from .dgp import geometric_adstock, hill, hill_grad

# Run chains truly in parallel across the box's cores (2-4 chains ~ cost of 1).
numpyro.set_host_device_count(4)

# Correct-spec: the estimator's assumed response == the DGP's.
CORRECT_SPEC = {"name": "correct", "adstock": "correct", "saturation": "hill",
                "editorial": True, "likelihood": "nb"}


# --------------------------------------------------------------------------- #
# Design construction (the misspecification axis)                             #
# --------------------------------------------------------------------------- #
def build_design(data, spec=None) -> dict:
    """Build the estimator's design matrix X and its marginal-ROAS constant.

    mROAS[r,c] = (1 + kappa_eff) * beta[r,c] * mroas_const[r,c], where
    mroas_const = (d response / d adstock at the operating point) / (1 - theta).
    For `correct` this reproduces the DGP's Cmat exactly; for a misspecified
    response it is the *estimator's own* marginal effect, which is what a bias
    measurement against the true mROAS should use.
    """
    spec = spec or CORRECT_SPEC
    spend = np.asarray(data["spend_u"])                 # (T, U, C)
    T, U, Cn = spend.shape
    N, Gn = data["N"], data["G"]
    k, s, true_theta = data["k"], data["s"], data["theta"]

    # --- adstock retention --------------------------------------------------
    if spec.get("adstock", "correct") == "wrong":
        theta = np.full(Cn, 0.5)                         # single shared (no per-channel)
    else:
        theta = true_theta
    adstock = geometric_adstock(spend, theta)            # (T, U, C)
    abar = adstock.reshape(T, N, Gn, Cn).mean(axis=(0, 2))   # (N, C) operating point

    # --- saturation ---------------------------------------------------------
    sat = spec.get("saturation", "hill")
    if sat == "hill":
        X = hill(adstock, k[None, None, :], s[None, None, :])
        dresp = hill_grad(abar, k[None, :], s[None, :])  # (N, C)
    elif sat == "linear":
        X = adstock / k[None, None, :]                   # scale to O(1)
        dresp = np.ones((N, Cn)) / k[None, :]
    elif sat == "log":
        X = np.log1p(adstock / k[None, None, :])
        dresp = 1.0 / ((k[None, :] + abar) )             # d/da log1p(a/k) = 1/(k+a)
    else:
        raise ValueError(sat)

    mroas_const = dresp / (1.0 - theta[None, :])         # (N, C)
    return dict(
        X=np.asarray(X), mroas_const=np.asarray(mroas_const),
        editorial_on=bool(spec.get("editorial", True)),
        likelihood=spec.get("likelihood", "nb"),
        spec_name=spec.get("name", "correct"),
    )


def _carryover_scan(paid, lam):
    """stock[t] = paid[t] + lam * stock[t-1], scanned over time. paid: (T, U)."""
    def step(prev, p):
        new = p + lam * prev
        return new, new
    _, out = jax.lax.scan(step, jnp.zeros(paid.shape[1]), paid)
    return out


def _beta_centers(data, X, mode="global"):
    """Prior centre(s) for log beta on the log scale.

    global      — one channel-symmetric centre (mean NNLS effect); never leaks
                  which channel is strong, but pulls weak low-spend channels up.
    per_channel — each channel centred on its own pooled NNLS coefficient; the
                  low-spend-bias mitigation (lets a weak channel be small a priori).
    """
    T, U, Cn = X.shape
    y = np.asarray(data["streams"]).reshape(-1)
    Xc = np.asarray(X).reshape(-1, Cn)
    weeks = np.arange(T)
    decay = np.exp(-weeks / cfg.BASELINE_TAU)
    tshare = np.array([cfg.TERRITORY_SHARE[g] for g in cfg.TERRITORIES])
    base = (data["fanbase"][data["rel_of_unit"]][None, :]
            * tshare[data["terr_of_unit"]][None, :] * decay[:, None]).reshape(-1)
    Z = np.column_stack([base, np.ones_like(base), Xc])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    chan = np.clip(coef[2:], 1e-3, None)
    if mode == "per_channel":
        return np.log(chan)
    return np.full(Cn, float(np.log(np.maximum(np.mean(chan), 1.0))))


def mmm_model(X, streams, rel_of_unit, terr_of_unit, zfan, editorial,
              tshare, beta0_loc, beta0_scale, a0_loc, lam, n_releases, n_terr,
              hier=True, editorial_on=True, likelihood="nb"):
    """Negative-Binomial (or Poisson) hierarchical MMM. Arrays are JAX-ready."""
    T, U, Cn = X.shape

    # --- channel effects ----------------------------------------------------
    if hier:
        beta0 = numpyro.sample("beta0", dist.Normal(beta0_loc, beta0_scale))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.7 * jnp.ones(Cn)))
        with numpyro.plate("rel", n_releases, dim=-2):
            with numpyro.plate("chan", Cn, dim=-1):
                u = numpyro.sample("u", dist.Normal(0.0, 1.0))      # non-centered
        beta = numpyro.deterministic("beta", jnp.exp(beta0 + sigma * u))
    else:
        with numpyro.plate("rel", n_releases, dim=-2):
            with numpyro.plate("chan", Cn, dim=-1):
                logb = numpyro.sample("logb", dist.Normal(beta0_loc, 2.0))
        beta = numpyro.deterministic("beta", jnp.exp(logb))

    # --- baseline (organic) -------------------------------------------------
    # a0 is centred on log(mean streams) (passed as data); centring it on the
    # data scale rather than 0 removes a ~3-sigma stretch that otherwise funnels
    # the geometry, diverges, and saturates the tree depth.
    a0 = numpyro.sample("a0", dist.Normal(a0_loc, 1.0))
    a1 = numpyro.sample("a1", dist.Normal(0.5, 0.5))
    tau = numpyro.sample("tau", dist.LogNormal(jnp.log(3.0), 0.4))
    weeks = jnp.arange(T)
    decay = jnp.exp(-weeks / tau)
    base_level = jnp.exp(a0 + a1 * zfan)
    base_unit = base_level[rel_of_unit] * tshare[terr_of_unit] * n_terr
    baseline = base_unit[None, :] * decay[:, None]
    if editorial_on:                                  # omitted-confounder variant drops this
        ed_coef = numpyro.sample("ed_coef", dist.HalfNormal(1.0))
        baseline = baseline * (1.0 + ed_coef * editorial)

    # --- paid + spillover ---------------------------------------------------
    beta_unit = beta[rel_of_unit]
    paid = jnp.einsum("tuc,uc->tu", X, beta_unit)
    paid_stock = _carryover_scan(paid, lam)
    kappa = numpyro.sample("kappa", dist.HalfNormal(0.5))
    mu = jnp.clip(baseline + paid + kappa * paid_stock, 1e-3, None)

    # --- likelihood ---------------------------------------------------------
    if likelihood == "poisson":
        numpyro.sample("streams", dist.Poisson(mu), obs=streams)
    else:
        phi = numpyro.sample("phi", dist.LogNormal(jnp.log(10.0), 0.5))
        numpyro.sample("streams", dist.NegativeBinomial2(mu, phi), obs=streams)


def fit(data, estimator, geo_prior=None, num_warmup=1000, num_samples=1000,
        num_chains=4, seed=0, target_accept=0.9, max_tree_depth=10,
        spec=None, prior_center="global", prior_scale=2.0,
        chain_method=os.environ.get("VEGA_CHAIN_METHOD", "parallel")):
    """Run NUTS for one estimator/spec on one simulated roster.

    Returns (mcmc, design). `design` carries the estimator's X and mROAS
    constant (which differ from the DGP under misspecification).
    """
    hier = estimator != "nopool"
    use_geo = (estimator == "pooled_geo") and (geo_prior is not None)
    gp = geo_prior if use_geo else None
    design = build_design(data, spec)
    X = design["X"]

    centers = _beta_centers(data, X, mode=prior_center)      # (C,)
    zfan = (np.log(data["fanbase"]) - np.log(data["fanbase"]).mean())
    zfan = zfan / (zfan.std() + 1e-9)

    streams = np.asarray(data["streams"])
    terr = data["terr_of_unit"]
    by_terr = np.array([streams[:, terr == g].sum() for g in range(data["G"])])
    tshare = by_terr / by_terr.sum()

    Cn = data["C"]
    beta0_loc = centers.astype(float).copy()
    beta0_scale = np.full(Cn, float(prior_scale))
    if gp is not None:
        beta0_loc[cfg.META] = gp["mean"]
        beta0_scale[cfg.META] = gp["sd"]
    # baseline-level prior centre on the data scale (see mmm_model)
    a0_loc = float(np.log(max(np.mean(np.asarray(data["streams"])), 1.0)))

    args = dict(
        X=jnp.asarray(X),
        streams=jnp.asarray(data["streams"]),
        rel_of_unit=jnp.asarray(data["rel_of_unit"]),
        terr_of_unit=jnp.asarray(data["terr_of_unit"]),
        zfan=jnp.asarray(zfan),
        editorial=jnp.asarray(data["editorial"]),
        tshare=jnp.asarray(tshare),
        beta0_loc=jnp.asarray(beta0_loc),
        beta0_scale=jnp.asarray(beta0_scale),
        a0_loc=a0_loc,
        lam=data["lam"], n_releases=data["N"], n_terr=data["G"], hier=hier,
        editorial_on=design["editorial_on"], likelihood=design["likelihood"],
    )

    kernel = NUTS(mmm_model, target_accept_prob=target_accept,
                  max_tree_depth=max_tree_depth, dense_mass=False)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method=chain_method, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",), **args)
    return mcmc, design


def mroas_posterior(mcmc, data, design):
    """Posterior beta & kappa draws -> roster-level marginal-ROAS draws.

    Uses the *estimator's* mROAS constant (design['mroas_const']), so a
    misspecified response is scored on its own marginal effect against the
    true mROAS.
    """
    s = mcmc.get_samples()
    beta = np.asarray(s["beta"])                  # (draws, R, C)
    kappa = np.asarray(s["kappa"])                # (draws,)
    Cmat = design["mroas_const"]                  # (R, C)
    kappa_eff = kappa / (1.0 - data["lam"])
    mroas = (1.0 + kappa_eff)[:, None, None] * beta * Cmat[None]
    roster = mroas.mean(axis=1)                   # (draws, C)
    return roster, kappa, beta


# --------------------------------------------------------------------------- #
# Convergence diagnostics (Task 2)                                            #
# --------------------------------------------------------------------------- #
# Gate on the estimand-driving parameters (what roster mROAS + kappa depend on).
_DIAG_VARS = ("beta0", "logb", "sigma", "kappa")


def divergence_count(mcmc):
    extra = mcmc.get_extra_fields(group_by_chain=False)
    if "diverging" in extra:
        return int(np.asarray(extra["diverging"]).sum())
    return -1


def convergence_diagnostics(mcmc) -> dict:
    """Max R-hat and min bulk/tail ESS over the estimand-driving parameters
    (beta0/sigma/kappa, or logb for no-pool), the divergence count, and the
    divergence RATE. Operates on small arrays via arviz's array API (no full
    InferenceData build), so it is ~free vs the sampling."""
    import arviz as az
    grouped = mcmc.get_samples(group_by_chain=True)
    n_chains, n_draws = (np.asarray(next(iter(grouped.values()))).shape[:2]
                         if grouped else (1, 1))
    rhats, ess_b, ess_t = [], [], []
    for v in _DIAG_VARS:
        if v not in grouped:
            continue
        arr = np.asarray(grouped[v])
        if arr.ndim == 2:
            arr = arr[:, :, None]
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
        for j in range(arr.shape[-1]):
            a = arr[:, :, j]
            try:
                rhats.append(float(az.rhat(a)))
                ess_b.append(float(az.ess(a, method="bulk")))
                ess_t.append(float(az.ess(a, method="tail")))
            except Exception:
                pass
    fin = lambda xs: [x for x in xs if np.isfinite(x)]
    rhats, ess_b, ess_t = fin(rhats), fin(ess_b), fin(ess_t)
    ndiv = divergence_count(mcmc)
    total = int(n_chains * n_draws)
    return dict(
        max_rhat=float(np.max(rhats)) if rhats else np.nan,
        n_divergent=ndiv, total_draws=total,
        div_rate=float(ndiv / total) if total else np.nan,
        min_ess_bulk=float(np.min(ess_b)) if ess_b else np.nan,
        min_ess_tail=float(np.min(ess_t)) if ess_t else np.nan,
    )
