"""The estimator under test — NumPyro hierarchical NB MMM (study §6).

Structurally identical to the DGP (so any failure is identification, not
misspecification). Adstock retention, Hill saturation and the spillover
half-life are treated as KNOWN calibration constants (set by priors in
practice); we estimate what the product claim depends on:

  * beta[r, c]  — channel effects, with non-centered partial pooling (the
                  scarcity rescue, H4);
  * kappa       — the organic->paid multiplier (the music estimand, H5);
  * baseline    — organic level, decay, editorial control;
  * phi         — Negative-Binomial overdispersion.

Three estimators are exposed via flags so we can attribute the gains:
  nopool      — independent diffuse beta per release (no strength borrowed);
  pooled      — hierarchical partial pooling, generic prior;
  pooled_geo  — hierarchical + informative geo prior on beta0_meta (H8).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from . import config as cfg

# Run chains truly in parallel across the box's cores (2-4 chains ~ cost of 1).
numpyro.set_host_device_count(4)


def _carryover_scan(paid, lam):
    """stock[t] = paid[t] + lam * stock[t-1], scanned over time. paid: (T, U)."""
    def step(prev, p):
        new = p + lam * prev
        return new, new
    _, out = jax.lax.scan(step, jnp.zeros(paid.shape[1]), paid)
    return out


def _beta_scale_guess(data):
    """Empirical, channel-symmetric prior centre for beta (no per-channel leak).

    A pooled non-negative least squares of streams on [baseline, Hill responses]
    gives an overall effect scale; we centre every channel's prior on it so the
    prior never encodes *which* channel is strong.
    """
    T, U, Cn = data["X"].shape
    y = np.asarray(data["streams"]).reshape(-1)
    Xc = np.asarray(data["X"]).reshape(-1, Cn)
    weeks = np.arange(T)
    decay = np.exp(-weeks / cfg.BASELINE_TAU)
    tshare = np.array([cfg.TERRITORY_SHARE[g] for g in cfg.TERRITORIES])
    base = (data["fanbase"][data["rel_of_unit"]][None, :]
            * tshare[data["terr_of_unit"]][None, :] * decay[:, None]).reshape(-1)
    Z = np.column_stack([base, np.ones_like(base), Xc])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    chan = np.clip(coef[2:], 1e-3, None)
    return float(np.log(np.maximum(np.mean(chan), 1.0)))


def mmm_model(X, streams, rel_of_unit, terr_of_unit, zfan, editorial,
              tshare, lam, n_releases, n_terr, beta_center, hier=True, geo_prior=None):
    """Negative-Binomial hierarchical MMM. All arrays are JAX-ready.

    `tshare` (territory size shares) is fixed to its data-empirical value — it is
    a well-identified nuisance that otherwise fights the baseline scale and slows
    sampling, so we profile it out rather than sample it.
    """
    T, U, Cn = X.shape

    # --- channel effects ----------------------------------------------------
    if hier:
        if geo_prior is not None:
            # informative geo prior on Meta's roster log-mean; generic elsewhere
            means = jnp.array(np.where(np.arange(Cn) == cfg.META,
                                       geo_prior["mean"], beta_center))
            sds = jnp.array(np.where(np.arange(Cn) == cfg.META,
                                     geo_prior["sd"], 2.0))
            beta0 = numpyro.sample("beta0", dist.Normal(means, sds))
        else:
            beta0 = numpyro.sample(
                "beta0", dist.Normal(beta_center * jnp.ones(Cn), 2.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.7 * jnp.ones(Cn)))
        with numpyro.plate("rel", n_releases, dim=-2):
            with numpyro.plate("chan", Cn, dim=-1):
                u = numpyro.sample("u", dist.Normal(0.0, 1.0))      # non-centered
        beta = numpyro.deterministic("beta", jnp.exp(beta0 + sigma * u))
    else:
        # no partial pooling: independent diffuse beta per release
        with numpyro.plate("rel", n_releases, dim=-2):
            with numpyro.plate("chan", Cn, dim=-1):
                logb = numpyro.sample("logb", dist.Normal(beta_center, 2.0))
        beta = numpyro.deterministic("beta", jnp.exp(logb))

    # --- baseline (organic) -------------------------------------------------
    a0 = numpyro.sample("a0", dist.Normal(0.0, 2.0))
    a1 = numpyro.sample("a1", dist.Normal(1.0, 1.0))
    tau = numpyro.sample("tau", dist.LogNormal(jnp.log(3.0), 0.4))
    ed_coef = numpyro.sample("ed_coef", dist.HalfNormal(1.0))

    weeks = jnp.arange(T)
    decay = jnp.exp(-weeks / tau)                                   # (T,)
    base_level = jnp.exp(a0 + a1 * zfan)                           # (R,)
    base_unit = base_level[rel_of_unit] * tshare[terr_of_unit] * n_terr  # (U,)
    baseline = base_unit[None, :] * decay[:, None] * (1.0 + ed_coef * editorial)

    # --- paid + spillover ---------------------------------------------------
    beta_unit = beta[rel_of_unit]                                  # (U, C)
    paid = jnp.einsum("tuc,uc->tu", X, beta_unit)                 # (T, U)
    paid_stock = _carryover_scan(paid, lam)
    kappa = numpyro.sample("kappa", dist.HalfNormal(0.5))
    spillover = kappa * paid_stock

    mu = jnp.clip(baseline + paid + spillover, 1e-3, None)

    # --- Negative-Binomial likelihood --------------------------------------
    phi = numpyro.sample("phi", dist.LogNormal(jnp.log(10.0), 0.5))
    numpyro.sample("streams", dist.NegativeBinomial2(mu, phi), obs=streams)


def fit(data, estimator, geo_prior=None, num_warmup=400, num_samples=400,
        num_chains=2, seed=0):
    """Run NUTS for one estimator on one simulated roster. Returns InferenceData-ish."""
    hier = estimator != "nopool"
    use_geo = (estimator == "pooled_geo") and (geo_prior is not None)
    gp = geo_prior if use_geo else None

    beta_center = _beta_scale_guess(data)
    zfan = (np.log(data["fanbase"]) - np.log(data["fanbase"]).mean())
    zfan = zfan / (zfan.std() + 1e-9)

    # empirical territory shares from observed streams (fixed nuisance)
    streams = np.asarray(data["streams"])
    terr = data["terr_of_unit"]
    by_terr = np.array([streams[:, terr == g].sum() for g in range(data["G"])])
    tshare = by_terr / by_terr.sum()

    args = dict(
        X=jnp.asarray(data["X"]),
        streams=jnp.asarray(data["streams"]),
        rel_of_unit=jnp.asarray(data["rel_of_unit"]),
        terr_of_unit=jnp.asarray(data["terr_of_unit"]),
        zfan=jnp.asarray(zfan),
        editorial=jnp.asarray(data["editorial"]),
        tshare=jnp.asarray(tshare),
        lam=data["lam"], n_releases=data["N"], n_terr=data["G"],
        beta_center=beta_center, hier=hier, geo_prior=gp,
    )

    kernel = NUTS(mmm_model, target_accept_prob=0.85, max_tree_depth=8)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), **args)
    return mcmc


def mroas_posterior(mcmc, data):
    """Map posterior beta & kappa draws to roster-level marginal-ROAS draws.

    mROAS[r,c] = (1 + kappa/(1-lam)) * beta[r,c] * Cmat[r,c]
    roster mROAS_c = mean_r mROAS[r,c].  Returns (draws, C) array.
    """
    s = mcmc.get_samples()
    beta = np.asarray(s["beta"])                  # (draws, R, C)
    kappa = np.asarray(s["kappa"])                # (draws,)
    Cmat = data["Cmat"]                           # (R, C)
    kappa_eff = kappa / (1.0 - data["lam"])
    mroas = (1.0 + kappa_eff)[:, None, None] * beta * Cmat[None]
    roster = mroas.mean(axis=1)                   # (draws, C)
    return roster, kappa, beta


def divergence_count(mcmc):
    extra = mcmc.get_extra_fields(group_by_chain=False)
    if "diverging" in extra:
        return int(np.asarray(extra["diverging"]).sum())
    return -1
