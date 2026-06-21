"""The data-generating process (study §4).

We control the ground truth, so "did we get it back?" is unambiguous. The DGP
bakes in the four pathologies that make music hard:

  * collinearity  — channels co-launch (shared ramp w[t], tuned by rho)
  * scarcity      — 12 weeks of signal, a handful of territories
  * thin data     — rescued only by roster pooling
  * endogeneity   — bigger releases get more budget (gamma)

Adstock, Hill saturation and spillover carry-over are deterministic transforms
shared by the DGP and the estimator, so any recovery failure is *identification*,
not misspecification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as cfg


# --------------------------------------------------------------------------- #
# Deterministic transforms (used by both the DGP and the estimator)           #
# --------------------------------------------------------------------------- #
def geometric_adstock(spend: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """adstock[t] = spend[t] + theta * adstock[t-1].   spend: (T, U, C)."""
    T = spend.shape[0]
    out = np.empty_like(spend, dtype=float)
    out[0] = spend[0]
    for t in range(1, T):
        out[t] = spend[t] + theta * out[t - 1]
    return out


def hill(a: np.ndarray, k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Saturating response in [0, 1):  a^s / (a^s + k^s)."""
    a = np.maximum(a, 0.0)
    num = a ** s
    return num / (num + k ** s + 1e-12)


def hill_grad(a: np.ndarray, k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """d/da Hill(a) = s k^s a^(s-1) / (a^s + k^s)^2 — the marginal slope."""
    a = np.maximum(a, 1e-9)
    ks = k ** s
    asu = a ** s
    return s * ks * a ** (s - 1) / (asu + ks) ** 2


def carryover(x: np.ndarray, lam: float) -> np.ndarray:
    """stock[t] = x[t] + lam * stock[t-1].  x: (T, U) or (T, ...)."""
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, x.shape[0]):
        out[t] = x[t] + lam * out[t - 1]
    return out


# --------------------------------------------------------------------------- #
# Planted truth                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Truth:
    beta: np.ndarray          # (R, C) structural coefficients
    beta0: np.ndarray         # (C,) roster log-means
    sigma: np.ndarray         # (C,) release-level sd of log beta
    theta: np.ndarray         # (C,) adstock retention (known)
    k: np.ndarray             # (C,) Hill half-saturation (known)
    s: np.ndarray             # (C,) Hill shape (known)
    kappa: float              # organic->paid multiplier
    lam: float                # spillover carryover
    fanbase: np.ndarray       # (R,) daily-stream baseline
    priority: np.ndarray      # (R,) bool
    abar: np.ndarray          # (R, C) campaign-mean adstock (operating point)
    mroas: np.ndarray         # (R, C) TRUE marginal ROAS per release (the target)
    roster_mroas: np.ndarray  # (C,) roster-mean true mROAS

    @property
    def kappa_eff(self) -> float:
        return self.kappa / (1.0 - self.lam)


def _mroas_constant(abar: np.ndarray, k, s, theta) -> np.ndarray:
    """C[r,c] = Hill'(abar) / (1 - theta) — the spend->streams chain rule factor.

    mROAS[r,c] = (1 + kappa_eff) * beta[r,c] * C[r,c].  Identical for plant and
    recovery, so recovering mROAS reduces to recovering beta * (1 + kappa_eff).
    """
    return hill_grad(abar, k[None, :], s[None, :]) / (1.0 - theta[None, :])


# --------------------------------------------------------------------------- #
# Spend (collinearity + endogeneity)                                          #
# --------------------------------------------------------------------------- #
def _draw_spend(rng, N, rho, gamma, fanbase, priority):
    """spend[t, u, c] with co-launch collinearity (rho) and endogeneity (gamma)."""
    T, Gn, Cn = cfg.N_PERIODS, cfg.G, cfg.C

    # --- per-release budget: endogenous in expected organic -----------------
    z = (np.log(fanbase) - np.log(fanbase).mean()) / (np.log(fanbase).std() + 1e-9)
    log_B = (np.log(cfg.BUDGET_MEDIAN) + gamma * z
             + rng.normal(0, cfg.BUDGET_SIGMA, size=N))
    B = np.exp(log_B)
    B[priority] *= cfg.PRIORITY_MULTIPLIER

    # --- channel mix --------------------------------------------------------
    alpha = np.array([cfg.CHANNEL_MIX[c] for c in cfg.CHANNELS]) * cfg.DIRICHLET_CONCENTRATION
    shares = rng.dirichlet(alpha, size=N)                     # (N, C)

    # --- shared launch ramp + collinearity dial -----------------------------
    weeks = np.arange(T)
    ramp = np.exp(-weeks / 4.0)                               # high at launch
    ramp = ramp / ramp.sum()
    # profile[t, c] = sqrt(rho)*common[t] + sqrt(1-rho)*idio[t,c]  -> corr ~ rho
    common = rng.lognormal(0.0, 0.4, size=(T, 1)) * ramp[:, None]
    idio = rng.lognormal(0.0, 0.4, size=(T, Cn)) * ramp[:, None]
    profile = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    profile = profile / profile.sum(axis=0, keepdims=True)    # each channel sums to 1

    geo_share = np.array([cfg.TERRITORY_SHARE[g] for g in cfg.TERRITORIES])  # (G,)

    # spend[t, r, g, c] = B[r] * shares[r,c] * profile[t,c] * geo_share[g]
    spend = np.einsum("r,rc,tc,g->trgc", B, shares, profile, geo_share)
    return spend, B, shares


def _apply_holdouts(rng, spend, frac):
    """Randomized Meta holdout: zero Meta spend in anchor territories for a
    random subset of releases over a random window -> exogenous variation."""
    T, N, Gn, Cn = spend.shape
    terr_idx = [cfg.TERRITORIES.index(t) for t in cfg.ANCHOR_TERRITORIES]
    held = rng.random(N) < frac
    out = spend.copy()
    for r in np.where(held)[0]:
        start = rng.integers(2, max(3, T - 3))               # randomized window
        for g in terr_idx:
            out[start:, r, g, cfg.META] = 0.0
    return out, held


# --------------------------------------------------------------------------- #
# Simulate one roster                                                          #
# --------------------------------------------------------------------------- #
def simulate_roster(scn: cfg.Scenario, seed: int):
    """Plant the truth, then emit the synthetic outcome panel (study §4, §10)."""
    rng = np.random.default_rng(seed)
    N, T, Gn, Cn = scn.N, cfg.N_PERIODS, cfg.G, cfg.C

    # --- release metadata ---------------------------------------------------
    fanbase = rng.lognormal(np.log(cfg.FANBASE_MEDIAN), cfg.FANBASE_SIGMA, size=N)
    priority = rng.random(N) < cfg.PRIORITY_FRACTION

    # --- spend (collinearity + endogeneity) ---------------------------------
    spend, B, shares = _draw_spend(rng, N, scn.rho, scn.gamma, fanbase, priority)
    held = np.zeros(N, dtype=bool)
    if scn.geo_anchor:
        spend, held = _apply_holdouts(rng, spend, scn.geo_frac)

    # --- known adstock / saturation constants -------------------------------
    theta = np.array([cfg.THETA[c] for c in cfg.CHANNELS])
    s = np.array([cfg.HILL_SHAPE[c] for c in cfg.CHANNELS])
    # flatten (release, territory) -> unit so transforms see (T, U, C)
    U = N * Gn
    rel_of_unit = np.repeat(np.arange(N), Gn)
    terr_of_unit = np.tile(np.arange(Gn), N)
    spend_u = spend.reshape(T, U, Cn)
    adstock_u = geometric_adstock(spend_u, theta)
    # k_c scaled to each channel's median (positive) adstock
    def _median_pos(col):
        pos = col[col > 0]
        return float(np.median(pos)) if pos.size else 1.0
    k = np.array([_median_pos(adstock_u[:, :, c]) for c in range(Cn)])
    X = hill(adstock_u, k[None, None, :], s[None, None, :])    # (T, U, C) Hill response

    # --- operating point + mROAS chain-rule constant ------------------------
    adstock_rgc = adstock_u.reshape(T, N, Gn, Cn)
    abar = adstock_rgc.mean(axis=(0, 2))                       # (N, C) release-mean adstock
    Cmat = _mroas_constant(abar, k, s, theta)                 # (N, C)

    # --- calibrate beta0 to hit target roster mROAS -------------------------
    kappa_eff = scn.kappa / (1.0 - cfg.SPILLOVER_LAMBDA)
    sigma = np.full(Cn, 0.35)                                  # release-level log-sd
    target = np.array([cfg.TRUE_MROAS[c] for c in cfg.CHANNELS])
    meanC = Cmat.mean(axis=0)                                  # (C,)
    beta0 = np.log(target / ((1.0 + kappa_eff) * meanC)) - 0.5 * sigma ** 2
    u = rng.normal(0, 1, size=(N, Cn))
    beta = np.exp(beta0[None, :] + sigma[None, :] * u)        # (N, C)

    # --- TRUE marginal ROAS (the recovery target) ---------------------------
    mroas = (1.0 + kappa_eff) * beta * Cmat                   # (N, C)
    roster_mroas = mroas.mean(axis=0)

    # --- mean structure -----------------------------------------------------
    beta_unit = beta[rel_of_unit]                             # (U, C)
    paid = np.einsum("tuc,uc->tu", X, beta_unit)             # (T, U)
    paid_stock = carryover(paid, cfg.SPILLOVER_LAMBDA)
    spillover = scn.kappa * paid_stock

    # baseline: fanbase * territory_share * decay * season * (1 + editorial)
    weeks = np.arange(T)
    decay = np.exp(-weeks / cfg.BASELINE_TAU)                 # (T,)
    terr_share = np.array([cfg.TERRITORY_SHARE[g] for g in cfg.TERRITORIES])
    base_level = fanbase                                      # daily -> use as level
    base_unit = base_level[rel_of_unit] * terr_share[terr_of_unit]   # (U,)
    editorial = _editorial_shocks(rng, T, U)                  # (T, U) multiplicative frac
    baseline = base_unit[None, :] * decay[:, None] * (1.0 + editorial)

    mu = baseline + paid + spillover
    mu = np.maximum(mu, 1e-3)
    streams = _negbinom(rng, mu, cfg.NB_PHI)                  # (T, U)

    truth = Truth(beta=beta, beta0=beta0, sigma=sigma, theta=theta, k=k, s=s,
                  kappa=scn.kappa, lam=cfg.SPILLOVER_LAMBDA, fanbase=fanbase,
                  priority=priority, abar=abar, mroas=mroas, roster_mroas=roster_mroas)

    # per-(release, channel) total adstocked spend -- the exposure measure the
    # minimum-spend floor (Task 5) is built on; roster total per channel too.
    rel_chan_adstock = adstock_u.reshape(T, N, Gn, Cn).sum(axis=(0, 2))   # (N, C)
    chan_adstock_total = rel_chan_adstock.sum(axis=0)                     # (C,)

    data = dict(
        streams=streams, X=X, spend_u=spend_u, editorial=editorial,
        rel_of_unit=rel_of_unit, terr_of_unit=terr_of_unit,
        fanbase=fanbase, N=N, T=T, G=Gn, C=Cn, U=U,
        k=k, s=s, theta=theta, lam=cfg.SPILLOVER_LAMBDA, abar=abar,
        Cmat=Cmat, held=held, geo_anchor=scn.geo_anchor,
        rel_chan_adstock=rel_chan_adstock, chan_adstock_total=chan_adstock_total,
    )
    return data, truth


def _editorial_shocks(rng, T, U):
    """Poisson-arriving, LogNormal-magnitude playlist adds that decay over weeks."""
    shocks = np.zeros((T, U))
    arrivals = rng.poisson(cfg.EDITORIAL_RATE, size=(T, U))
    mag = rng.lognormal(cfg.EDITORIAL_LOGMEAN, cfg.EDITORIAL_LOGSIGMA, size=(T, U))
    bump = arrivals * mag
    # each shock decays over subsequent weeks (half-life ~1.5 weeks)
    return carryover(bump, 0.6)


def _negbinom(rng, mu, phi):
    """NB with mean mu, Var = mu + mu^2/phi via Gamma-Poisson mixture."""
    shape = phi
    scale = mu / phi
    rate = rng.gamma(shape, scale)
    return rng.poisson(rate).astype(float)
