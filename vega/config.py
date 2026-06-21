"""Planted truth, scenario grid, and decision thresholds (study §5, §8).

These are the *defensible starting values for a mid distributor* from the design
spec. Anything a reviewer might want to re-tune lives here, not buried in code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

# --- Channels & geography (spec §5) -----------------------------------------
CHANNELS = ["meta", "tiktok", "spotify", "plugger"]
C = len(CHANNELS)
META = CHANNELS.index("meta")

TERRITORIES = ["DE", "NL", "UK", "US", "FR"]          # DACH-first ICP
G = len(TERRITORIES)
ANCHOR_TERRITORIES = ["DE", "NL"]                      # clean-geo holdout cells

# --- Campaign window --------------------------------------------------------
# Spec DGP is 84 daily points; we model at WEEKLY cadence (the "8-12 week
# campaign") for tractability. Same structure, ~7x fewer observations.
N_PERIODS = 12                                         # weeks of signal

# --- Planted marginal-ROAS targets (streams / EUR), spec §5 -----------------
# Two channels ABOVE the break-even bar, two BELOW -> the classification job.
TRUE_MROAS = {"meta": 9.0, "tiktok": 14.0, "spotify": 4.0, "plugger": 2.5}
BREAK_EVEN_BAR = 6.0                                   # label's decision threshold

# --- Channel mix (Dirichlet means) & spend skew -----------------------------
CHANNEL_MIX = {"meta": 0.40, "tiktok": 0.30, "spotify": 0.20, "plugger": 0.10}
DIRICHLET_CONCENTRATION = 30.0                         # how tight the mix is
BUDGET_MEDIAN = 2000.0                                 # EUR, LogNormal median
BUDGET_SIGMA = 0.7                                     # -> p90 ~ EUR 8k
PRIORITY_FRACTION = 0.10                               # share of priority releases
PRIORITY_MULTIPLIER = 15.0                             # their budget blow-up

# --- Adstock / saturation / spillover (treated as KNOWN calibration) --------
THETA = {"meta": 0.50, "tiktok": 0.35, "spotify": 0.45, "plugger": 0.60}
HILL_SHAPE = {"meta": 2.0, "tiktok": 2.0, "spotify": 2.0, "plugger": 2.0}
# Hill half-saturation k_c is scaled to each channel's median adstock at sim time.
SPILLOVER_LAMBDA = 0.5                                 # weekly carryover of paidStock

# --- Baseline (organic) -----------------------------------------------------
FANBASE_MEDIAN = 1500.0                                # daily streams, LogNormal
FANBASE_SIGMA = 0.6
BASELINE_TAU = 3.0                                     # weeks, post-launch decay
EDITORIAL_RATE = 0.15                                  # Poisson arrivals / release-week
EDITORIAL_LOGMEAN = 0.3                                # LogNormal magnitude (frac of baseline)
EDITORIAL_LOGSIGMA = 0.8

# --- Negative-Binomial overdispersion ---------------------------------------
NB_PHI = 10.0                                          # Var = mu + mu^2 / phi

# --- Territory size shares (sum to 1) ---------------------------------------
TERRITORY_SHARE = {"DE": 0.30, "NL": 0.15, "UK": 0.20, "US": 0.25, "FR": 0.10}


@dataclass(frozen=True)
class Scenario:
    """One cell of the Monte-Carlo grid (spec §7)."""
    N: int                 # roster size
    rho: float             # inter-channel spend correlation (collinearity dial)
    kappa: float           # organic->paid multiplier
    gamma: float           # endogeneity strength
    geo_anchor: bool       # randomized Meta holdout present?
    geo_frac: float = 0.40 # fraction of releases with a holdout

    def label(self) -> str:
        g = "geo" if self.geo_anchor else "nogeo"
        return f"N{self.N}_rho{self.rho}_k{self.kappa}_g{self.gamma}_{g}"


# --- Full spec grid (~144 cells) --------------------------------------------
def full_grid() -> list[Scenario]:
    cells = []
    for N, rho, kappa, gamma, geo in product(
        [10, 25, 50, 100], [0.3, 0.6, 0.9], [0.0, 0.3, 0.8], [0.0, 0.5], [False, True]
    ):
        cells.append(Scenario(N=N, rho=rho, kappa=kappa, gamma=gamma, geo_anchor=geo))
    return cells


# --- Reduced, decision-relevant grid (what we actually run) -----------------
def gating_grid() -> list[Scenario]:
    """The cells that drive the GO / RESCOPE / KILL verdict (spec §8).

    * the realistic GO cell, with and without the geo anchor (H8)
    * a rho-sweep at N=50 to locate the collinearity boundary (H6)
    * the KILL stress cell (rho=0.9, no geo)
    * an N-sweep to show pooling pays (H4)
    """
    cells = [
        # rho sweep, geo ON  (H1-H3, H6)
        Scenario(N=50, rho=0.3, kappa=0.8, gamma=0.5, geo_anchor=True),
        Scenario(N=50, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),   # GO cell
        Scenario(N=50, rho=0.9, kappa=0.8, gamma=0.5, geo_anchor=True),
        # rho sweep, geo OFF (H8 contrast + KILL stress at rho=0.9)
        Scenario(N=50, rho=0.3, kappa=0.8, gamma=0.5, geo_anchor=False),
        Scenario(N=50, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=False),
        Scenario(N=50, rho=0.9, kappa=0.8, gamma=0.5, geo_anchor=False),  # KILL stress
        # N sweep at the GO setting (H4 — pooling pays). The top point is N=25
        # rather than N=100: the verdict does not need N=100 and the larger model
        # x parallel-chain device partitions exhausts memory in a small container.
        Scenario(N=10, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),
        Scenario(N=25, rho=0.6, kappa=0.8, gamma=0.5, geo_anchor=True),
    ]
    return cells


# --- Estimators under test (spec §6) ----------------------------------------
ESTIMATORS = ["nopool", "pooled", "pooled_geo"]

# --- GO / RESCOPE / KILL thresholds (spec §2, §8) ---------------------------
CLASS_ACCURACY_GO = 0.80          # H3
REL_BIAS_GO = 0.25                # H1
COVERAGE_LO, COVERAGE_HI = 0.70, 0.90   # H2
POOLING_RMSE_REDUCTION = 0.20     # H4
KAPPA_EXCLUDES_ZERO_GO = 0.80     # H5 (fraction of sims)
