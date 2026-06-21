"""Vega — Counterpoint identification study (synthetic-data parameter recovery).

A hierarchical Bayesian MMM + geo-lift calibration is tested for its ability to
recover *planted* channel marginal-ROAS and the organic->paid multiplier under
the four music-industry data pathologies: collinearity, scarcity, thin
per-release data, and endogeneity.

Modules
-------
config   : scenario grid, planted-truth constants, decision thresholds.
dgp      : the data-generating process (plant truth -> simulate the 8 tables).
geo      : geo-lift module (randomized-holdout lift -> meta-analysis -> prior).
model    : the estimator under test (NumPyro hierarchical NB MMM) + mROAS post.
metrics  : recovery scoring (bias, coverage, width, classification, kappa).
study    : the Monte-Carlo harness over the scenario grid + the GO/KILL verdict.
"""

__all__ = ["config", "dgp", "geo", "model", "metrics", "study"]
