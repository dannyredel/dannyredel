"""Recovery scoring (study §2, §7).

Each fit is scored against the planted truth on the quantities the hypotheses
turn on: bias (H1), 80% credible-interval coverage (H2), interval width and
channel classification vs. the break-even bar (H3), RMSE for pooling value (H4),
and whether the kappa interval excludes zero (H5).
"""
from __future__ import annotations

import numpy as np

from . import config as cfg


def _ci(draws, lo=0.10, hi=0.90):
    return np.quantile(draws, lo), np.quantile(draws, hi)


def score_fit(roster_draws, kappa_draws, truth, data) -> dict:
    """roster_draws: (draws, C) posterior of roster-mean mROAS per channel."""
    true = truth.roster_mroas                       # (C,)
    med = np.median(roster_draws, axis=0)           # (C,)
    bar = cfg.BREAK_EVEN_BAR

    per_channel = {}
    classified, covered, widths, rel_bias = [], [], [], []
    for c, name in enumerate(cfg.CHANNELS):
        lo, hi = _ci(roster_draws[:, c])
        # H1 — relative bias
        rb = (med[c] - true[c]) / true[c]
        # H2 — coverage of the true value
        cov = bool(lo <= true[c] <= hi)
        # H3 — classification vs the bar (sign of median - bar)
        pred_above = med[c] > bar
        true_above = true[c] > bar
        correct = bool(pred_above == true_above)
        straddles = bool(lo <= bar <= hi)
        width = hi - lo

        per_channel[name] = dict(
            true=float(true[c]), median=float(med[c]), lo=float(lo), hi=float(hi),
            rel_bias=float(rb), covered=cov, width=float(width),
            classified_correct=correct, straddles_bar=straddles,
        )
        classified.append(correct)
        covered.append(cov)
        widths.append(width)
        rel_bias.append(abs(rb))

    # H5 — kappa interval excludes zero
    klo, khi = _ci(kappa_draws)
    kappa_excludes_zero = bool(klo > 0.0)

    return dict(
        per_channel=per_channel,
        class_accuracy=float(np.mean(classified)),
        median_abs_rel_bias=float(np.median(rel_bias)),
        mean_coverage=float(np.mean(covered)),
        mean_width=float(np.mean(widths)),
        kappa_median=float(np.median(kappa_draws)),
        kappa_lo=float(klo), kappa_hi=float(khi),
        kappa_excludes_zero=kappa_excludes_zero,
        # RMSE of roster mROAS across channels (for the H4 pooling contrast)
        rmse=float(np.sqrt(np.mean((med - true) ** 2))),
        n_straddle=int(sum(v["straddles_bar"] for v in per_channel.values())),
    )


def aggregate_cell(fit_scores: list[dict]) -> dict:
    """Aggregate per-sim scores into the cell-level recovery summary."""
    if not fit_scores:
        return {}
    A = lambda key: float(np.mean([f[key] for f in fit_scores]))
    out = dict(
        n_sims=len(fit_scores),
        class_accuracy=A("class_accuracy"),
        median_abs_rel_bias=float(np.median([f["median_abs_rel_bias"] for f in fit_scores])),
        coverage=A("mean_coverage"),
        mean_width=A("mean_width"),
        rmse=A("rmse"),
        kappa_excludes_zero_frac=A("kappa_excludes_zero"),
        kappa_median=float(np.median([f["kappa_median"] for f in fit_scores])),
        mean_straddle=A("n_straddle"),
    )
    # per-channel coverage / classification (the H3 money detail)
    chans = {}
    for name in cfg.CHANNELS:
        chans[name] = dict(
            coverage=float(np.mean([f["per_channel"][name]["covered"] for f in fit_scores])),
            class_accuracy=float(np.mean(
                [f["per_channel"][name]["classified_correct"] for f in fit_scores])),
            median_rel_bias=float(np.median(
                [f["per_channel"][name]["rel_bias"] for f in fit_scores])),
            mean_width=float(np.mean([f["per_channel"][name]["width"] for f in fit_scores])),
            straddle_frac=float(np.mean(
                [f["per_channel"][name]["straddles_bar"] for f in fit_scores])),
        )
    out["channels"] = chans
    return out


def verdict(go_cell: dict, kill_cell: dict, pooling_reduction: float) -> str:
    """Apply the GO / RESCOPE / KILL decision rules (study §8)."""
    go_ok = (
        go_cell.get("class_accuracy", 0) > cfg.CLASS_ACCURACY_GO
        and go_cell.get("median_abs_rel_bias", 1) < cfg.REL_BIAS_GO
        and cfg.COVERAGE_LO <= go_cell.get("coverage", 0) <= cfg.COVERAGE_HI
        and go_cell.get("kappa_excludes_zero_frac", 0) >= cfg.KAPPA_EXCLUDES_ZERO_GO
    )
    overconfident = go_cell.get("coverage", 1) < 0.5
    pooling_pays = pooling_reduction >= cfg.POOLING_RMSE_REDUCTION

    if go_ok and pooling_pays:
        return "GO"
    if overconfident or not pooling_pays:
        return "KILL"
    # channel-level fails but if aggregate-ish signals survive -> rescope
    return "RESCOPE"
