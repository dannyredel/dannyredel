"""Recovery scoring (study §2, §7) with convergence gating (Task 2).

Each fit is scored against the planted truth on the quantities the hypotheses
turn on: bias (H1), 80% credible-interval coverage (H2), interval width and
channel classification vs. the break-even bar (H3), RMSE for pooling value (H4),
and whether the kappa interval clears a small ROPE around 0 (H5 / the kappa=0
null). Cell metrics are computed only over *converged* sims; the discard rate is
itself reported.
"""
from __future__ import annotations

import numpy as np

from . import config as cfg


def _ci(draws, lo=0.10, hi=0.90):
    return np.quantile(draws, lo), np.quantile(draws, hi)


def score_fit(roster_draws, kappa_draws, truth, data, design=None) -> dict:
    """roster_draws: (draws, C) posterior of roster-mean mROAS per channel."""
    true = truth.roster_mroas                       # (C,)
    med = np.median(roster_draws, axis=0)           # (C,)
    bar = cfg.BREAK_EVEN_BAR
    chan_exposure = np.asarray(data["chan_adstock_total"])   # (C,) adstocked spend

    per_channel = {}
    classified, covered, widths, rel_bias = [], [], [], []
    for c, name in enumerate(cfg.CHANNELS):
        lo, hi = _ci(roster_draws[:, c])
        rb = (med[c] - true[c]) / true[c]                      # H1
        cov = bool(lo <= true[c] <= hi)                       # H2
        pred_above = med[c] > bar
        true_above = true[c] > bar
        correct = bool(pred_above == true_above)              # H3
        straddles = bool(lo <= bar <= hi)
        per_channel[name] = dict(
            true=float(true[c]), median=float(med[c]), lo=float(lo), hi=float(hi),
            rel_bias=float(rb), covered=cov, width=float(hi - lo),
            classified_correct=correct, straddles_bar=straddles,
            exposure=float(chan_exposure[c]),
        )
        classified.append(correct)
        covered.append(cov)
        widths.append(hi - lo)
        rel_bias.append(abs(rb))

    # H5 / null — kappa "detected" if the 80% CI clears the ROPE around 0
    klo, khi = _ci(kappa_draws)
    kappa_detected = bool(klo > cfg.KAPPA_ROPE)

    return dict(
        per_channel=per_channel,
        class_accuracy=float(np.mean(classified)),
        median_abs_rel_bias=float(np.median(rel_bias)),
        mean_coverage=float(np.mean(covered)),
        mean_width=float(np.mean(widths)),
        kappa_median=float(np.median(kappa_draws)),
        kappa_lo=float(klo), kappa_hi=float(khi),
        kappa_detected=kappa_detected,
        rmse=float(np.sqrt(np.mean((med - true) ** 2))),
        n_straddle=int(sum(v["straddles_bar"] for v in per_channel.values())),
    )


def is_converged(diag: dict) -> bool:
    """Convergence gate (Task 2): estimand R-hat <= RHAT_MAX and divergence
    rate <= DIV_RATE_MAX (the strict zero-divergence rule is reported, not gated
    on — see config)."""
    rhat = diag.get("max_rhat", np.inf)
    if rhat is None or not np.isfinite(rhat):
        return False
    return (rhat <= cfg.RHAT_MAX) and (diag.get("div_rate", 1.0) <= cfg.DIV_RATE_MAX)


def convergence_summary(diags: list[dict]) -> dict:
    """Aggregate per-fit diagnostics over ALL attempted sims of a cell."""
    if not diags:
        return {}
    rhats = [d["max_rhat"] for d in diags if np.isfinite(d.get("max_rhat", np.nan))]
    div = [d.get("n_divergent", -1) for d in diags]
    essb = [d["min_ess_bulk"] for d in diags if np.isfinite(d.get("min_ess_bulk", np.nan))]
    esst = [d["min_ess_tail"] for d in diags if np.isfinite(d.get("min_ess_tail", np.nan))]
    n_conv = int(sum(is_converged(d) for d in diags))
    return dict(
        n_attempted=len(diags), n_converged=n_conv,
        frac_dropped=float(1.0 - n_conv / len(diags)),
        max_rhat=float(np.max(rhats)) if rhats else np.nan,
        median_rhat=float(np.median(rhats)) if rhats else np.nan,
        mean_divergent=float(np.mean(div)),
        frac_with_divergences=float(np.mean([d > 0 for d in div])),     # strict-zero finding
        frac_strict_zero_div=float(np.mean([d == 0 for d in div])),
        min_ess_bulk=float(np.min(essb)) if essb else np.nan,
        min_ess_tail=float(np.min(esst)) if esst else np.nan,
    )


def aggregate_cell(fit_scores: list[dict], diags: list[dict] | None = None) -> dict:
    """Aggregate per-sim scores (converged only) + the convergence summary."""
    conv = convergence_summary(diags) if diags else {}
    if not fit_scores:
        return {"convergence": conv} if conv else {}
    A = lambda key: float(np.mean([f[key] for f in fit_scores]))
    out = dict(
        n_sims=len(fit_scores),
        class_accuracy=A("class_accuracy"),
        median_abs_rel_bias=float(np.median([f["median_abs_rel_bias"] for f in fit_scores])),
        coverage=A("mean_coverage"),
        mean_width=A("mean_width"),
        rmse=A("rmse"),
        kappa_detected_frac=A("kappa_detected"),
        kappa_median=float(np.median([f["kappa_median"] for f in fit_scores])),
        mean_straddle=A("n_straddle"),
    )
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
            exposure=float(np.mean([f["per_channel"][name]["exposure"] for f in fit_scores])),
        )
    out["channels"] = chans
    if conv:
        out["convergence"] = conv
    return out


def verdict(go_cell: dict, kill_cell: dict, pooling_reduction: float) -> str:
    """Apply the GO / RESCOPE / KILL decision rules (study §8)."""
    go_ok = (
        go_cell.get("class_accuracy", 0) > cfg.CLASS_ACCURACY_GO
        and go_cell.get("median_abs_rel_bias", 1) < cfg.REL_BIAS_GO
        and cfg.COVERAGE_LO <= go_cell.get("coverage", 0) <= cfg.COVERAGE_HI
        and go_cell.get("kappa_detected_frac", 0) >= cfg.KAPPA_EXCLUDES_ZERO_GO
    )
    overconfident = go_cell.get("coverage", 1) < 0.5
    pooling_pays = pooling_reduction >= cfg.POOLING_RMSE_REDUCTION
    if go_ok and pooling_pays:
        return "GO"
    if overconfident or not pooling_pays:
        return "KILL"
    return "RESCOPE"
