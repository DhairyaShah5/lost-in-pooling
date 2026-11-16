"""Paired bootstrap of the difference in macro AUROC between two read-outs.

Two read-outs scored on the *same* test images are compared by resampling the
images **once** per draw and evaluating both read-outs on that draw. The
difference of their per-bin macro AUROCs is then a paired statistic: image-level
noise that moves both read-outs the same way cancels, so the confidence interval
of the difference is much tighter than what two independent bootstraps (one per
read-out) would give. ``paired=False`` reproduces the independent variant for
reference.

Per bin the result holds the point-estimate difference ``delta`` (A minus B),
its percentile 95% CI and a two-sided bootstrap p-value (``p_two_sided``: the
fraction of draws whose sign is opposite to the point estimate, doubled and
capped at 1; its resolution is ``2 / n_boot``). The ``overall`` block does the
same for the pooled (all sizes) macro AUROC and for mAP, the latter via a
vectorised weighted average precision that mirrors the weighted Mann-Whitney
core of ``lip.eval.metrics``.
"""
from __future__ import annotations

import warnings

import numpy as np

from lip.eval.bins import BIN_EDGES, BIN_NAMES, N_BINS, assign_bins
from lip.eval.metrics import (
    MIN_POS,
    N_BOOT,
    _check_inputs,
    _weighted_auroc_by_bin,
    bootstrap_weights,
    macro_curve,
    overall_metrics,
    per_class_bin_auroc,
)


# --------------------------------------------------------------------------------------
# Weighted average precision (vectorised over bootstrap draws)
# --------------------------------------------------------------------------------------
def weighted_average_precision(s: np.ndarray, w: np.ndarray, is_pos: np.ndarray) -> np.ndarray:
    """Average precision of one class for every weight vector at once.

    Parameters
    ----------
    s
        ``(N,)`` scores of one class.
    w
        ``(B, N)`` non-negative weights (image multiplicities), one row per draw.
    is_pos
        ``(N,)`` bool positives.

    Returns
    -------
    np.ndarray
        ``(B,)`` AP; ``nan`` for a draw with no positive weight. With a row of
        ones (or integer multiplicities) this equals
        ``sklearn.metrics.average_precision_score`` on the (repeated) images:
        ``AP = sum_k (R_k - R_{k-1}) P_k`` over the distinct score values ``k``
        in descending order, ties grouped, exactly as sklearn's PR curve does.
    """
    s = np.asarray(s, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    is_pos = np.asarray(is_pos).astype(bool)
    n = s.shape[0]
    order = np.argsort(-s, kind="stable")  # descending score
    s_sorted = s[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.not_equal(s_sorted[1:], s_sorted[:-1], out=new_group[1:])
    first = np.flatnonzero(new_group)
    last = np.append(first[1:], n) - 1  # last sorted position of each tie group

    w_sorted = w[:, order]
    cum_all = np.cumsum(w_sorted, axis=1)[:, last]  # (B, U) weight at or above threshold
    cum_pos = np.cumsum(w_sorted * is_pos[order], axis=1)[:, last]  # (B, U) positive weight
    total_pos = cum_pos[:, -1:]
    precision = np.divide(cum_pos, cum_all, out=np.zeros_like(cum_pos), where=cum_all > 0)
    recall = np.divide(cum_pos, total_pos, out=np.zeros_like(cum_pos), where=total_pos > 0)
    ap = (np.diff(recall, axis=1, prepend=0.0) * precision).sum(axis=1)
    ap[total_pos[:, 0] <= 0] = np.nan
    return ap


# --------------------------------------------------------------------------------------
# Draws of the macro statistics under a given weight matrix
# --------------------------------------------------------------------------------------
def macro_draws(
    scores: np.ndarray, y: np.ndarray, bins: np.ndarray, w: np.ndarray, min_pos: int = MIN_POS
) -> dict[str, np.ndarray]:
    """Macro statistics of one read-out on every bootstrap draw of ``w``.

    Parameters
    ----------
    scores, y
        ``(N, C)`` scores and presence labels.
    bins
        ``(N, C)`` size-bin index per cell (from :func:`lip.eval.bins.assign_bins`).
    w
        ``(B, N)`` image multiplicities shared by whichever read-outs are compared.
    min_pos
        Per-bin class eligibility, fixed from the full data as in
        :func:`lip.eval.metrics.bootstrap_from_bins`.

    Returns
    -------
    dict
        ``"per_bin"`` ``(B, N_BINS)`` macro AUROC per bin, ``"auroc_macro"`` ``(B,)``
        pooled macro AUROC (classes with >= 1 positive and negative, as
        :func:`overall_metrics`) and ``"map"`` ``(B,)`` mean AP over the same classes.
    """
    n_boot = w.shape[0]
    n_classes = scores.shape[1]
    draws = np.full((n_boot, n_classes, N_BINS), np.nan)
    draws_all = np.full((n_boot, n_classes), np.nan)
    draws_ap = np.full((n_boot, n_classes), np.nan)
    for c in range(n_classes):
        pos = y[:, c] == 1
        neg = y[:, c] == 0
        if neg.sum() == 0 or pos.sum() == 0:
            continue
        s = scores[:, c]
        pos_bins = np.where(pos, bins[:, c], -1).astype(np.int64)
        n_pos = np.bincount(pos_bins[pos_bins >= 0], minlength=N_BINS)[:N_BINS]
        eligible = n_pos >= min_pos
        if eligible.any():
            auroc, _ = _weighted_auroc_by_bin(s, w, neg, pos_bins, N_BINS)
            auroc[:, ~eligible] = np.nan
            draws[:, c, :] = auroc
        pooled, _ = _weighted_auroc_by_bin(s, w, neg, np.where(pos, 0, -1).astype(np.int64), 1)
        draws_all[:, c] = pooled[:, 0]
        draws_ap[:, c] = weighted_average_precision(s, w, pos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-nan slices
        return {
            "per_bin": np.nanmean(draws, axis=1),
            "auroc_macro": np.nanmean(draws_all, axis=1),
            "map": np.nanmean(draws_ap, axis=1),
        }


def summarise_difference(delta: float, draws: np.ndarray) -> dict:
    """Percentile CI and two-sided bootstrap p-value of a point estimate and its draws.

    ``p_two_sided`` is the fraction of finite draws whose sign is opposite to
    ``delta``, doubled and capped at 1 (``1.0`` when ``delta == 0``; ``nan`` when
    ``delta`` or every draw is non-finite).
    """
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    if not np.isfinite(delta) or d.size == 0:
        return {"delta": float(delta), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_two_sided": float("nan"), "n_draws": int(d.size)}
    lo, hi = np.percentile(d, [2.5, 97.5])
    if delta > 0:
        frac = float(np.mean(d < 0))
    elif delta < 0:
        frac = float(np.mean(d > 0))
    else:
        frac = 0.5
    return {"delta": float(delta), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_two_sided": float(min(1.0, 2.0 * frac)), "n_draws": int(d.size)}


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def paired_bootstrap_difference(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    y: np.ndarray,
    max_box_frac: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 0,
    min_pos: int = MIN_POS,
    paired: bool = True,
) -> dict:
    """Paired image-level bootstrap of the macro-AUROC difference A minus B.

    Parameters
    ----------
    scores_a, scores_b
        ``(N, C)`` per-image per-class scores of the two read-outs, same images in
        the same order.
    y, max_box_frac
        ``(N, C)`` presence labels and relative size of the largest instance.
    n_boot, seed
        Bootstrap draws and the seed of ``numpy.random.default_rng``.
    min_pos
        Macro-average threshold (a class enters a bin when it has at least this
        many positives there, decided on the full data and shared by A and B).
    paired
        ``True`` evaluates both read-outs on the same resampled images (the
        default and the point of this module); ``False`` draws an independent
        resample for B, which only serves to show how much the pairing helps.

    Returns
    -------
    dict
        ``{"per_bin": {bin_name: {"delta", "ci_lo", "ci_hi", "p_two_sided",
        "n_draws", "auroc_a", "auroc_b", "n_classes", "n_pos"}},
        "overall": {"auroc_macro": {...same keys..., "a", "b"}, "map": {...}},
        "n_images", "n_boot", "seed", "min_pos", "paired", "bins"}``.
    """
    scores_a, y = _check_inputs(scores_a, y)
    scores_b, _ = _check_inputs(scores_b, y)
    max_box_frac = np.asarray(max_box_frac)
    if max_box_frac.shape != y.shape:
        raise ValueError(f"max_box_frac {max_box_frac.shape} must match y {y.shape}")
    bins = assign_bins(max_box_frac)
    n = y.shape[0]

    curve_a = macro_curve(per_class_bin_auroc(scores_a, y, max_box_frac, min_pos=min_pos), min_pos=min_pos)
    curve_b = macro_curve(per_class_bin_auroc(scores_b, y, max_box_frac, min_pos=min_pos), min_pos=min_pos)
    overall_a = overall_metrics(scores_a, y)
    overall_b = overall_metrics(scores_b, y)

    rng = np.random.default_rng(seed)
    w_a = bootstrap_weights(n, n_boot, rng)
    w_b = w_a if paired else bootstrap_weights(n, n_boot, rng)
    draws_a = macro_draws(scores_a, y, bins, w_a, min_pos=min_pos)
    draws_b = macro_draws(scores_b, y, bins, w_b, min_pos=min_pos)

    per_bin: dict = {}
    for b, name in enumerate(BIN_NAMES):
        auroc_a = curve_a[b]["auroc_macro"]
        auroc_b = curve_b[b]["auroc_macro"]
        cell = summarise_difference(auroc_a - auroc_b, draws_a["per_bin"][:, b] - draws_b["per_bin"][:, b])
        cell.update(auroc_a=float(auroc_a), auroc_b=float(auroc_b),
                    n_classes=int(curve_a[b]["n_classes"]), n_pos=int(curve_a[b]["n_pos"]))
        per_bin[name] = cell
    overall: dict = {}
    for key in ("auroc_macro", "map"):
        cell = summarise_difference(overall_a[key] - overall_b[key], draws_a[key] - draws_b[key])
        cell.update(a=float(overall_a[key]), b=float(overall_b[key]), n_classes=int(overall_a["n_classes"]))
        overall[key] = cell
    return {
        "per_bin": per_bin,
        "overall": overall,
        "n_images": int(n),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "min_pos": int(min_pos),
        "paired": bool(paired),
        "bins": {"edges": list(BIN_EDGES), "names": list(BIN_NAMES)},
    }


def format_compare_lines(cmp: dict, name_a: str, name_b: str) -> list[str]:
    """One summary line per bin plus two overall lines (``delta`` is ``name_a - name_b``)."""
    def fmt(cell: dict, label: str, ka: str, kb: str) -> str:
        if not np.isfinite(cell["delta"]):
            return f"{label:>10}  {name_a}-{name_b}  delta=   nan  (no eligible class)"
        return (f"{label:>10}  {name_a}-{name_b}  delta={cell['delta']:+.3f}  "
                f"95% CI [{cell['ci_lo']:+.3f}, {cell['ci_hi']:+.3f}]  p={cell['p_two_sided']:.3f}  "
                f"({name_a} {cell[ka]:.3f}, {name_b} {cell[kb]:.3f}, n_classes={cell['n_classes']})")

    lines = [fmt(cell, name, "auroc_a", "auroc_b") for name, cell in cmp["per_bin"].items()]
    lines.append(fmt(cmp["overall"]["auroc_macro"], "all AUROC", "a", "b"))
    lines.append(fmt(cmp["overall"]["map"], "all mAP", "a", "b"))
    return lines
