"""AUROC as a function of object size: per-class/per-bin AUROC, macro curve, bootstrap CI.

Definitions (docs/CONTRACTS.md, "Metric"): for class ``c`` and size bin ``b`` the
positives are images with ``y[:, c] == 1`` whose ``max_box_frac[:, c]`` falls in bin
``b``; the negatives are *all* images with ``y[:, c] == 0`` and are shared across
bins. The macro curve averages over classes with at least ``min_pos`` positives in
the bin. The bootstrap resamples images with replacement.

Point estimates use ``sklearn.metrics.roc_auc_score``. The bootstrap uses an
equivalent weighted Mann-Whitney statistic (``_weighted_auroc_by_bin``) that
evaluates all draws, all bins of one class in a few vectorised numpy passes, which
is what makes 500 draws on 5,000 images take seconds rather than minutes.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from lip.eval.bins import N_BINS, assign_bins

#: AUROC of a scorer that carries no information; the reference line in every plot.
expected_chance: float = 0.5
EXPECTED_CHANCE: float = expected_chance

#: Minimum positives for a (class, bin) cell to enter the macro average.
MIN_POS: int = 10
#: Default bootstrap draws (CONTRACTS.md asks for 500; it runs in seconds, see module doc).
N_BOOT: int = 500


# --------------------------------------------------------------------------------------
# AUROC primitives
# --------------------------------------------------------------------------------------
def auroc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic with average ranks for ties.

    Numerically identical to ``sklearn.metrics.roc_auc_score`` (tested). Returns
    ``nan`` when either class is empty.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    pos = np.asarray(labels).ravel().astype(bool)
    n_pos = int(pos.sum())
    n_neg = int(pos.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    new_group = np.empty(s.size, dtype=bool)
    new_group[0] = True
    np.not_equal(s_sorted[1:], s_sorted[:-1], out=new_group[1:])
    group = np.cumsum(new_group) - 1
    first = np.flatnonzero(new_group)
    last = np.append(first[1:], s.size) - 1
    avg_rank = 0.5 * (first + last) + 1.0  # 1-based average rank per tie group
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = avg_rank[group]
    u = ranks[pos].sum() - 0.5 * n_pos * (n_pos + 1)
    return float(u / (n_pos * n_neg))


def _weighted_auroc_by_bin(
    s: np.ndarray, w: np.ndarray, is_neg: np.ndarray, pos_bins: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted AUROC of one class for every bin and every weight vector at once.

    Parameters
    ----------
    s
        ``(N,)`` scores of one class.
    w
        ``(B, N)`` non-negative weights, one row per bootstrap draw (multiplicity of
        each image). A row of ones reproduces the unweighted AUROC exactly.
    is_neg
        ``(N,)`` bool, the shared negatives (``y == 0``).
    pos_bins
        ``(N,)`` int, bin index of each positive and ``-1`` for every other image.
    n_bins
        Number of bins.

    Returns
    -------
    auroc, w_pos
        ``(B, n_bins)`` AUROC (``nan`` where a draw has no positive weight in the bin
        or no negative weight) and ``(B, n_bins)`` total positive weight per bin.

    Notes
    -----
    AUROC = P(s_pos > s_neg) + 0.5 P(s_pos == s_neg). Sorting the scores once and
    taking a cumulative sum of the negative weights gives, for each tie group, the
    negative weight strictly below it and the negative weight tied with it; each
    positive then contributes ``w_i * (below_i + 0.5 * tied_i)``.
    """
    n = s.shape[0]
    w = np.asarray(w, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.not_equal(s_sorted[1:], s_sorted[:-1], out=new_group[1:])
    group = np.cumsum(new_group) - 1  # tie-group id per sorted position
    first = np.flatnonzero(new_group)  # first sorted position of each group
    last = np.append(first[1:], n) - 1  # last sorted position of each group

    neg_w_sorted = w[:, order] * is_neg[order]  # (B, N)
    cum = np.cumsum(neg_w_sorted, axis=1)  # inclusive cumulative negative weight
    below = cum[:, first] - neg_w_sorted[:, first]  # (B, U) strictly below the group
    tied = cum[:, last] - below  # (B, U) negatives inside the group

    pos_idx = np.flatnonzero(pos_bins >= 0)
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    g_pos = group[inv[pos_idx]]  # tie group of each positive
    w_pos_img = w[:, pos_idx]  # (B, P)
    contrib = w_pos_img * (below[:, g_pos] + 0.5 * tied[:, g_pos])  # (B, P)

    onehot = np.zeros((pos_idx.size, n_bins), dtype=np.float64)
    onehot[np.arange(pos_idx.size), pos_bins[pos_idx]] = 1.0
    num = contrib @ onehot  # (B, n_bins)
    w_pos = w_pos_img @ onehot  # (B, n_bins)
    w_neg = neg_w_sorted.sum(axis=1)  # (B,)
    denom = w_pos * w_neg[:, None]
    ok = denom > 0
    auroc = np.full(denom.shape, np.nan)
    auroc[ok] = num[ok] / denom[ok]
    return auroc, w_pos


# --------------------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------------------
def _check_inputs(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y)
    if scores.ndim != 2 or y.shape != scores.shape:
        raise ValueError(f"scores {scores.shape} and y {y.shape} must both be (N, C)")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or inf")
    return scores, y


def per_class_auroc_from_bins(
    scores: np.ndarray, y: np.ndarray, bins: np.ndarray, n_bins: int, min_pos: int = MIN_POS
) -> dict:
    """Per-class, per-bin AUROC for an arbitrary ``(N, C)`` integer bin assignment.

    ``bins`` holds a bin index ``0..n_bins-1`` for every positive (any value for
    non-positives is ignored). Returns ``{c: {"bins": {b: {"auroc", "n_pos"}},
    "auroc_all", "n_pos_all", "n_neg"}}``; ``auroc`` is ``nan`` when the cell has
    fewer than ``min_pos`` positives or the class has no negatives. ``auroc_all``
    pools positives of every size and is ``nan`` only when a class has no positive
    or no negative (the rule :func:`overall_metrics` uses).
    """
    scores, y = _check_inputs(scores, y)
    bins = np.asarray(bins)
    if bins.shape != scores.shape:
        raise ValueError(f"bins {bins.shape} must match scores {scores.shape}")
    n, n_classes = scores.shape
    out: dict = {}
    for c in range(n_classes):
        pos = y[:, c] == 1
        neg = y[:, c] == 0
        s_neg = scores[neg, c]
        n_neg = int(neg.sum())
        n_pos_all = int(pos.sum())
        per_bin: dict = {}
        for b in range(n_bins):
            sel = pos & (bins[:, c] == b)
            n_pos = int(sel.sum())
            auroc = float("nan")
            if n_pos >= min_pos and n_neg >= 1:
                s_pos = scores[sel, c]
                yy = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)])
                auroc = float(roc_auc_score(yy, np.concatenate([s_pos, s_neg])))
            per_bin[b] = {"auroc": auroc, "n_pos": n_pos}
        auroc_all = float("nan")
        if n_pos_all >= 1 and n_neg >= 1:
            auroc_all = float(roc_auc_score(y[:, c] == 1, scores[:, c]))
        out[c] = {"bins": per_bin, "auroc_all": auroc_all, "n_pos_all": n_pos_all, "n_neg": n_neg}
    return out


def per_class_bin_auroc(
    scores: np.ndarray, y: np.ndarray, max_box_frac: np.ndarray, min_pos: int = MIN_POS
) -> dict:
    """Per-class, per-size-bin AUROC (docs/CONTRACTS.md "Metric").

    Parameters
    ----------
    scores
        ``(N, 80)`` float per-image per-class scores (higher = more present).
    y
        ``(N, 80)`` uint8 presence labels.
    max_box_frac
        ``(N, 80)`` relative bbox area of the largest instance (``0`` when absent).
    min_pos
        Cells with fewer positives get ``auroc = nan`` (``n_pos`` is still reported).

    Returns
    -------
    dict
        ``{c: {"bins": {b: {"auroc": float, "n_pos": int}}, "auroc_all": float,
        "n_pos_all": int, "n_neg": int}}`` for ``c`` in ``0..79`` and ``b`` in ``0..5``.
    """
    bins = assign_bins(max_box_frac)
    return per_class_auroc_from_bins(scores, y, bins, N_BINS, min_pos=min_pos)


def macro_curve(per_class_bin: dict, min_pos: int = MIN_POS) -> dict:
    """Macro-average the output of :func:`per_class_bin_auroc` over classes.

    For each bin: ``auroc_macro`` = mean AUROC over classes with ``n_pos >= min_pos``
    and a finite AUROC (``nan`` if none), ``n_classes`` = number of such classes,
    ``n_pos`` = total positives across those classes. Keys are bin indices.
    """
    if not per_class_bin:
        return {}
    n_bins = len(next(iter(per_class_bin.values()))["bins"])
    out: dict = {}
    for b in range(n_bins):
        vals: list[float] = []
        n_pos = 0
        for entry in per_class_bin.values():
            cell = entry["bins"][b]
            if cell["n_pos"] >= min_pos and np.isfinite(cell["auroc"]):
                vals.append(cell["auroc"])
                n_pos += int(cell["n_pos"])
        out[b] = {
            "auroc_macro": float(np.mean(vals)) if vals else float("nan"),
            "n_classes": len(vals),
            "n_pos": n_pos,
        }
    return out


def overall_metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    """All sizes pooled: macro AUROC and mAP over classes with >= 1 positive and >= 1 negative."""
    scores, y = _check_inputs(scores, y)
    aurocs: list[float] = []
    aps: list[float] = []
    for c in range(scores.shape[1]):
        yy = (y[:, c] == 1)
        if 0 < yy.sum() < yy.size:
            aurocs.append(float(roc_auc_score(yy, scores[:, c])))
            aps.append(float(average_precision_score(yy, scores[:, c])))
    return {
        "auroc_macro": float(np.mean(aurocs)) if aurocs else float("nan"),
        "map": float(np.mean(aps)) if aps else float("nan"),
        "n_classes": len(aurocs),
    }


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------
def bootstrap_weights(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_boot, n)`` multiplicities from resampling ``n`` images with replacement."""
    idx = rng.integers(0, n, size=(n_boot, n))
    flat = (np.arange(n_boot, dtype=np.int64)[:, None] * n + idx).ravel()
    return np.bincount(flat, minlength=n_boot * n).reshape(n_boot, n).astype(np.float64)


def bootstrap_from_bins(
    scores: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
    n_bins: int,
    n_boot: int = N_BOOT,
    seed: int = 0,
    min_pos: int = MIN_POS,
    return_draws: bool = False,
) -> dict | tuple[dict, np.ndarray]:
    """Image-level bootstrap of the macro curve for an arbitrary bin assignment.

    The set of contributing classes per bin is fixed from the full data (``n_pos >=
    min_pos`` and at least one negative), exactly as in :func:`macro_curve`; each
    draw re-computes those classes' AUROCs on the resampled images and averages
    them (a class whose resample happens to contain no positive in the bin is
    skipped in that draw). Returns ``{b: {"ci_lo", "ci_hi"}}`` with the 2.5 / 97.5
    percentiles; with ``return_draws`` also the ``(n_boot, n_bins)`` macro draws.
    """
    scores, y = _check_inputs(scores, y)
    bins = np.asarray(bins)
    n, n_classes = scores.shape
    rng = np.random.default_rng(seed)
    w = bootstrap_weights(n, n_boot, rng)

    draws = np.full((n_boot, n_classes, n_bins), np.nan)
    for c in range(n_classes):
        pos = y[:, c] == 1
        neg = y[:, c] == 0
        if neg.sum() == 0 or pos.sum() == 0:
            continue
        pos_bins = np.where(pos, bins[:, c], -1).astype(np.int64)
        n_pos = np.bincount(pos_bins[pos_bins >= 0], minlength=n_bins)[:n_bins]
        eligible = n_pos >= min_pos
        if not eligible.any():
            continue
        auroc, _ = _weighted_auroc_by_bin(scores[:, c], w, neg, pos_bins, n_bins)
        auroc[:, ~eligible] = np.nan
        draws[:, c, :] = auroc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-nan slices
        macro = np.nanmean(draws, axis=1)  # (n_boot, n_bins)
        lo = np.nanpercentile(macro, 2.5, axis=0)
        hi = np.nanpercentile(macro, 97.5, axis=0)
    out = {b: {"ci_lo": float(lo[b]), "ci_hi": float(hi[b])} for b in range(n_bins)}
    return (out, macro) if return_draws else out


def bootstrap_macro_curve(
    scores: np.ndarray,
    y: np.ndarray,
    max_box_frac: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 0,
    min_pos: int = MIN_POS,
) -> dict:
    """95% bootstrap CI of the per-bin macro AUROC by resampling images.

    Returns ``{b: {"ci_lo": float, "ci_hi": float}}`` for ``b`` in ``0..5``. Seeded
    with ``numpy.random.default_rng(seed)``. See :func:`bootstrap_from_bins` for the
    exact resampling rule and the speed trick.
    """
    bins = assign_bins(max_box_frac)
    return bootstrap_from_bins(scores, y, bins, N_BINS, n_boot=n_boot, seed=seed, min_pos=min_pos)
