"""Per-class analysis on small targets: what pooling loses and what the head recovers.

"Small" pools the two smallest size bins (largest instance below 0.3% of the
image, sub-patch at 224px input). For each class and read-out the small-target
AUROC uses the positives in those bins against *all* negatives of the class
(shared negatives, as in the per-bin metric) and is reported when the class has
at least ``min_pos`` such positives. From these per-class numbers:

* ``oracle_gap``    = oracle - mean: what a global mean vector loses relative to
  reading only the patches inside the box (the evidence lost in pooling);
* ``head_gain``     = attn - mean: what the learned attention head recovers;
* ``context_score`` = mean-probe small-target AUROC: how detectable the class is
  from a global vector that barely contains it (high = the rest of the image
  gives it away).

Small-target AUROCs can be recomputed from saved scores or pooled from the
``per_class_bin`` block of a results JSON: because negatives are shared across
bins, the AUROC of the pooled positives is exactly the ``n_pos``-weighted mean of
the per-bin AUROCs (:func:`small_target_auroc_from_results`), which only fails
when a non-empty small bin has fewer than ``min_pos`` positives and hence no
stored AUROC.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text

from lip.eval.bins import BIN_EDGES, BIN_NAMES, assign_bins
from lip.eval.metrics import MIN_POS, _check_inputs, auroc_rank
from lip.eval.plots import _AXIS, _GRID, _INK, _MUTED, _latex_escape

#: Indices of the bins pooled as "small" (``<0.1%`` and ``0.1-0.3%``).
SMALL_BINS: tuple[int, ...] = (0, 1)
#: Upper edge of the pooled small range, as a fraction of the image.
SMALL_MAX_FRAC: float = BIN_EDGES[SMALL_BINS[-1] + 1]
#: Label of the pooled small range.
SMALL_LABEL: str = "<0.3%"
#: Read-outs of the per-class analysis, in display order.
PERCLASS_READOUTS: tuple[str, ...] = ("cls", "mean", "attn", "oracle")
#: Colour of the single scatter series (categorical slot 1 of the validated palette).
_POINT = "#2a78d6"


# --------------------------------------------------------------------------------------
# Small-target AUROC per class
# --------------------------------------------------------------------------------------
def small_target_auroc(
    scores: np.ndarray,
    y: np.ndarray,
    max_box_frac: np.ndarray,
    min_pos: int = MIN_POS,
    small_bins: tuple[int, ...] = SMALL_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class AUROC of the positives in ``small_bins`` (pooled) against all negatives.

    Returns ``(auroc, n_small)``: ``(C,)`` float with ``nan`` where the class has
    fewer than ``min_pos`` small positives or no negatives, and ``(C,)`` int counts.
    """
    scores, y = _check_inputs(scores, y)
    max_box_frac = np.asarray(max_box_frac)
    if max_box_frac.shape != y.shape:
        raise ValueError(f"max_box_frac {max_box_frac.shape} must match y {y.shape}")
    bins = assign_bins(max_box_frac)
    small = (y == 1) & np.isin(bins, list(small_bins))
    n_small = small.sum(axis=0).astype(int)
    auroc = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        neg = y[:, c] == 0
        if n_small[c] >= min_pos and neg.any():
            s = np.concatenate([scores[small[:, c], c], scores[neg, c]])
            lab = np.concatenate([np.ones(n_small[c], dtype=bool), np.zeros(int(neg.sum()), dtype=bool)])
            auroc[c] = auroc_rank(s, lab)
    return auroc, n_small


def small_target_auroc_from_results(
    result: dict,
    cat_names: list[str],
    min_pos: int = MIN_POS,
    small_bins: tuple[int, ...] = SMALL_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool the small bins of a results JSON's ``per_class_bin`` block (see module doc).

    Returns ``(auroc, n_small)`` like :func:`small_target_auroc`; ``auroc`` is
    ``nan`` when the pooled count is below ``min_pos`` or any non-empty small bin
    has no finite stored AUROC (it had fewer than the JSON's ``min_pos``).
    """
    names = [BIN_NAMES[b] for b in small_bins]
    pcb = result.get("per_class_bin", {})
    auroc = np.full(len(cat_names), np.nan)
    n_small = np.zeros(len(cat_names), dtype=int)
    for c, name in enumerate(cat_names):
        cells = pcb.get(str(name))
        if cells is None:
            continue
        n_small[c] = int(sum(int(cells[b]["n_pos"]) for b in names))
        if n_small[c] < min_pos:
            continue
        num = 0.0
        ok = True
        for b in names:
            n_pos = int(cells[b]["n_pos"])
            if n_pos == 0:
                continue
            a = cells[b]["auroc"]
            if a is None or not math.isfinite(float(a)):
                ok = False
                break
            num += float(a) * n_pos
        if ok:
            auroc[c] = num / n_small[c]
    return auroc, n_small


# --------------------------------------------------------------------------------------
# Rows, summary and extremes
# --------------------------------------------------------------------------------------
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def perclass_rows(
    small_auroc: dict[str, np.ndarray],
    n_small: np.ndarray,
    cat_names: list[str],
    n_pos_all: np.ndarray | None = None,
    min_pos: int = MIN_POS,
    readouts: tuple[str, ...] = PERCLASS_READOUTS,
) -> list[dict]:
    """One row per class: small-target AUROC per read-out, gap, gain, context score.

    Parameters
    ----------
    small_auroc
        ``{readout: (C,) AUROC}`` from :func:`small_target_auroc` (missing
        read-outs give ``nan`` columns).
    n_small
        ``(C,)`` small positives per class (identical for every read-out).
    cat_names
        ``(C,)`` class names.
    n_pos_all
        Optional ``(C,)`` positives of any size, copied into the rows.
    min_pos, readouts
        Eligibility threshold and the read-out columns, in order.

    Returns
    -------
    list[dict]
        Keys ``class, n_small, n_pos_all, <readout>..., oracle_gap, head_gain,
        context_score, eligible`` (eligible = ``n_small >= min_pos`` and finite
        mean and oracle AUROC).
    """
    n_small = np.asarray(n_small).astype(int)
    rows: list[dict] = []
    for c, name in enumerate(cat_names):
        row: dict = {"class": str(name), "n_small": int(n_small[c]),
                     "n_pos_all": int(n_pos_all[c]) if n_pos_all is not None else None}
        for r in readouts:
            v = small_auroc.get(r)
            row[r] = float(v[c]) if v is not None else float("nan")
        row["oracle_gap"] = row.get("oracle", float("nan")) - row.get("mean", float("nan"))
        row["head_gain"] = row.get("attn", float("nan")) - row.get("mean", float("nan"))
        row["context_score"] = row.get("mean", float("nan"))
        row["eligible"] = bool(n_small[c] >= min_pos and math.isfinite(row["oracle_gap"]))
        rows.append(row)
    return rows


def extreme_classes(rows: list[dict], k: int = 8, key: str = "oracle_gap") -> tuple[list[dict], list[dict]]:
    """The ``k`` eligible rows with the largest ``key`` and the ``k`` with the smallest.

    Both lists are sorted by ``key`` (descending for the first, ascending for the
    second) and never overlap: with fewer than ``2k`` eligible rows the second
    list is shortened.
    """
    elig = [r for r in rows if r["eligible"] and math.isfinite(r[key])]
    desc = sorted(elig, key=lambda r: (-r[key], r["class"]))
    largest = desc[:k]
    taken = {r["class"] for r in largest}
    smallest = sorted([r for r in elig if r["class"] not in taken], key=lambda r: (r[key], r["class"]))[:k]
    return largest, smallest


def perclass_summary(rows: list[dict], k: int = 8) -> dict:
    """Aggregate numbers over eligible classes (means, rank correlations, extremes)."""
    elig = [r for r in rows if r["eligible"]]
    gap = np.array([r["oracle_gap"] for r in elig], dtype=np.float64)
    gain = np.array([r["head_gain"] for r in elig], dtype=np.float64)
    ctx = np.array([r["context_score"] for r in elig], dtype=np.float64)
    largest, smallest = extreme_classes(rows, k=k)

    def _m(v: np.ndarray, fn) -> float:
        v = v[np.isfinite(v)]
        return float(fn(v)) if v.size else float("nan")

    return {
        "n_classes_eligible": len(elig),
        "n_classes_total": len(rows),
        "oracle_gap_mean": _m(gap, np.mean),
        "oracle_gap_median": _m(gap, np.median),
        "head_gain_mean": _m(gain, np.mean),
        "head_gain_median": _m(gain, np.median),
        "context_score_mean": _m(ctx, np.mean),
        "frac_head_gain_positive": _m(gain, lambda v: np.mean(v > 0)),
        "spearman_context_vs_gap": _spearman(ctx, gap),
        "spearman_context_vs_head_gain": _spearman(ctx, gain),
        "spearman_gap_vs_head_gain": _spearman(gap, gain),
        "largest_gap": [r["class"] for r in largest],
        "smallest_gap": [r["class"] for r in smallest],
    }


# --------------------------------------------------------------------------------------
# Tables and CSV
# --------------------------------------------------------------------------------------
_TABLE_COLS: tuple[str, ...] = ("class", "n_small", "mean", "attn", "oracle", "oracle_gap")


def _fmt(v: float, signed: bool = False) -> str:
    if v is None or not math.isfinite(v):
        return "--"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def perclass_table_latex(largest: list[dict], smallest: list[dict]) -> str:
    """Booktabs tabular (no ``table`` environment): the extreme classes by oracle gap.

    Columns: class, number of small positives, mean, attn and oracle small-target
    AUROC, gap (oracle minus mean). The two blocks are separated by a rule and a
    one-line italic heading.
    """
    header = " & ".join(["class", r"$n_{\mathrm{small}}$", "mean", "attn", "oracle", "gap"])
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", header + r" \\", r"\midrule"]
    for heading, block in (("largest oracle gap", largest), ("smallest oracle gap", smallest)):
        lines.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{heading}}}}} \\")
        for r in block:
            cells = [_latex_escape(r["class"]), str(r["n_small"]), _fmt(r["mean"]), _fmt(r["attn"]),
                     _fmt(r["oracle"]), _fmt(r["oracle_gap"], signed=True)]
            lines.append(" & ".join(cells) + r" \\")
        if block is largest:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def perclass_table_text(largest: list[dict], smallest: list[dict]) -> str:
    """Plain-text version of :func:`perclass_table_latex` for stdout."""
    name_w = max([6] + [len(r["class"]) for r in largest + smallest])
    head = "class".ljust(name_w) + "".join(c.rjust(9) for c in ("n_small", "mean", "attn", "oracle", "gap"))
    lines = [head, "-" * len(head)]
    for heading, block in (("largest oracle gap", largest), ("smallest oracle gap", smallest)):
        lines.append(f"[{heading}]")
        for r in block:
            lines.append(r["class"].ljust(name_w) + str(r["n_small"]).rjust(9)
                         + "".join(_fmt(r[k]).rjust(9) for k in ("mean", "attn", "oracle"))
                         + _fmt(r["oracle_gap"], signed=True).rjust(9))
    return "\n".join(lines) + "\n"


def perclass_dataframe(rows: list[dict], readouts: tuple[str, ...] = PERCLASS_READOUTS) -> pd.DataFrame:
    """Rows as a DataFrame (one per class) with ``small_auroc_<readout>`` columns, for CSV."""
    cols = ["class", "n_small", "n_pos_all"] + list(readouts) + ["oracle_gap", "head_gain", "context_score", "eligible"]
    df = pd.DataFrame([{k: r.get(k) for k in cols} for r in rows])
    return df.rename(columns={r: f"small_auroc_{r}" for r in readouts})


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
def _box_overlap(a: tuple, b: tuple) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _place_labels(
    ax, fig, pts: list[tuple[float, float, str]], others_xy: np.ndarray, others_s: np.ndarray,
    fontsize: float, pad: float = 2.0,
) -> None:
    """Annotate ``pts`` with greedy collision-free offsets, using rendered text extents.

    Candidate offsets (in points) are tried in order and scored by the overlap of
    the padded text box with labels already placed, with the markers of the other
    points (``others_xy`` in data units, ``others_s`` marker areas in pt^2) and
    with the region outside the axes; the first zero-score candidate wins, else
    the lowest score. The figure must have been drawn once so the transforms
    and renderer are final.
    """
    renderer = fig.canvas.get_renderer()
    ab = ax.get_window_extent(renderer)
    ax_box = (ab.x0, ab.y0, ab.x1, ab.y1)
    px_per_pt = fig.dpi / 72.0
    cands = [(6, -8), (6, 6), (-6, -8), (-6, 6), (9, -1), (-9, -1), (0, -11), (0, 9),
             (14, -16), (-14, -16), (14, 12), (-14, 12), (22, -6), (-22, -6),
             (0, -24), (-20, -26), (20, -26), (-34, -14), (34, -14), (-40, -30), (0, -38)]
    if len(others_xy):
        o_px = ax.transData.transform(others_xy)
        o_r = np.sqrt(np.asarray(others_s, dtype=np.float64) / math.pi) * px_per_pt
        marker_boxes = [(px - r, py - r, px + r, py + r) for (px, py), r in zip(o_px, o_r)]
    else:
        marker_boxes = []
    placed: list[tuple] = []
    arrow = {"arrowstyle": "-", "linewidth": 0.5, "color": _MUTED, "shrinkA": 0, "shrinkB": 2.5}
    for x, y, txt in pts:
        best, best_box, best_score = None, None, float("inf")
        for dx, dy in cands:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
            ann = ax.annotate(txt, xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                              fontsize=fontsize, color=_INK, zorder=5, arrowprops=arrow)
            ann.update_positions(renderer)
            bb = Text.get_window_extent(ann, renderer)  # text only, without the leader line
            box = (bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
            # Text over text is unreadable, text over a marker merely ugly: weight them apart.
            score = 10.0 * sum(_box_overlap(box, p) for p in placed) + sum(_box_overlap(box, m) for m in marker_boxes)
            area = (box[2] - box[0]) * (box[3] - box[1])
            score += 3.0 * (area - _box_overlap(box, ax_box))  # leaving the axes
            if score < best_score:
                if best is not None:
                    best.remove()
                best, best_box, best_score = ann, box, score
            else:
                ann.remove()
            if score == 0:
                break
        placed.append(best_box)


def plot_perclass_scatter(
    rows: list[dict],
    out_path: Path | str,
    label_classes: list[str] | None = None,
    n_label: tuple[int, int] = (4, 2),
    figsize: tuple[float, float] = (3.4, 3.0),
    title: str | None = None,
) -> list[Path]:
    """Scatter of mean-probe vs. oracle small-target AUROC per class; saves ``.png`` + ``.pdf``.

    Parameters
    ----------
    rows
        Output of :func:`perclass_rows`; only eligible rows are drawn.
    out_path
        Output path; its suffix is replaced by ``.png`` (dpi 200) and ``.pdf``.
    label_classes
        Classes to annotate; default: the ``n_label[0]`` largest and
        ``n_label[1]`` smallest oracle gaps.
    figsize
        Sized for one LaTeX column (about 3.3in wide).
    title
        Optional axes title.
    """
    elig = [r for r in rows if r["eligible"]]
    if not elig:
        raise ValueError("no eligible class to plot")
    x = np.array([r["context_score"] for r in elig], dtype=np.float64)
    y = np.array([r["oracle"] for r in elig], dtype=np.float64)
    n = np.array([r["n_small"] for r in elig], dtype=np.float64)
    names = [r["class"] for r in elig]
    if label_classes is None:
        largest, smallest = extreme_classes(rows, k=max(n_label))
        label_classes = [r["class"] for r in largest[: n_label[0]]] + [r["class"] for r in smallest[: n_label[1]]]

    n_max = float(n.max())
    size = 10.0 + 110.0 * n / n_max  # marker area proportional to n (plus a floor)
    lo = math.floor((min(float(x.min()), float(y.min())) - 0.02) * 20) / 20
    hi = 1.005

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)
    ax.plot([lo, hi], [lo, hi], color=_MUTED, linestyle="--", linewidth=0.9, zorder=1)
    ax.text(lo + 0.01, lo + 0.015, "oracle = mean", fontsize=6.5, color=_MUTED, rotation=0, ha="left", va="bottom")
    ax.scatter(x, y, s=size, color=_POINT, alpha=0.8, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"mean-probe AUROC, small targets ({SMALL_LABEL})", fontsize=8)
    ax.set_ylabel(f"oracle AUROC, small targets ({SMALL_LABEL})", fontsize=8)
    ax.tick_params(axis="both", labelsize=7, colors=_INK, length=3, width=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=8)

    # Size legend: three reference counts, hollow so they read as a key rather than data.
    ref = [v for v in (10, 50, 150) if v <= n_max] or [int(n_max)]
    handles = [ax.scatter([], [], s=10.0 + 110.0 * v / n_max, facecolor="none", edgecolor=_INK,
                          linewidth=0.6, label=f"n = {v}") for v in ref]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=6.5, title="small positives",
                    title_fontsize=6.5, frameon=False, labelspacing=0.9, handletextpad=0.6, borderpad=0.2)
    leg.set_zorder(4)

    # Draw once so transData reflects the constrained layout, then place labels.
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    idx = {nm: i for i, nm in enumerate(names)}
    pts = [(float(x[idx[c]]), float(y[idx[c]]), c) for c in label_classes if c in idx]
    labelled = {c for _, _, c in pts}
    keep = np.array([nm not in labelled for nm in names], dtype=bool)
    _place_labels(ax, fig, pts, np.column_stack([x[keep], y[keep]]), size[keep], fontsize=7.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png = out_path.with_suffix(".png")
    pdf = out_path.with_suffix(".pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    return [png, pdf]
