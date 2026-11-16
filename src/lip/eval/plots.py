"""Blindness-curve figure and per-bin tables from results JSON dicts.

Read-outs are always drawn and listed in the fixed order ``cls, mean, cls_mean, max,
maxmil, attn*, oracle`` (oracle last, in black). Colours are bound to read-out names
rather than to plotting order, so a read-out keeps its colour across figures; every
series also gets its own marker shape so identity never rests on colour alone.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lip.eval.bins import BIN_EDGES, BIN_NAMES, N_BINS, patch_area_frac
from lip.eval.metrics import expected_chance

#: Fixed display order of the named read-outs; ``attn*`` variants follow, oracle is last.
READOUT_ORDER: tuple[str, ...] = ("cls", "mean", "cls_mean", "max", "maxmil")
ORACLE = "oracle"

# Colour slots (validated colour-blind-safe categorical palette, in this adjacency order).
_FIXED_COLORS: dict[str, str] = {
    "cls": "#2a78d6",  # blue
    "mean": "#eb6834",  # orange
    "cls_mean": "#1baf7a",  # aqua
    "max": "#eda100",  # yellow
    "maxmil": "#e87ba4",  # magenta
    ORACLE: "#000000",
}
# Remaining validated slots go to attn* variants in sorted order; beyond that we fall
# back to two extra hues plus a dashed line style (rare: more than three attn variants).
_ATTN_COLORS: tuple[str, ...] = ("#008300", "#6250d6", "#e34948", "#8c564b", "#17becf")
_FIXED_MARKERS: dict[str, str] = {
    "cls": "o", "mean": "s", "cls_mean": "D", "max": "^", "maxmil": "v", ORACLE: "*",
}
_ATTN_MARKERS: tuple[str, ...] = ("P", "X", "p", "h", "<", ">")

# Chart chrome (ink / muted / hairline), from the design reference.
_INK = "#0b0b0b"
_MUTED = "#898781"
_AXIS = "#c3c2b7"
_GRID = "#e1e0d9"


# --------------------------------------------------------------------------------------
# Ordering and styling
# --------------------------------------------------------------------------------------
def readout_rank(name: str) -> tuple[int, str]:
    """Sort key implementing the fixed order (unknown names sit between attn* and oracle)."""
    if name in READOUT_ORDER:
        return (READOUT_ORDER.index(name), name)
    if name.startswith("attn"):
        return (len(READOUT_ORDER), name)
    if name == ORACLE:
        return (len(READOUT_ORDER) + 2, name)
    return (len(READOUT_ORDER) + 1, name)


def sort_results(results: list[dict]) -> list[dict]:
    """Results sorted by :func:`readout_rank` of their ``readout`` field."""
    return sorted(results, key=lambda r: readout_rank(str(r["readout"])))


def series_styles(names: list[str]) -> dict[str, dict]:
    """Colour / marker / linestyle per read-out name (stable across figures)."""
    extra = sorted(n for n in set(names) if n not in _FIXED_COLORS)
    styles: dict[str, dict] = {}
    for n in names:
        if n in _FIXED_COLORS:
            styles[n] = {"color": _FIXED_COLORS[n], "marker": _FIXED_MARKERS[n], "ls": "-"}
        else:
            k = extra.index(n)
            styles[n] = {
                "color": _ATTN_COLORS[k % len(_ATTN_COLORS)],
                "marker": _ATTN_MARKERS[k % len(_ATTN_MARKERS)],
                "ls": "-" if k < len(_ATTN_COLORS) else "--",
            }
    return styles


def _bin_values(r: dict, key: str) -> np.ndarray:
    vals = [r["per_bin"][name].get(key) for name in BIN_NAMES]
    return np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)


def one_patch_boundary(res: int) -> int:
    """Index ``k`` in ``1..5`` of the bin edge nearest (in log area) to one patch at ``res``.

    The boundary between bin ``k-1`` and bin ``k`` sits at ``x = k - 0.5`` on the
    categorical axis.
    """
    target = math.log(patch_area_frac(res))
    interior = BIN_EDGES[1:-1]
    k = int(np.argmin([abs(math.log(e) - target) for e in interior]))
    return k + 1


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
def plot_blindness_curves(
    results: list[dict], out_path: Path | str, res: int = 224, title: str | None = None
) -> list[Path]:
    """Draw macro AUROC vs. size bin for every read-out and save ``.png`` + ``.pdf``.

    Parameters
    ----------
    results
        Results dicts (as written by ``lip.eval.results``), any order.
    out_path
        Output path; its suffix is replaced by ``.png`` (dpi 200) and ``.pdf``.
    res
        Input resolution used to place the dotted "one patch" marker.
    title
        Optional axes title.

    Returns
    -------
    list[Path]
        The two written files.
    """
    if not results:
        raise ValueError("no results to plot")
    results = sort_results(results)
    names = [str(r["readout"]) for r in results]
    styles = series_styles(names)
    x = np.arange(N_BINS)

    fig = Figure(figsize=(6.5, 4.0), layout="constrained")
    ax = fig.add_subplot(111)

    y_lo_data = 1.0
    for r in results:
        name = str(r["readout"])
        st = styles[name]
        ym = _bin_values(r, "auroc_macro")
        lo = _bin_values(r, "ci_lo")
        hi = _bin_values(r, "ci_hi")
        if np.isfinite(lo).any():
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.14, linewidth=0, zorder=1)
            y_lo_data = min(y_lo_data, float(np.nanmin(lo)))
        if np.isfinite(ym).any():
            y_lo_data = min(y_lo_data, float(np.nanmin(ym)))
        # Filled plus/cross/star glyphs are thin, so they get a larger size and no ring.
        thin = st["marker"] in ("P", "X", "*")
        ax.plot(
            x, ym, color=st["color"], linestyle=st["ls"], marker=st["marker"], linewidth=2.0,
            markersize=7.5 if thin else 5.5, markeredgecolor=st["color"] if thin else "white",
            markeredgewidth=0.0 if thin else 0.8, label=name, zorder=3 if name == ORACLE else 2,
        )

    # Axis limits: leave room for the legend in the lower right.
    y_lo = math.floor(min(0.4, y_lo_data - 0.03) * 20) / 20
    y_hi = 1.02
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.4, N_BINS - 0.6)

    # Reference line at chance.
    ax.axhline(expected_chance, color=_MUTED, linestyle="--", linewidth=1.0, zorder=0)
    ax.text(-0.35, expected_chance + 0.005, "chance", ha="left", va="bottom", fontsize=8, color=_MUTED)

    # Dotted line at the bin boundary closest to one patch token.
    k = one_patch_boundary(res)
    xv = k - 0.5
    ax.axvline(xv, color=_MUTED, linestyle=":", linewidth=1.0, zorder=0)
    ax.text(xv, 1.0, "one patch", ha="center", va="bottom", fontsize=8, color=_MUTED,
            transform=ax.get_xaxis_transform(), clip_on=False)

    # Categorical x axis with light n_classes annotations under the tick labels.
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_NAMES, fontsize=9)
    n_classes = np.array([_bin_values(r, "n_classes") for r in results])
    for b in range(N_BINS):
        col = n_classes[:, b]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        lo_n, hi_n = int(col.min()), int(col.max())
        txt = f"n={lo_n}" if lo_n == hi_n else f"n={lo_n}–{hi_n}"
        ax.annotate(txt, xy=(b, 0), xycoords=("data", "axes fraction"), xytext=(0, -18),
                    textcoords="offset points", ha="center", va="top", fontsize=7, color=_MUTED,
                    annotation_clip=False)
    ax.set_xlabel("relative area of largest instance (bbox / image)", fontsize=9, labelpad=16)
    ax.set_ylabel("macro AUROC", fontsize=9)
    ax.tick_params(axis="both", labelsize=8, colors=_INK, length=3, width=0.8)

    # Recessive chrome.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10)

    leg = ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=1.0,
                    edgecolor="none", facecolor="white", borderpad=0.6, handlelength=2.2)
    leg.set_zorder(4)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png = out_path.with_suffix(".png")
    pdf = out_path.with_suffix(".pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    return [png, pdf]


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def _cell_values(results: list[dict]) -> np.ndarray:
    """``(n_readouts, 7)`` per-bin macro AUROC plus the pooled ``all`` column."""
    rows = []
    for r in results:
        all_v = r.get("overall", {}).get("auroc_macro")
        rows.append(np.append(_bin_values(r, "auroc_macro"), np.nan if all_v is None else float(all_v)))
    return np.array(rows, dtype=np.float64).reshape(len(results), N_BINS + 1)


def _best_mask(results: list[dict], vals: np.ndarray) -> np.ndarray:
    """Bold mask: best rounded value per column, excluding the oracle row."""
    names = [str(r["readout"]) for r in results]
    rounded = np.round(vals, 3)
    cand = rounded.copy()
    for i, n in enumerate(names):
        if n == ORACLE:
            cand[i] = np.nan
    best = np.full(vals.shape, False)
    for j in range(vals.shape[1]):
        col = cand[:, j]
        if np.isfinite(col).any():
            best[:, j] = col == np.nanmax(col)
    return best


def _latex_escape(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _latex_bin_name(name: str) -> str:
    name = name.replace("%", r"\%")
    if name.startswith("<"):
        return "$<$" + name[1:]
    if name.startswith(">"):
        return "$>$" + name[1:]
    return name.replace("-", "--")


def results_table_latex(
    results: list[dict], caption: str = "Macro AUROC per size bin.", label: str = "tab:blindness",
    floating: bool = True,
) -> str:
    """Booktabs table: rows = read-outs (fixed order), columns = 6 bins + ``all``.

    Cells are AUROC to 3 decimals; the best value per column among non-oracle rows
    is bold; missing cells are ``--``.
    """
    results = sort_results(results)
    vals = _cell_values(results)
    best = _best_mask(results, vals)
    header = " & ".join(["read-out"] + [_latex_bin_name(n) for n in BIN_NAMES] + ["all"])
    lines = ([
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ] if floating else []) + [
        r"\begin{tabular}{l" + "c" * (N_BINS + 1) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    oracle_rows = [i for i, r in enumerate(results) if str(r["readout"]) == ORACLE]
    for i, r in enumerate(results):
        if oracle_rows and i == oracle_rows[0]:
            lines.append(r"\midrule")
        cells = []
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                cells.append("--")
            else:
                s = f"{v:.3f}"
                cells.append(rf"\textbf{{{s}}}" if best[i, j] else s)
        lines.append(" & ".join([_latex_escape(str(r["readout"]))] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"] + ([r"\end{table}"] if floating else [])
    return "\n".join(lines) + "\n"


def results_table_text(results: list[dict]) -> str:
    """Plain-text version of :func:`results_table_latex` for stdout (``*`` marks the best)."""
    results = sort_results(results)
    vals = _cell_values(results)
    best = _best_mask(results, vals)
    cols = list(BIN_NAMES) + ["all"]
    name_w = max([8] + [len(str(r["readout"])) for r in results])
    col_w = max(8, max(len(c) for c in cols) + 1)
    lines = ["read-out".ljust(name_w) + "".join(c.rjust(col_w) for c in cols)]
    lines.append("-" * (name_w + col_w * len(cols)))
    for i, r in enumerate(results):
        cells = []
        for j in range(vals.shape[1]):
            v = vals[i, j]
            s = "--" if not np.isfinite(v) else f"{v:.3f}" + ("*" if best[i, j] else " ")
            cells.append(s.rjust(col_w))
        lines.append(str(r["readout"]).ljust(name_w) + "".join(cells))
    return "\n".join(lines) + "\n"
