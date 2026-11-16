"""Aggregate seed repeats of one read-out: mean and std of the macro AUROC per bin.

Seed repeats are stored as ordinary results JSONs whose read-out name carries a
suffix (``attn``, ``attn_s1``, ``attn_s2``, ...). :func:`aggregate_seeds` takes
the loaded dicts and reports, per bin and overall, the mean and the sample
standard deviation (``ddof=1``) across them. With a single file the std is
reported as ``0`` and ``note`` says so, so downstream tables still render.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from lip.eval.bins import BIN_NAMES
from lip.eval.plots import _latex_bin_name, _latex_escape
from lip.eval.results import RESULTS_DIR, results_path


def seed_results_path(readout: str, suffix: str, backbone: str, res: int, results_dir: Path | str = RESULTS_DIR) -> Path:
    """``results/{readout}{suffix}__{backbone}__res{res}.json``."""
    return results_path(f"{readout}{suffix}", backbone, res, results_dir)


def _mean_std(values: list[float], ddof: int) -> tuple[float, float, int]:
    v = np.array([float("nan") if x is None else float(x) for x in values], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    std = float(np.std(v, ddof=ddof)) if v.size > ddof else 0.0
    return float(np.mean(v)), std, int(v.size)


def aggregate_seeds(
    results: list[dict], readout: str | None = None, files: list[str] | None = None, ddof: int = 1
) -> dict:
    """Mean / std across seed repeats of the per-bin macro AUROC and the overall numbers.

    Parameters
    ----------
    results
        Loaded results dicts of the same read-out family, backbone and resolution.
    readout
        Base read-out name recorded in the output (default: first result's name).
    files
        Optional file names recorded for provenance.
    ddof
        Delta degrees of freedom of the std (``1`` = sample std).

    Returns
    -------
    dict
        ``{"readout", "backbone", "res", "n_seeds", "seeds", "readouts", "files",
        "std_ddof", "note", "per_bin": {bin: {"auroc_macro_mean", "auroc_macro_std",
        "n", "values", "n_classes"}}, "overall": {"auroc_macro": {"mean", "std", "n",
        "values"}, "map": {...}}}``.
    """
    if not results:
        raise ValueError("no results to aggregate")
    backbones = {str(r.get("backbone")) for r in results}
    ress = {int(r.get("res", -1)) for r in results}
    if len(backbones) != 1 or len(ress) != 1:
        raise ValueError(f"seed repeats must share backbone and res, got {sorted(backbones)} x {sorted(ress)}")
    per_bin: dict = {}
    for name in BIN_NAMES:
        vals = [r["per_bin"].get(name, {}).get("auroc_macro") for r in results]
        mean, std, n = _mean_std(vals, ddof)
        per_bin[name] = {
            "auroc_macro_mean": mean, "auroc_macro_std": std, "n": n,
            "values": [None if v is None or not math.isfinite(float(v)) else float(v) for v in vals],
            "n_classes": results[0]["per_bin"].get(name, {}).get("n_classes"),
        }
    overall: dict = {}
    for key in ("auroc_macro", "map"):
        vals = [r.get("overall", {}).get(key) for r in results]
        mean, std, n = _mean_std(vals, ddof)
        overall[key] = {"mean": mean, "std": std, "n": n,
                        "values": [None if v is None or not math.isfinite(float(v)) else float(v) for v in vals]}
    n_seeds = len(results)
    note = (f"single seed: std reported as 0" if n_seeds == 1
            else f"sample std (ddof={ddof}) over {n_seeds} seed repeats")
    return {
        "readout": readout if readout is not None else str(results[0].get("readout")),
        "backbone": backbones.pop(),
        "res": ress.pop(),
        "n_seeds": n_seeds,
        "seeds": [r.get("seed") for r in results],
        "readouts": [str(r.get("readout")) for r in results],
        "files": list(files) if files is not None else None,
        "std_ddof": int(ddof),
        "note": note,
        "per_bin": per_bin,
        "overall": overall,
    }


def _pm(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "--"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def seeds_table_latex(agg: dict) -> str:
    """Tabular-only booktabs row of ``mean ± std`` cells: 6 bins, pooled AUROC, mAP."""
    header = " & ".join(["read-out"] + [_latex_bin_name(n) for n in BIN_NAMES] + ["all", "mAP"])
    cells = [_pm(agg["per_bin"][n]["auroc_macro_mean"], agg["per_bin"][n]["auroc_macro_std"]) for n in BIN_NAMES]
    cells += [_pm(agg["overall"]["auroc_macro"]["mean"], agg["overall"]["auroc_macro"]["std"]),
              _pm(agg["overall"]["map"]["mean"], agg["overall"]["map"]["std"])]
    label = _latex_escape(str(agg["readout"])) + f" ($k={agg['n_seeds']}$)"
    lines = [
        r"\begin{tabular}{l" + "c" * (len(BIN_NAMES) + 2) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        " & ".join([label] + cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def seeds_table_text(agg: dict) -> str:
    """Plain-text version of :func:`seeds_table_latex` for stdout."""
    cols = list(BIN_NAMES) + ["all", "mAP"]
    w = 15
    lines = ["read-out".ljust(14) + "".join(c.rjust(w) for c in cols)]
    lines.append("-" * (14 + w * len(cols)))
    vals = [(agg["per_bin"][n]["auroc_macro_mean"], agg["per_bin"][n]["auroc_macro_std"]) for n in BIN_NAMES]
    vals += [(agg["overall"]["auroc_macro"]["mean"], agg["overall"]["auroc_macro"]["std"]),
             (agg["overall"]["map"]["mean"], agg["overall"]["map"]["std"])]
    cells = ["--" if not math.isfinite(m) else f"{m:.3f} ± {s:.3f}" for m, s in vals]
    lines.append(f"{agg['readout']} (k={agg['n_seeds']})".ljust(14) + "".join(c.rjust(w) for c in cells))
    lines.append(f"note: {agg['note']}")
    return "\n".join(lines) + "\n"
