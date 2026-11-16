"""Evaluate a read-out's scores into the results JSON of docs/CONTRACTS.md, and I/O helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from lip.eval.bins import BIN_EDGES, BIN_NAMES, N_BINS, SML_NAMES, coco_sml
from lip.eval.metrics import (
    MIN_POS,
    N_BOOT,
    bootstrap_from_bins,
    bootstrap_macro_curve,
    macro_curve,
    overall_metrics,
    per_class_auroc_from_bins,
    per_class_bin_auroc,
)

RESULTS_DIR = Path("results")


def results_path(readout: str, backbone: str, res: int, results_dir: Path | str = RESULTS_DIR) -> Path:
    """``results/{readout}__{backbone}__res{res}.json``."""
    return Path(results_dir) / f"{readout}__{backbone}__res{res}.json"


def scores_path(readout: str, backbone: str, res: int, results_dir: Path | str = RESULTS_DIR) -> Path:
    """``results/scores/{readout}__{backbone}__res{res}.npy``."""
    return Path(results_dir) / "scores" / f"{readout}__{backbone}__res{res}.npy"


def _bin_block(curve: dict, ci: dict, names: list[str]) -> dict:
    return {
        names[b]: {
            "auroc_macro": curve[b]["auroc_macro"],
            "ci_lo": ci[b]["ci_lo"],
            "ci_hi": ci[b]["ci_hi"],
            "n_classes": curve[b]["n_classes"],
            "n_pos": curve[b]["n_pos"],
        }
        for b in range(len(names))
    }


def evaluate_readout(
    scores: np.ndarray,
    labels: dict,
    readout: str,
    backbone: str,
    res: int,
    split: str = "test",
    train_split: str = "train",
    notes: str = "",
    n_boot: int = N_BOOT,
    width: np.ndarray | None = None,
    height: np.ndarray | None = None,
    min_pos: int = MIN_POS,
    seed: int = 0,
) -> dict:
    """Compute every number of the results JSON for one read-out.

    Parameters
    ----------
    scores
        ``(N, 80)`` per-image per-class scores in the order of ``labels["image_ids"]``.
    labels
        Dict from ``lip.data.labels.load_labels(split=...)`` with keys ``y``,
        ``max_box_frac``, ``max_seg_frac``, ``cat_names``, ``image_ids``.
    readout, backbone, res, split, train_split, notes
        Metadata copied into the JSON.
    n_boot
        Bootstrap draws for the 95% CI (``0`` disables it; CIs become ``nan``).
    width, height
        Optional ``(N,)`` original pixel sizes. When both are given a ``coco_sml``
        block (small/medium/large macro AUROC from ``max_seg_frac``) is added.
    min_pos, seed
        Macro-average threshold and bootstrap seed; both are recorded.

    Returns
    -------
    dict
        JSON-ready results (``nan`` values are written as ``null`` by :func:`save_results`).
    """
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels["y"])
    max_box_frac = np.asarray(labels["max_box_frac"])
    cat_names = [str(n) for n in labels["cat_names"]]
    n_images = scores.shape[0]
    if scores.shape != y.shape or max_box_frac.shape != y.shape:
        raise ValueError(
            f"shape mismatch: scores {scores.shape}, y {y.shape}, max_box_frac {max_box_frac.shape}"
        )
    if len(cat_names) != scores.shape[1]:
        raise ValueError(f"{len(cat_names)} category names for {scores.shape[1]} classes")
    if "split" in labels:
        found = set(np.asarray(labels["split"]).tolist())
        if found and found != {split}:
            raise ValueError(f"labels are for split(s) {sorted(found)}, not {split!r}")

    pcb = per_class_bin_auroc(scores, y, max_box_frac, min_pos=min_pos)
    curve = macro_curve(pcb, min_pos=min_pos)
    if n_boot > 0:
        ci = bootstrap_macro_curve(scores, y, max_box_frac, n_boot=n_boot, seed=seed, min_pos=min_pos)
    else:
        ci = {b: {"ci_lo": float("nan"), "ci_hi": float("nan")} for b in range(N_BINS)}
    overall = overall_metrics(scores, y)

    out: dict = {
        "readout": readout,
        "backbone": backbone,
        "res": int(res),
        "split": split,
        "n_images": int(n_images),
        "train_split": train_split,
        "notes": notes,
        "seed": int(seed),
        "min_pos": int(min_pos),
        "n_boot": int(n_boot),
        "bins": {"edges": list(BIN_EDGES), "names": list(BIN_NAMES)},
        "per_bin": _bin_block(curve, ci, BIN_NAMES),
        "overall": {"auroc_macro": overall["auroc_macro"], "map": overall["map"]},
        "per_class_bin": {
            cat_names[c]: {
                BIN_NAMES[b]: {"auroc": pcb[c]["bins"][b]["auroc"], "n_pos": pcb[c]["bins"][b]["n_pos"]}
                for b in range(N_BINS)
            }
            for c in range(len(cat_names))
        },
        "per_class_all": {
            cat_names[c]: {"auroc": pcb[c]["auroc_all"], "n_pos": pcb[c]["n_pos_all"]}
            for c in range(len(cat_names))
        },
        "coco_sml": None,
    }

    if width is not None and height is not None:
        sml = coco_sml(np.asarray(labels["max_seg_frac"]), width, height)
        pcb_sml = per_class_auroc_from_bins(scores, y, sml, len(SML_NAMES), min_pos=min_pos)
        curve_sml = macro_curve(pcb_sml, min_pos=min_pos)
        if n_boot > 0:
            ci_sml = bootstrap_from_bins(
                scores, y, sml, len(SML_NAMES), n_boot=n_boot, seed=seed, min_pos=min_pos
            )
        else:
            ci_sml = {b: {"ci_lo": float("nan"), "ci_hi": float("nan")} for b in range(len(SML_NAMES))}
        out["coco_sml"] = _bin_block(curve_sml, ci_sml, SML_NAMES)
    return out


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def _jsonable(obj):
    """Recursively convert numpy scalars/arrays and map non-finite floats to ``None``."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def save_results(d: dict, path: Path | str) -> Path:
    """Write a results dict as indented JSON (``NaN``/``inf`` become ``null``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(d), indent=2, allow_nan=False) + "\n")
    return path


def load_results(path: Path | str) -> dict:
    """Read one results JSON."""
    return json.loads(Path(path).read_text())


def save_scores(scores: np.ndarray, path: Path | str) -> Path:
    """Save ``(n, 80)`` scores as float32 ``.npy``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(scores, dtype=np.float32))
    return path


def load_all_results(results_dir: Path | str = RESULTS_DIR) -> list[dict]:
    """Load every results JSON directly inside ``results_dir`` (sorted by file name).

    Files without a ``per_bin`` block (other JSON written next to the results)
    are skipped.
    """
    out: list[dict] = []
    for p in sorted(Path(results_dir).glob("*.json")):
        d = load_results(p)
        if not (isinstance(d, dict) and "per_bin" in d and "readout" in d and "overall" in d and "bins" in d):
            continue
        # only canonical result files: {readout}__{backbone}__res{R}.json (skips localization__/history__ side files)
        if p.stem != f"{d['readout']}__{d.get('backbone')}__res{d.get('res')}":
            continue
        out.append(d)
    return out
