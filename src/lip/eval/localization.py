"""Does the learned attention land on the object?  Box-based localization metrics.

Boxes are used here for *evaluation only*; the head never sees them.

For every positive (image, class) pair and any per-patch map m (n, C, P) over the
G x G grid (attention that sums to 1, or per-patch scores softmaxed with `is_logits=True`):

* `mass_in_box`  : sum of m over patches inside the union of that class's GT boxes
* `chance_mass`  : fraction of patches inside the union (what uniform attention gives)
* `pointing_hit` : 1 if the argmax patch lies inside the union (pointing game)
* `size_bin`     : size bin of `max_box_frac[i, c]` (docs/CONTRACTS.md, six log-spaced bins)

`summarise_localization` aggregates per bin and overall and adds the analytic
"uniform" reference (mass = chance, pointing accuracy = chance).

The patch mask and bin helpers come from `lip.eval.bins` when that module is importable;
private copies of the contract definitions are used otherwise, so this module and its
tests do not depend on the evaluation harness being present.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_ROOT = Path("results")
BIN_EDGES = [0.0, 0.001, 0.003, 0.01, 0.03, 0.10, 1.0]
BIN_NAMES = ["<0.1%", "0.1-0.3%", "0.3-1%", "1-3%", "3-10%", ">10%"]


# ---------------------------------------------------------- contract fallbacks
def _box_to_patch_mask(box_xyxy, grid: int) -> np.ndarray:
    """(G*G,) bool: patch centres inside the box; the centre patch if none are."""
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    c = (np.arange(grid) + 0.5) / grid
    inside_x = (c >= x0) & (c < x1)
    inside_y = (c >= y0) & (c < y1)
    mask = inside_y[:, None] & inside_x[None, :]          # (rows, cols)
    if not mask.any():
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        col = min(max(int(cx * grid), 0), grid - 1)
        row = min(max(int(cy * grid), 0), grid - 1)
        mask[row, col] = True
    return mask.reshape(-1)


def _boxes_to_patch_mask(boxes, grid: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    mask = np.zeros(grid * grid, dtype=bool)
    for b in boxes:
        mask |= _box_to_patch_mask(b, grid)
    return mask


def _assign_bins(frac) -> np.ndarray:
    frac = np.asarray(frac, dtype=np.float64)
    return np.clip(np.searchsorted(BIN_EDGES, frac, side="right") - 1, 0, len(BIN_NAMES) - 1)


def _bins_impl():
    """(boxes_to_patch_mask, assign_bins, names, edges), preferring lip.eval.bins."""
    try:
        from lip.eval import bins as B
        return (B.boxes_to_patch_mask, B.assign_bins, list(B.BIN_NAMES),
                list(getattr(B, "BIN_EDGES", BIN_EDGES)))
    except Exception:  # module absent or mid-write: use the contract copies above
        return _boxes_to_patch_mask, _assign_bins, list(BIN_NAMES), list(BIN_EDGES)


# ----------------------------------------------------------------- the table
def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def localization_table(maps: np.ndarray, boxes: pd.DataFrame, labels: dict, grid: int,
                       is_logits: bool = False, image_ids=None) -> pd.DataFrame:
    """One row per positive (image, class) pair.

    maps     : (n, C, P) attention (rows sum to 1) or per-patch scores (`is_logits=True`,
               softmaxed over P here), in the order of `labels["image_ids"]`
               (or `image_ids` if given).
    boxes    : DataFrame with image_id, cat_idx, x0, y0, x1, y1 (normalised).
    labels   : dict for the split with y (n, C) and max_box_frac (n, C).
    Columns  : image_id, cat_idx, mass_in_box, chance_mass, pointing_hit, size_bin,
               bin_name, box_frac, argmax_patch, n_boxes.
    """
    to_mask, assign, names, _ = _bins_impl()
    ids = np.asarray(labels["image_ids"] if image_ids is None else image_ids)
    y = np.asarray(labels["y"])
    frac = np.asarray(labels["max_box_frac"])
    n, C, P = maps.shape
    if len(ids) != n or y.shape != (n, C):
        raise ValueError(f"maps {maps.shape} do not match labels y {y.shape} / ids {len(ids)}")
    if P != grid * grid:
        raise ValueError(f"maps have {P} patches but grid {grid} gives {grid * grid}")

    groups = {k: g[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float64)
              for k, g in boxes.groupby(["image_id", "cat_idx"], sort=False)}

    rows_i, rows_c = np.nonzero(y > 0)
    bins = assign(frac[rows_i, rows_c])
    rec = []
    for i, c, b in zip(rows_i, rows_c, bins):
        if b < 0:
            continue  # absent-class convention of lip.eval.bins; never true for y = 1
        key = (int(ids[i]), int(c))
        bx = groups.get(key)
        if bx is None or len(bx) == 0:
            continue  # positive label without boxes (should not happen for COCO)
        m = np.asarray(maps[i, c], dtype=np.float64)
        if is_logits:
            m = _softmax_rows(m)
        mask = to_mask(bx, grid)
        arg = int(np.argmax(m))
        rec.append({"image_id": key[0], "cat_idx": key[1],
                    "mass_in_box": float(m[mask].sum()),
                    "chance_mass": float(mask.mean()),
                    "pointing_hit": int(bool(mask[arg])),
                    "size_bin": int(b), "bin_name": names[int(b)],
                    "box_frac": float(frac[i, c]), "argmax_patch": arg, "n_boxes": int(len(bx))})
    cols = ["image_id", "cat_idx", "mass_in_box", "chance_mass", "pointing_hit", "size_bin",
            "bin_name", "box_frac", "argmax_patch", "n_boxes"]
    return pd.DataFrame(rec, columns=cols)


# --------------------------------------------------------------- summaries
def _block(df: pd.DataFrame) -> dict:
    n = int(len(df))
    if n == 0:
        return {"mass_in_box": None, "chance_mass": None, "mass_ratio": None,
                "pointing_acc": None, "n": 0}
    mass = float(df["mass_in_box"].mean())
    chance = float(df["chance_mass"].mean())
    return {"mass_in_box": mass, "chance_mass": chance,
            "mass_ratio": float(mass / chance) if chance > 0 else None,
            "pointing_acc": float(df["pointing_hit"].mean()), "n": n}


def _uniform_block(df: pd.DataFrame) -> dict:
    """Uniform attention: mass equals chance; a random argmax hits with prob = chance."""
    n = int(len(df))
    if n == 0:
        return {"mass_in_box": None, "chance_mass": None, "mass_ratio": None,
                "pointing_acc": None, "n": 0}
    chance = float(df["chance_mass"].mean())
    return {"mass_in_box": chance, "chance_mass": chance, "mass_ratio": 1.0,
            "pointing_acc": chance, "n": n}


def summarise_localization(df: pd.DataFrame, name: str = "attn") -> dict:
    """Per size bin and overall: mean mass in box, chance, ratio, pointing accuracy, n.

    Also includes the analytic `uniform` reference computed on the same pairs.
    """
    _, _, names, edges = _bins_impl()
    per_bin = {nm: _block(df[df["size_bin"] == b]) for b, nm in enumerate(names)}
    uni_bin = {nm: _uniform_block(df[df["size_bin"] == b]) for b, nm in enumerate(names)}
    return {"map": name, "n_pairs": int(len(df)),
            "bins": {"edges": edges, "names": names},
            "overall": _block(df), "per_bin": per_bin,
            "uniform": {"overall": _uniform_block(df), "per_bin": uni_bin}}


def compare_localization(maps: dict, boxes: pd.DataFrame, labels: dict, grid: int,
                         is_logits=False) -> dict:
    """{name: summary} for several (n, C, P) maps, e.g. attention vs max-MIL patch scores.

    `is_logits` may be a bool (applied to all) or a {name: bool} dict.
    """
    out = {}
    for name, m in maps.items():
        flag = is_logits.get(name, False) if isinstance(is_logits, dict) else bool(is_logits)
        out[name] = summarise_localization(localization_table(m, boxes, labels, grid, flag), name)
    return out


def format_summary(summary: dict) -> str:
    rows = [f"localization [{summary.get('map', '?')}]  pairs={summary['n_pairs']}"]
    hdr = f"{'bin':>10} {'n':>6} {'mass':>7} {'chance':>7} {'ratio':>6} {'point':>6} {'unif':>6}"
    rows.append(hdr)
    for nm in summary["bins"]["names"] + ["overall"]:
        b = summary["overall"] if nm == "overall" else summary["per_bin"][nm]
        u = summary["uniform"]["overall"] if nm == "overall" else summary["uniform"]["per_bin"][nm]
        if b["n"] == 0:
            rows.append(f"{nm:>10} {0:>6}")
            continue
        rows.append(f"{nm:>10} {b['n']:>6} {b['mass_in_box']:>7.3f} {b['chance_mass']:>7.3f} "
                    f"{b['mass_ratio']:>6.2f} {b['pointing_acc']:>6.3f} {u['pointing_acc']:>6.3f}")
    return "\n".join(rows)


# -------------------------------------------------------------------- I/O
def localization_path(readout: str, backbone: str, res: int, root: Path | str = RESULTS_ROOT) -> Path:
    return Path(root) / f"localization__{readout}__{backbone}__res{res}.json"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_localization(summary: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=_json_default))
    return path
