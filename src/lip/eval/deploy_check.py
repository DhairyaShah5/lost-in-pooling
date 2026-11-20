"""Pre-deployment check: how much does a frozen global embedding lose on a target of a given relative size?

Two fitted curves, both logistic in a log size variable:
  g(s)  global read-out AUROC vs token fraction s (the target's share of the image area, which equals its share
        of tokens under squash resizing);
  o(p)  oracle region read-out AUROC vs patches covered p = s * grid^2.
`fit_curve` fits a + (1 - a) * sigmoid(k * (x - m)) by weighted least squares on per-class per-bin points.
The CLI (`python -m lip.eval.deploy_check --grid 16 --target-frac 0.002`) loads results/deploy_check.json and
prints the predicted loss for a target size and token grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BIN_EDGES = [0, 0.001, 0.003, 0.01, 0.03, 0.10, 1.0]
BIN_NAMES = ["<0.1%", "0.1-0.3%", "0.3-1%", "1-3%", "3-10%", ">10%"]
BIN_MIDS = [0.0005, np.sqrt(0.001 * 0.003), np.sqrt(0.003 * 0.01), np.sqrt(0.01 * 0.03), np.sqrt(0.03 * 0.10), 0.25]
PATCH = {"dinov2_vitb14_reg": 14, "dinov2_vits14_reg": 14, "rad_dino": 14, "clip_vitb16": 16, "siglip_vitb16": 16,
         "dino_vitb16": 16}
DEFAULT_JSON = Path("results/deploy_check.json")


def points_from_results(d: dict, min_pos: int, grid: int) -> list[dict]:
    """Per-class per-bin points (s, patches, auroc, n_pos) from a results JSON."""
    out = []
    for cls, bins in d["per_class_bin"].items():
        for b, v in bins.items():
            if b not in BIN_NAMES or v.get("auroc") is None or v.get("n_pos", 0) < min_pos:
                continue
            s = BIN_MIDS[BIN_NAMES.index(b)]
            out.append({"class": cls, "bin": b, "s": float(s), "patches": float(s * grid * grid),
                        "auroc": float(v["auroc"]), "n_pos": int(v["n_pos"])})
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def curve(params, x):
    a, k, m = params
    return a + (1.0 - a) * sigmoid(k * (x - m))


def fit_curve(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Weighted least-squares fit of a + (1-a) sigmoid(k (x - m)); x is log10 of the size variable."""
    from scipy.optimize import least_squares
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.sqrt(np.asarray(w, float))

    def resid(p):
        return w * (curve(p, x) - y)
    p0 = [max(0.0, min(0.95, float(np.min(y)))), 1.5, float(np.median(x))]
    r = least_squares(resid, p0, bounds=([0.0, 0.05, -10.0], [0.999, 20.0, 10.0]))
    pred = curve(r.x, x)
    return {"a": float(r.x[0]), "k": float(r.x[1]), "m": float(r.x[2]), "n_points": int(len(x)),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}


def predict(fit: dict, size_variable: float) -> float:
    return float(curve((fit["a"], fit["k"], fit["m"]), np.log10(size_variable)))


def macro_by_bin(points: list[dict]) -> dict:
    out = {}
    for b in BIN_NAMES:
        v = [p["auroc"] for p in points if p["bin"] == b]
        out[b] = float(np.mean(v)) if v else None
    return out


def check(grid: int, target_frac: float, fits: dict, tau: float = 0.05) -> dict:
    """Predicted AUROCs for a target covering `target_frac` of the image on a `grid` x `grid` token grid."""
    patches = target_frac * grid * grid
    g_ctx = predict(fits["g_coco"], target_frac)
    g_noctx = predict(fits["g_chest"], target_frac) if "g_chest" in fits else None
    o = predict(fits["o_patches"], max(patches, 1e-3))
    gap_ctx = o - g_ctx
    gap_noctx = None if g_noctx is None else o - g_noctx
    verdict = "region or attention read-out recommended" if (gap_ctx > tau or (gap_noctx or 0) > tau) \
        else "global pooling acceptable at this size"
    return {"grid": grid, "target_frac": target_frac, "patches_covered": patches,
            "global_auroc_with_scene_prior": g_ctx, "global_auroc_no_scene_prior": g_noctx,
            "region_auroc": o, "gap_with_scene_prior": gap_ctx, "gap_no_scene_prior": gap_noctx,
            "threshold": tau, "verdict": verdict}


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-deployment check for frozen global embeddings on small targets.")
    p.add_argument("--grid", type=int, required=True, help="token grid side, e.g. 16 for ViT-B/14 at 224 px")
    p.add_argument("--target-frac", type=float, required=True, help="expected target area as a fraction of the image")
    p.add_argument("--fits", default=str(DEFAULT_JSON))
    p.add_argument("--tau", type=float, default=0.05)
    a = p.parse_args()
    fits = json.loads(Path(a.fits).read_text())["fits"]
    r = check(a.grid, a.target_frac, fits, a.tau)
    print(f"target {a.target_frac*100:.2f}% of the image on a {a.grid}x{a.grid} grid -> {r['patches_covered']:.2f} patches covered")
    print(f"  predicted global-pooling AUROC: {r['global_auroc_with_scene_prior']:.3f} with a scene prior (COCO fit)"
          + (f", {r['global_auroc_no_scene_prior']:.3f} without one (radiograph fit)" if r['global_auroc_no_scene_prior'] is not None else ""))
    print(f"  predicted region read-out AUROC: {r['region_auroc']:.3f}  (gap {r['gap_with_scene_prior']:+.3f}"
          + (f" / {r['gap_no_scene_prior']:+.3f}" if r['gap_no_scene_prior'] is not None else "") + ")")
    print(f"  verdict: {r['verdict']}")


if __name__ == "__main__":
    main()
