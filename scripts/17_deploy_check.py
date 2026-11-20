"""Step 17: formalise and validate the token-fraction pre-deployment check (plan, week of Nov 16).

A. Resolution invariance: g(s) fitted on DINOv2 mean pooling at 224/336/448 px; o(p) on the oracle in patch units.
B. Cross-backbone: leave-one-backbone-out prediction of each backbone's per-bin macro from the others' fit.
C. Domain transfer: COCO fit vs observed radiograph probes (image-level CLS/mean, DINOv2 and RAD-DINO) and a chest fit.
D. Sample size: how many boxed test images are needed to detect the oracle-vs-global gap on small targets.
Writes results/deploy_check.json (fits + tables) and the figure deploy_check.pdf.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lip.eval.deploy_check import (BIN_NAMES, PATCH, fit_curve, macro_by_bin, points_from_results, predict)


def load(results_dir, readout, backbone, res):
    p = Path(results_dir) / f"{readout}__{backbone}__res{res}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pts(results_dir, readout, backbone, res, min_pos):
    d = load(results_dir, readout, backbone, res)
    if d is None:
        return []
    grid = res // PATCH[backbone]
    return points_from_results(d, min_pos, grid)


def fit_pts(points, var="s"):
    x = np.log10([p[var] for p in points]); y = np.array([p["auroc"] for p in points])
    return fit_curve(x, y)


def small_bins_mask(bins_idx, n_small):
    return bins_idx < n_small


def power_analysis(scores_a, scores_b, y, max_box_frac, n_list, n_draws, small_bins, tau, min_pos, seed=0):
    """Fraction of random test subsets of n images on which the macro AUROC gap (a - b) over classes with
    >= min_pos small-target positives exceeds tau. Positives = small-target positives; negatives = label-absent."""
    from sklearn.metrics import roc_auc_score
    from lip.eval.bins import assign_bins
    rng = np.random.default_rng(seed)
    bins = assign_bins(max_box_frac)                       # (N, C) bin index, -1 for non-positives
    small = (y == 1) & (bins >= 0) & (bins < small_bins)
    neg = (y == 0)
    out = {}
    N = y.shape[0]
    for n in n_list:
        if n > N:
            continue
        gaps = []
        for _ in range(n_draws):
            idx = rng.choice(N, size=n, replace=False)
            ga, gb = [], []
            for c in range(y.shape[1]):
                pos = idx[small[idx, c]]; ng = idx[neg[idx, c]]
                if len(pos) < min_pos or len(ng) < min_pos:
                    continue
                rows = np.concatenate([pos, ng]); lab = np.r_[np.ones(len(pos)), np.zeros(len(ng))]
                ga.append(roc_auc_score(lab, scores_a[rows, c])); gb.append(roc_auc_score(lab, scores_b[rows, c]))
            if ga:
                gaps.append(float(np.mean(ga) - np.mean(gb)))
        if gaps:
            g = np.array(gaps)
            out[int(n)] = {"n_draws": int(len(g)), "gap_mean": float(g.mean()), "gap_sd": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
                           "power": float((g > tau).mean()), "frac_positive": float((g > 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--results-cxr", default="results_cxr")
    ap.add_argument("--out", default="results/deploy_check.json")
    ap.add_argument("--fig", default="results/figures/deploy_check.pdf")
    ap.add_argument("--report-fig-dir", default="report/figures")
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--tau", type=float, default=0.05)
    a = ap.parse_args()
    R, X = a.results_dir, a.results_cxr
    out = {"fits": {}, "A_resolution": {}, "B_backbones": {}, "C_domain": {}, "D_power": {}}

    # ---- A. resolution invariance (DINOv2, probes trained on the same 5,000 images) -------------------------
    for res, readout in [(224, "mean_train5k"), (336, "mean"), (448, "mean")]:
        P = pts(R, readout, "dinov2_vitb14_reg", res, 10)
        if P:
            out["A_resolution"][f"g_mean_{res}"] = {**fit_pts(P, "s"), "macro_by_bin": macro_by_bin(P)}
    for res in (224, 336, 448):
        P = pts(R, "oracle", "dinov2_vitb14_reg", res, 10)
        if P:
            out["A_resolution"][f"o_oracle_{res}"] = {**fit_pts(P, "patches"), "macro_by_bin": macro_by_bin(P)}
    sgrid = np.log10(np.logspace(-4, -0.5, 50))
    gs = [v for k, v in out["A_resolution"].items() if k.startswith("g_mean")]
    if len(gs) > 1:
        curves = np.array([[predict(f, 10 ** x) for x in sgrid] for f in gs])
        out["A_resolution"]["g_max_abs_diff_between_resolutions"] = float(np.max(np.ptp(curves, axis=0)))
    pgrid = np.log10(np.logspace(-1, 2.6, 50))
    os_ = [v for k, v in out["A_resolution"].items() if k.startswith("o_oracle")]
    if len(os_) > 1:
        curves = np.array([[predict(f, 10 ** x) for x in pgrid] for f in os_])
        out["A_resolution"]["o_max_abs_diff_between_resolutions"] = float(np.max(np.ptp(curves, axis=0)))
    # oracle in token-fraction units for contrast (should NOT coincide across resolutions)
    os_s = []
    for res in (224, 336, 448):
        P = pts(R, "oracle", "dinov2_vitb14_reg", res, 10)
        if P:
            os_s.append(fit_pts(P, "s"))
    if len(os_s) > 1:
        curves = np.array([[predict(f, 10 ** x) for x in sgrid] for f in os_s])
        out["A_resolution"]["o_in_s_units_max_abs_diff"] = float(np.max(np.ptp(curves, axis=0)))

    # ---- B. cross-backbone pooling and leave-one-out ------------------------------------------------------
    backbones = [b for b in ("dinov2_vitb14_reg", "clip_vitb16", "siglip_vitb16", "dino_vitb16") if load(R, "mean", b, 224)]
    P_by_bb = {b: pts(R, "mean", b, 224, 10) for b in backbones}
    all_pts = [p for b in backbones for p in P_by_bb[b]]
    out["fits"]["g_coco"] = {**fit_pts(all_pts, "s"), "backbones": backbones, "readout": "mean", "res": 224}
    O_by = {b: pts(R, "oracle", b, 224, 10) for b in backbones}
    o_all = [p for b in backbones for p in O_by[b]]
    for res in (336, 448):   # add the DINOv2 resolution points so the patch-units fit spans more patches
        o_all += pts(R, "oracle", "dinov2_vitb14_reg", res, 10)
    out["fits"]["o_patches"] = {**fit_pts(o_all, "patches"), "backbones": backbones, "readout": "oracle"}
    for b in backbones:
        rest = [p for bb in backbones if bb != b for p in P_by_bb[bb]]
        if not rest or not P_by_bb[b]:
            continue
        f = fit_pts(rest, "s")
        obs = macro_by_bin(P_by_bb[b])
        pred = {bn: predict(f, s) for bn, s in zip(BIN_NAMES, [0.0005, np.sqrt(3e-6), np.sqrt(3e-5), np.sqrt(3e-4), np.sqrt(3e-3), 0.25])}
        errs = [pred[bn] - obs[bn] for bn in BIN_NAMES if obs[bn] is not None]
        out["B_backbones"][b] = {"observed": obs, "predicted_from_others": pred,
                                 "mae": float(np.mean(np.abs(errs))), "max_abs_err": float(np.max(np.abs(errs)))}

    # ---- C. radiographs: observed vs COCO prediction, and a chest fit -------------------------------------
    chest_pts = []
    for bb in ("dinov2_vitb14_reg", "rad_dino"):
        for readout in ("cls", "mean"):
            P = pts(X, readout, bb, 224, 5)
            if P:
                obs = macro_by_bin(P)
                pred = {bn: predict(out["fits"]["g_coco"], s) for bn, s in zip(BIN_NAMES, [0.0005, np.sqrt(3e-6), np.sqrt(3e-5), np.sqrt(3e-4), np.sqrt(3e-3), 0.25])}
                out["C_domain"][f"{readout}__{bb}"] = {"observed": obs, "coco_prediction": pred,
                                                       "mean_signed_err": float(np.mean([pred[b] - obs[b] for b in BIN_NAMES if obs[b] is not None]))}
                chest_pts += P
    if chest_pts:
        out["fits"]["g_chest"] = {**fit_pts(chest_pts, "s"), "readouts": "cls+mean image-level probes, DINOv2 and RAD-DINO, 224 px"}
    chest_or = []
    for bb in ("dinov2_vitb14_reg", "rad_dino"):
        for res in (224, 448):
            chest_or += pts(X, "oracle_cv5", bb, res, 5)
    if chest_or:
        f = out["fits"]["o_patches"]
        errs = [predict(f, p["patches"]) - p["auroc"] for p in chest_or]
        out["C_domain"]["oracle_cv5_vs_coco_patch_fit"] = {"n_points": len(chest_or), "mean_signed_err": float(np.mean(errs)),
                                                           "mae": float(np.mean(np.abs(errs))), "chest_fit": fit_pts(chest_or, "patches")}

    # ---- D. sample size for detecting the gap ------------------------------------------------------------
    from lip.data.labels import load_labels
    lab = load_labels("data/coco/labels.npz", split="test")
    sa = np.load(Path(R) / "scores" / "oracle__dinov2_vitb14_reg__res224.npy")
    sb = np.load(Path(R) / "scores" / "mean__dinov2_vitb14_reg__res224.npy")
    out["D_power"]["coco_sub0.3pct_oracle_vs_mean"] = power_analysis(sa, sb, lab["y"], lab["max_box_frac"],
                                                                      [50, 100, 200, 400, 800, 1600, 3200, 5000], a.n_draws, 2, a.tau, 5)
    labx = load_labels("data/nih_cxr/labels.npz", split="test")
    pa = Path(X) / "scores" / "oracle_cv5__dinov2_vitb14_reg__res224.npy"
    pb = Path(X) / "scores" / "cls_cv5__dinov2_vitb14_reg__res224.npy"
    if pa.exists() and pb.exists():
        out["D_power"]["chest_sub1pct_oracle_vs_cls_cv5"] = power_analysis(np.load(pa), np.load(pb), labx["y"], labx["max_box_frac"],
                                                                           [50, 100, 200, 400, 880], a.n_draws, 3, a.tau, 3)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[17] wrote {a.out}")
    print("[17] A:", {k: round(v, 4) for k, v in out["A_resolution"].items() if isinstance(v, float)})
    print("[17] fits:", {k: {kk: round(vv, 3) for kk, vv in v.items() if isinstance(vv, float)} for k, v in out["fits"].items()})
    print("[17] B:", {b: round(v["mae"], 3) for b, v in out["B_backbones"].items()})
    print("[17] C:", {k: round(v["mean_signed_err"], 3) for k, v in out["C_domain"].items() if "mean_signed_err" in v})
    for k, v in out["D_power"].items():
        print(f"[17] D {k}:", {n: (round(r['gap_mean'], 3), round(r['power'], 2)) for n, r in v.items()})

    # ---- figure ----------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    ax = axes[0]
    xs = np.logspace(-4, -0.4, 100)
    for b in backbones:
        mb = macro_by_bin(P_by_bb[b])
        ax.plot([0.0005, np.sqrt(3e-6), np.sqrt(3e-5), np.sqrt(3e-4), np.sqrt(3e-3), 0.25], [mb[n] for n in BIN_NAMES], "o", ms=4, alpha=0.7, label=f"{b} mean (obs.)")
    ax.plot(xs, [predict(out["fits"]["g_coco"], x) for x in xs], "k-", lw=1.5, label="COCO fit g(s)")
    for k, v in out["C_domain"].items():
        if "observed" in v:
            ax.plot([0.0005, np.sqrt(3e-6), np.sqrt(3e-5), np.sqrt(3e-4), np.sqrt(3e-3), 0.25], [v["observed"][n] for n in BIN_NAMES], "s", ms=4, alpha=0.7, label=f"chest {k.split('__')[0]} {k.split('__')[1][:8]}")
    if "g_chest" in out["fits"]:
        ax.plot(xs, [predict(out["fits"]["g_chest"], x) for x in xs], "k--", lw=1.5, label="chest fit")
    ax.set_xscale("log"); ax.set_xlabel("token fraction s (target area / image area)"); ax.set_ylabel("global read-out AUROC")
    ax.set_title("Global pooling vs token fraction"); ax.legend(fontsize=6, loc="lower right"); ax.grid(alpha=0.3)
    ax = axes[1]
    ps = np.logspace(-1, 2.6, 100)
    for k, v in out["A_resolution"].items():
        if k.startswith("o_oracle"):
            res = int(k.split("_")[-1]); grid = res // 14
            ax.plot([m * grid * grid for m in [0.0005, np.sqrt(3e-6), np.sqrt(3e-5), np.sqrt(3e-4), np.sqrt(3e-3), 0.25]],
                    [v["macro_by_bin"][n] for n in BIN_NAMES], "o", ms=4, label=f"oracle {res} px (obs.)")
    ax.plot(ps, [predict(out["fits"]["o_patches"], x) for x in ps], "k-", lw=1.5, label="fit o(p)")
    if "oracle_cv5_vs_coco_patch_fit" in out["C_domain"]:
        cf = out["C_domain"]["oracle_cv5_vs_coco_patch_fit"]["chest_fit"]
        ax.plot(ps, [predict(cf, x) for x in ps], "k--", lw=1.5, label="chest oracle fit")
    ax.set_xscale("log"); ax.set_xlabel("patches covered p = s * grid^2"); ax.set_ylabel("oracle AUROC")
    ax.set_title("Region read-out vs patches covered"); ax.legend(fontsize=6, loc="lower right"); ax.grid(alpha=0.3)
    ax = axes[2]
    for k, v in out["D_power"].items():
        ns = sorted(v); ax.plot(ns, [v[n]["power"] for n in ns], "o-", label=k.replace("_", " "))
    ax.axhline(0.8, color="grey", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("boxed test images"); ax.set_ylabel(f"P(estimated gap > {a.tau})")
    ax.set_title("Detecting the gap from a small boxed sample"); ax.legend(fontsize=6, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, bbox_inches="tight")
    Path(a.report_fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(a.report_fig_dir) / Path(a.fig).name, bbox_inches="tight")
    print(f"[17] figure -> {a.fig}")


if __name__ == "__main__":
    main()
