"""Step 22: LUNA16 nodule slices, AUROC by nodule diameter for every read-out with saved scores.

Positives are slices through an annotated nodule, binned by the largest nodule's diameter; negatives are the
nodule-free lung slices. Reads results_luna/scores/*.npy written by steps 14 (scan-disjoint CV) and 05 (head).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lip.data.luna16 import DIAMETER_BINS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--results", default="results_luna")
    p.add_argument("--manifest", default="data/luna16/manifest.parquet")
    p.add_argument("--labels", default="data/luna16/labels.npz")
    p.add_argument("--labels-head", default="data/luna16/labels_head.npz")
    p.add_argument("--readouts", default="cls_cv5,mean_cv5,max_cv5,oracle_cv5,context_cv5,attn,attn_s1,attn_s2")
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--fig", default="results_luna/figures/luna_curve.pdf")
    p.add_argument("--report-fig-dir", default="report/figures")
    a = p.parse_args()
    man = pd.read_parquet(a.manifest).set_index("image_id")
    rng = np.random.default_rng(0)
    out = {"backbone": a.backbone, "res": a.res, "bins": [f"{lo}-{hi} mm" if hi < 100 else f">{lo} mm" for lo, hi in DIAMETER_BINS], "readouts": {}}
    for r in [x for x in a.readouts.split(",") if x]:
        sp = Path(a.results) / "scores" / f"{r}__{a.backbone}__res{a.res}.npy"
        if not sp.exists():
            continue
        s = np.load(sp)[:, 0]
        labs = np.load(a.labels_head if not r.endswith("cv5") else a.labels, allow_pickle=False)
        split = labs["split"]; mask = split == "test"
        ids = labs["image_ids"][mask]; y = labs["y"][mask, 0]
        if len(s) != len(ids):
            # head scores follow the head manifest's test split order; CV scores the CV ("all test") order
            print(f"[22] {r}: {len(s)} scores vs {len(ids)} labelled test slices; skipped"); continue
        d = man.loc[ids, "max_diameter_mm"].to_numpy()
        neg = y == 0
        rec = {"overall_auroc": float(roc_auc_score(y, s)), "n_pos": int(y.sum()), "n_neg": int(neg.sum()), "by_diameter": {}}
        for (lo, hi), name in zip(DIAMETER_BINS, out["bins"]):
            pos = (y == 1) & (d >= lo) & (d < hi)
            if pos.sum() < 5:
                rec["by_diameter"][name] = {"auroc": None, "n_pos": int(pos.sum())}; continue
            rows = np.where(pos | neg)[0]; lab = pos[rows].astype(int)
            au = roc_auc_score(lab, s[rows])
            boots = []
            for _ in range(a.n_boot):
                b = rng.choice(rows, size=len(rows), replace=True)
                if pos[b].sum() > 0 and neg[b].sum() > 0:
                    boots.append(roc_auc_score(pos[b].astype(int), s[b]))
            rec["by_diameter"][name] = {"auroc": float(au), "n_pos": int(pos.sum()),
                                        "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5))}
        out["readouts"][r] = rec
        print(f"[22] {r:12s} overall {rec['overall_auroc']:.3f} (pos {rec['n_pos']}, neg {rec['n_neg']}) | "
              + " ".join(f"{k}={v['auroc']:.3f}({v['n_pos']})" if v['auroc'] is not None else f"{k}=--({v['n_pos']})" for k, v in rec["by_diameter"].items()))
    Path(a.results, f"luna_curve__{a.backbone}__res{a.res}.json").write_text(json.dumps(out, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(out["bins"]))
    for r, rec in out["readouts"].items():
        if r.endswith(("_s1", "_s2")):
            continue
        ys = [rec["by_diameter"][b]["auroc"] if rec["by_diameter"][b]["auroc"] is not None else np.nan for b in out["bins"]]
        lo = [rec["by_diameter"][b].get("ci_lo", np.nan) for b in out["bins"]]; hi = [rec["by_diameter"][b].get("ci_hi", np.nan) for b in out["bins"]]
        ax.plot(x, ys, "o-", label=r.replace("_cv5", " (CV)"))
        ax.fill_between(x, lo, hi, alpha=0.12)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(out["bins"]); ax.set_xlabel("nodule diameter"); ax.set_ylabel("AUROC (nodule slice vs nodule-free slice)")
    ax.set_title(f"LUNA16 slices, {a.backbone}, {a.res} px"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); Path(a.fig).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.fig, bbox_inches="tight")
    Path(a.report_fig_dir).mkdir(parents=True, exist_ok=True); fig.savefig(Path(a.report_fig_dir) / Path(a.fig).name, bbox_inches="tight")
    print(f"[22] -> {a.fig}")


if __name__ == "__main__":
    main()
