"""Step 15: the pre-proposal's mechanism hypothesis, tested.

Hypothesis (pre-proposal, Sec. 2.4): the cliff tracks the ratio of target extent to patch area,
not absolute size. If true, AUROC plotted against *object size in patches* should coincide for
224 px and 448 px inputs (same images, same backbone). We re-bin the saved test scores by
max_box_frac * grid^2 (patches covered by the largest instance) and plot both resolutions on
that axis. The 224 px probes are the 5k-train variants so that only the resolution differs.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lip.data.labels import load_labels
from lip.eval.metrics import bootstrap_from_bins, per_class_auroc_from_bins

EDGES = [0, 0.5, 1, 2, 4, 8, 16, np.inf]
NAMES = ["<0.5", "0.5-1", "1-2", "2-4", "4-8", "8-16", ">16"]
COLORS = {"cls": "#4477AA", "mean": "#EE6677", "max": "#CCBB44", "oracle": "#000000"}


def macro(pcb: dict, n_bins: int, min_pos: int):
    out = []
    for b in range(n_bins):
        vals = [v["bins"][b]["auroc"] for c, v in pcb.items()
                if v["bins"][b]["n_pos"] >= min_pos and np.isfinite(v["bins"][b]["auroc"])]
        out.append((float(np.mean(vals)) if vals else np.nan, len(vals)))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--patch", type=int, default=14)
    p.add_argument("--min-pos", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-fig", default="results/figures/patch_units__dinov2_vitb14_reg")
    p.add_argument("--report-fig-dir", default="report/figures")
    a = p.parse_args()

    lab = load_labels(split="test")
    y, frac = lab["y"], lab["max_box_frac"]
    variants = {224: {"cls": "cls_train5k", "mean": "mean_train5k", "max": "max_train5k", "oracle": "oracle"},
                448: {"cls": "cls", "mean": "mean", "max": "max", "oracle": "oracle"}}
    summary = {}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for res, ls in ((224, "-"), (448, "--")):
        grid = res // a.patch
        units = frac * grid * grid
        bins = np.digitize(units, EDGES[1:-1]).astype(np.int16)  # 0..6
        bins[(y == 0) | (frac <= 0)] = -1
        summary[res] = {}
        for name, readout in variants[res].items():
            path = Path(a.results_dir) / "scores" / f"{readout}__{a.backbone}__res{res}.npy"
            if not path.exists():
                print("missing", path)
                continue
            scores = np.load(path)
            pcb = per_class_auroc_from_bins(scores, y, bins, len(NAMES), min_pos=a.min_pos)
            m = macro(pcb, len(NAMES), a.min_pos)
            ci = bootstrap_from_bins(scores, y, bins, len(NAMES), n_boot=a.n_boot, seed=0, min_pos=a.min_pos)
            lo = [ci.get(b, ci.get(str(b), {})).get("ci_lo", np.nan) if isinstance(ci, dict) else np.nan for b in range(len(NAMES))]
            hi = [ci.get(b, ci.get(str(b), {})).get("ci_hi", np.nan) if isinstance(ci, dict) else np.nan for b in range(len(NAMES))]
            summary[res][name] = {"readout": readout, "auroc": [v for v, _ in m], "n_classes": [n for _, n in m],
                                  "ci_lo": [float(v) for v in lo], "ci_hi": [float(v) for v in hi]}
            xs = np.arange(len(NAMES)); ys = np.array([v for v, _ in m])
            ax.plot(xs, ys, ls=ls, marker="o" if res == 224 else "s", ms=4, color=COLORS[name],
                    label=f"{name} @ {res}px", lw=2 if name in ("mean", "oracle") else 1.3)
            if np.isfinite(lo).all():
                ax.fill_between(xs, lo, hi, color=COLORS[name], alpha=0.08, lw=0)
            print(f"{res}px {name:7s}", " ".join(f"{NAMES[b]}={v:.3f}({n})" for b, (v, n) in enumerate(m)))
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES)
    ax.set_xlabel("largest instance, in patches covered (bbox area / patch area)")
    ax.set_ylabel("macro AUROC"); ax.set_ylim(0.78, 1.0)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = Path(a.out_fig); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
        Path(a.report_fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(a.report_fig_dir) / f"{out.name}.{ext}", dpi=200)
    Path(a.results_dir, f"patch_units__{a.backbone}.json").write_text(json.dumps({"edges": EDGES[:-1] + ["inf"], "names": NAMES, "by_res": summary}, indent=2, default=float))
    print("wrote", out)
