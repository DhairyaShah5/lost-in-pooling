"""Step 9: per-class analysis on small targets (largest instance < 0.3% of the image).

For the read-outs cls, mean, attn and oracle of one backbone/resolution, computes
each class's small-target AUROC (positives in the two smallest bins pooled vs. all
negatives, classes with >= 10 such positives), the oracle gap (oracle - mean), the
head gain (attn - mean) and the context score (mean-probe small-target AUROC).
Small-target AUROCs are recomputed from the saved test scores when present and
pooled from the results JSON otherwise.

Writes ``results/perclass__{backbone}__res{R}.json`` / ``.csv``, the tabular-only
LaTeX table ``results/perclass_table__{backbone}__res{R}.tex`` (8 largest and 8
smallest oracle gaps) and the scatter figure
``results/figures/perclass__{backbone}__res{R}.png/.pdf`` (+ copy to report/figures).

    .venv/bin/python scripts/09_perclass.py --backbone dinov2_vitb14_reg --res 224
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from lip.data.labels import load_labels
from lip.eval.bins import BIN_NAMES
from lip.eval.metrics import MIN_POS
from lip.eval.perclass import (
    PERCLASS_READOUTS,
    SMALL_BINS,
    SMALL_LABEL,
    SMALL_MAX_FRAC,
    extreme_classes,
    perclass_dataframe,
    perclass_rows,
    perclass_summary,
    perclass_table_latex,
    perclass_table_text,
    plot_perclass_scatter,
    small_target_auroc,
    small_target_auroc_from_results,
)
from lip.eval.results import load_results, results_path, save_results, scores_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--readouts", default=",".join(PERCLASS_READOUTS),
                   help="comma-separated read-outs; must include mean and oracle (attn for the head gain)")
    p.add_argument("--split", default="test")
    p.add_argument("--min-pos", type=int, default=MIN_POS)
    p.add_argument("--k", type=int, default=8, help="classes per extreme block of the table")
    p.add_argument("--labels", default="data/coco/labels.npz")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--fig-dir", default="results/figures")
    p.add_argument("--report-fig-dir", default="report/figures")
    p.add_argument("--from-json", action="store_true", help="pool from the results JSON even when scores exist")
    a = p.parse_args(argv)

    readouts = tuple(r for r in a.readouts.split(",") if r)
    results_dir = Path(a.results_dir)
    labels = load_labels(a.labels, split=a.split)
    y = np.asarray(labels["y"])
    max_box_frac = np.asarray(labels["max_box_frac"])
    cat_names = [str(n) for n in labels["cat_names"]]

    small_auroc: dict[str, np.ndarray] = {}
    source: dict[str, str] = {}
    n_small = None
    for name in readouts:
        s_path = scores_path(name, a.backbone, a.res, results_dir)
        r_path = results_path(name, a.backbone, a.res, results_dir)
        if s_path.exists() and not a.from_json:
            scores = np.load(s_path).astype(np.float64)
            if scores.shape != y.shape:
                print(f"{s_path}: scores {scores.shape} do not match labels {y.shape}", file=sys.stderr)
                return 1
            auroc, n = small_target_auroc(scores, y, max_box_frac, min_pos=a.min_pos)
            source[name] = str(s_path)
        elif r_path.exists():
            auroc, n = small_target_auroc_from_results(load_results(r_path), cat_names, min_pos=a.min_pos)
            source[name] = str(r_path)
        else:
            print(f"skipping {name}: neither {s_path} nor {r_path} exists", file=sys.stderr)
            continue
        small_auroc[name] = auroc
        n_small = n if n_small is None else n_small
    if "mean" not in small_auroc or "oracle" not in small_auroc:
        print("need at least the mean and oracle read-outs", file=sys.stderr)
        return 1

    n_pos_all = (y == 1).sum(axis=0)
    rows = perclass_rows(small_auroc, n_small, cat_names, n_pos_all=n_pos_all, min_pos=a.min_pos, readouts=readouts)
    summary = perclass_summary(rows, k=a.k)
    largest, smallest = extreme_classes(rows, k=a.k)

    stem = f"perclass__{a.backbone}__res{a.res}"
    written: list[Path] = []
    out = {
        "backbone": a.backbone, "res": int(a.res), "split": a.split, "min_pos": int(a.min_pos),
        "small_bins": [BIN_NAMES[b] for b in SMALL_BINS], "small_label": SMALL_LABEL,
        "small_max_frac": float(SMALL_MAX_FRAC), "readouts": list(readouts), "source": source,
        "classes": {r["class"]: {k: v for k, v in r.items() if k != "class"} for r in rows},
        "summary": summary,
    }
    written.append(save_results(out, results_dir / f"{stem}.json"))
    csv_path = results_dir / f"{stem}.csv"
    perclass_dataframe(rows, readouts=readouts).to_csv(csv_path, index=False, float_format="%.6f")
    written.append(csv_path)
    tex_path = results_dir / f"perclass_table__{a.backbone}__res{a.res}.tex"
    tex_path.write_text(perclass_table_latex(largest, smallest))
    written.append(tex_path)

    fig_files = plot_perclass_scatter(rows, Path(a.fig_dir) / stem)
    written += fig_files
    report_dir = Path(a.report_fig_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for src in fig_files:
        dst = report_dir / src.name
        if dst.resolve() != src.resolve():
            shutil.copyfile(src, dst)
            written.append(dst)

    print(f"{summary['n_classes_eligible']}/{summary['n_classes_total']} classes with >= {a.min_pos} "
          f"small positives ({SMALL_LABEL}); backbone={a.backbone} res={a.res}")
    print(f"oracle gap mean {summary['oracle_gap_mean']:.3f} (median {summary['oracle_gap_median']:.3f}); "
          f"head gain mean {summary['head_gain_mean']:.3f} (median {summary['head_gain_median']:.3f}), "
          f"positive for {100 * summary['frac_head_gain_positive']:.0f}% of classes")
    print(f"Spearman: context vs gap {summary['spearman_context_vs_gap']:.2f}, "
          f"context vs head gain {summary['spearman_context_vs_head_gain']:.2f}, "
          f"gap vs head gain {summary['spearman_gap_vs_head_gain']:.2f}\n")
    print(perclass_table_text(largest, smallest))
    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
