"""Step 1b: dataset statistics for the report: positives per size bin, classes per bin."""
import argparse
import json
from pathlib import Path

import numpy as np

from lip.data.labels import load_labels

EDGES = [0, 0.001, 0.003, 0.01, 0.03, 0.10, 1.0]
NAMES = ["<0.1%", "0.1-0.3%", "0.3-1%", "1-3%", "3-10%", ">10%"]
TEX_NAMES = {"<0.1%": r"$<0.1\%$", "0.1-0.3%": r"$0.1$--$0.3\%$", "0.3-1%": r"$0.3$--$1\%$",
             "1-3%": r"$1$--$3\%$", "3-10%": r"$3$--$10\%$", ">10%": r"$>10\%$"}


def stats_for_split(lab: dict, min_pos: int = 10) -> dict:
    y, f = lab["y"], lab["max_box_frac"]
    b = np.digitize(f, EDGES[1:-1], right=False)  # 0..5
    out = {"n_images": int(y.shape[0]), "n_pos_pairs": int(y.sum()), "bins": {}}
    for k, name in enumerate(NAMES):
        m = (y == 1) & (b == k)
        per_class = m.sum(0)
        out["bins"][name] = {"n_pos": int(m.sum()), "n_classes_ge_min": int((per_class >= min_pos).sum()),
                             "median_pos_per_class": float(np.median(per_class[per_class > 0])) if (per_class > 0).any() else 0.0}
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/dataset_stats.json")
    p.add_argument("--tex", default="results/dataset_stats_table.tex")
    a = p.parse_args()
    res = {s: stats_for_split(load_labels(split=s)) for s in ("train", "val", "test")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    rows = []
    for name in NAMES:
        t, te = res["train"]["bins"][name], res["test"]["bins"][name]
        rows.append(f"{TEX_NAMES[name]} & {t['n_pos']} & {te['n_pos']} & {te['n_classes_ge_min']} \\\\")
    tex = "\n".join([
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{lrrr}\toprule",
        r"size bin & train pos. & test pos. & test classes ($\geq$10) \\ \midrule",
        *rows,
        r"\midrule all & " + f"{res['train']['n_pos_pairs']} & {res['test']['n_pos_pairs']} & 80" + r" \\",
        r"\bottomrule\end{tabular}",
        r"\caption{Positive (image, class) pairs per size bin of the largest instance (relative bbox area). "
        r"The last column counts classes with at least 10 test positives in the bin, i.e.\ those entering the macro average.}",
        r"\label{tab:data}\end{table}"])
    Path(a.tex).write_text(tex)
    for s in ("train", "test"):
        print(s, json.dumps(res[s]["bins"]))
