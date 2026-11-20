"""Summarise the tiled-teacher distillation runs (scripts/19_distill.py) as mean +/- std over seeds.

usage: python scripts/23_distill_table.py --dataset chest|coco
Prints one LaTeX row per read-out (baseline 224 px head, inference-time tiling, 448 px teacher,
students) and writes results*/distill_table__{bb}__res224.tex.
"""
from __future__ import annotations
import argparse, glob, json, math, os
import numpy as np

BINS = ["<0.1%", "0.1-0.3%", "0.3-1%", "1-3%", "3-10%", ">10%"]

def load(p):
    with open(p) as f:
        return json.load(f)

def per_bin_from_result(d):
    pb = d["per_bin"]
    return {b: (pb[b]["auroc_macro"] if b in pb and pb[b].get("auroc_macro") is not None else float("nan")) for b in BINS}, d["overall"]["auroc_macro"]

def collect(files):
    """files -> per-bin arrays (n_seeds x n_bins) and overall array."""
    rows, alls = [], []
    for p in files:
        d = load(p); pb, ov = per_bin_from_result(d); rows.append([pb[b] for b in BINS]); alls.append(ov)
    return np.array(rows, dtype=float), np.array(alls, dtype=float)

def fmt(a):
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    if a.size == 0: return "--"
    if a.size == 1: return f"{a[0]:.3f}"
    return f"{a.mean():.3f}$\\pm${a.std(ddof=1):.3f}"

def row(name, mat, alls):
    cells = [fmt(mat[:, j]) for j in range(mat.shape[1])] + [fmt(alls)]
    return f"{name} & " + " & ".join(cells) + " \\\\"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chest", choices=["chest", "coco"])
    ap.add_argument("--backbone", default="dinov2_vitb14_reg")
    a = ap.parse_args()
    bb = a.backbone
    if a.dataset == "chest":
        rd, suffix = "results_cxr", ""
        base = [f"{rd}/attn__{bb}__res224.json", f"{rd}/attn_s1__{bb}__res224.json", f"{rd}/attn_s2__{bb}__res224.json"]
        teacher = [f"{rd}/attn_full__{bb}__res448.json", f"{rd}/attn_full_s1__{bb}__res448.json", f"{rd}/attn_full_s2__{bb}__res448.json"]
    else:
        rd, suffix = "results", "_train5k"
        base = [f"{rd}/attn_train5k__{bb}__res224.json", f"{rd}/attn_train5k_s1__{bb}__res224.json", f"{rd}/attn_train5k_s2__{bb}__res224.json"]
        teacher = [f"{rd}/attn__{bb}__res448.json"]
    tiled = [f"{rd}/attn_tiled__{bb}__res448.json"]
    kd_t = [f"{rd}/attn_kd_tiled{suffix}__{bb}__res224.json"] + [f"{rd}/attn_kd_tiled{suffix}_s{s}__{bb}__res224.json" for s in (1, 2)]
    kd_h = [f"{rd}/attn_kd_hi{suffix}__{bb}__res224.json"] + [f"{rd}/attn_kd_hi{suffix}_s{s}__{bb}__res224.json" for s in (1, 2)]
    out = []
    for name, files in [("224 px head (baseline)", base), ("Inference-time tiling (448 px tokens)", tiled),
                        ("448 px head (teacher)", teacher), ("Student, tiled teacher", kd_t), ("Student, 448 px teacher", kd_h)]:
        files = [f for f in files if os.path.exists(f)]
        if not files:
            print(f"% {name}: no files yet"); continue
        mat, alls = collect(files)
        out.append(row(name + f" ({len(files)})", mat, alls))
    tex = "\n".join(out)
    print(" & ".join(["Read-out"] + BINS + ["All"]) + " \\\\")
    print(tex)
    dst = f"{rd}/distill_table__{bb}__res224.tex"
    with open(dst, "w") as f: f.write(tex + "\n")
    print("wrote", dst)

if __name__ == "__main__":
    main()
