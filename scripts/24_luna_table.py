"""LaTeX rows for the LUNA16 diameter table from results_luna/luna_curve__{bb}__res{R}.json (scripts/22).

Cross-validated read-outs (suffix _cv5) are single runs over every slice; the attention head is reported as
mean +/- sample std over the seeds found (attn, attn_s1, attn_s2) on the scan-level test split.
usage: python scripts/24_luna_table.py [--backbone BB] [--res 224,448]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

NAMES = {"cls_cv5": "CLS", "mean_cv5": "Mean", "max_cv5": "Max", "context_cv5": "Context (outside the box)", "oracle_cv5": "Oracle"}

def fmt(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if v.size == 0: return "--"
    if v.size == 1: return f"{v[0]:.3f}"
    return f"{v.mean():.3f}$\\pm${v.std(ddof=1):.3f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitb14_reg")
    ap.add_argument("--res", default="224,448")
    ap.add_argument("--results", default="results_luna")
    a = ap.parse_args()
    rows = []
    for res in [int(r) for r in a.res.split(",")]:
        p = f"{a.results}/luna_curve__{a.backbone}__res{res}.json"
        if not os.path.exists(p):
            print(f"% {p} missing"); continue
        d = json.load(open(p)); bins = d["bins"]; R = d["readouts"]
        if not rows:
            print("Read-out & " + " & ".join(bins) + " & All \\\\")
            npos = next(iter(R.values()))["by_diameter"]
            print("% positives per bin (CV slices): " + " ".join(f"{b}={npos[b]['n_pos']}" for b in bins))
        rows.append(f"\\multicolumn{{{len(bins)+2}}}{{l}}{{\\emph{{{res} px}}}} \\\\")
        for key, name in NAMES.items():
            if key not in R: continue
            rec = R[key]
            rows.append(f"{name} & " + " & ".join(fmt([rec['by_diameter'][b]['auroc']]) for b in bins) + f" & {rec['overall_auroc']:.3f} \\\\")
        heads = [k for k in ("attn", "attn_s1", "attn_s2") if k in R]
        if heads:
            cells = [fmt([R[h]["by_diameter"][b]["auroc"] for h in heads]) for b in bins]
            rows.append(f"Attention head ({len(heads)} seeds, test scans) & " + " & ".join(cells) + f" & {fmt([R[h]['overall_auroc'] for h in heads])} \\\\")
            hp = R[heads[0]]["by_diameter"]
            print("% head test positives per bin: " + " ".join(f"{b}={hp[b]['n_pos']}" for b in bins))
    print("\n".join(rows))
    header = ("\\begin{tabular}{lrrrrrr}\n\\toprule\nRead-out & 3--5 mm & 5--8 mm & 8--12 mm & 12--20 mm & $>20$ mm & All \\\\\n\\midrule\n")
    out = f"{a.results}/luna_table__{a.backbone}.tex"
    open(out, "w").write(header + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"); print("wrote", out)

if __name__ == "__main__":
    main()
