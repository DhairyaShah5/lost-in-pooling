"""Step 11: aggregate seed repeats of a read-out (mean and std per bin).

Loads ``results/{readout}{suffix}__{backbone}__res{R}.json`` for every suffix that
exists and writes ``results/seeds__{readout}__{backbone}__res{R}.json`` plus the
tabular-only LaTeX snippet ``results/seeds_table__{readout}__{backbone}__res{R}.tex``
with ``mean ± std`` cells. One seed is handled (std = 0, noted).

    .venv/bin/python scripts/11_seeds.py --readouts attn --suffixes "",_s1,_s2 --backbone dinov2_vitb14_reg --res 224
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lip.eval.results import load_results, save_results
from lip.eval.seeds import aggregate_seeds, seed_results_path, seeds_table_latex, seeds_table_text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--readouts", default="attn", help="comma-separated base read-out names")
    p.add_argument("--suffixes", default=",_s1,_s2",
                   help='comma-separated suffixes appended to each read-out ("" for the base run)')
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default=None, help="output directory (default: --results-dir)")
    a = p.parse_args(argv)

    readouts = [r for r in a.readouts.split(",") if r]
    suffixes = a.suffixes.split(",")  # keep "" entries: they denote the base run
    results_dir = Path(a.results_dir)
    out_dir = Path(a.out) if a.out else results_dir

    status = 0
    for readout in readouts:
        found: list[dict] = []
        files: list[str] = []
        for suf in suffixes:
            path = seed_results_path(readout, suf.strip(), a.backbone, a.res, results_dir)
            if path.exists():
                found.append(load_results(path))
                files.append(str(path))
            else:
                print(f"missing {path}", file=sys.stderr)
        if not found:
            print(f"no results for read-out {readout!r} ({a.backbone}, res {a.res})", file=sys.stderr)
            status = 1
            continue
        agg = aggregate_seeds(found, readout=readout, files=files)
        json_path = save_results(agg, out_dir / f"seeds__{readout}__{a.backbone}__res{a.res}.json")
        tex_path = out_dir / f"seeds_table__{readout}__{a.backbone}__res{a.res}.tex"
        tex_path.write_text(seeds_table_latex(agg))
        print(seeds_table_text(agg))
        print(f"wrote {json_path}\nwrote {tex_path}\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
