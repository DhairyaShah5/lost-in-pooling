"""Step 12: cross-backbone figure (one panel per backbone) and table.

Loads every results JSON, keeps the requested backbones at the given resolution
(skipping, and naming, those without results), and writes
``results/figures/cross_backbone__res{R}.png/.pdf`` (+ copy to report/figures) and
the tabular-only ``results/cross_backbone_table__res{R}.tex`` with rows =
backbone x read-out and the best non-oracle value per backbone and column in bold.

    .venv/bin/python scripts/12_cross_backbone.py --backbones dinov2_vitb14_reg,clip_vitb16,siglip_vitb16 --res 224
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from lip.eval.plots import cross_backbone_table_latex, cross_backbone_table_text, plot_backbone_panels
from lip.eval.results import load_all_results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbones", default="dinov2_vitb14_reg,clip_vitb16,siglip_vitb16",
                   help="comma-separated backbone names, in panel order")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--readouts", default="cls,mean,attn,oracle", help="comma-separated read-outs, in order")
    p.add_argument("--titles", default="", help='optional comma-separated "backbone=title" panel titles')
    p.add_argument("--patch-sizes", default="", help='optional comma-separated "backbone=patch" overrides')
    p.add_argument("--results-dir", default="results")
    p.add_argument("--fig-dir", default="results/figures")
    p.add_argument("--report-fig-dir", default="report/figures")
    a = p.parse_args(argv)

    backbones = [b for b in a.backbones.split(",") if b]
    readouts = tuple(r for r in a.readouts.split(",") if r)
    titles = dict(kv.split("=", 1) for kv in a.titles.split(",") if "=" in kv)
    patch_sizes = {k: int(v) for k, v in (kv.split("=", 1) for kv in a.patch_sizes.split(",") if "=" in kv)}

    results_dir = Path(a.results_dir)
    all_results = load_all_results(results_dir)
    by_backbone: dict[str, list[dict]] = {}
    for bb in backbones:
        rs = [r for r in all_results if str(r.get("backbone")) == bb and int(r.get("res", -1)) == a.res
              and str(r.get("readout")) in readouts]
        if not rs:
            print(f"skipping {bb}: no results at res {a.res} in {results_dir}")
            continue
        by_backbone[bb] = rs
        print(f"{bb}: {', '.join(sorted(str(r['readout']) for r in rs))}")
    if not by_backbone:
        print("no backbone has results; nothing written", file=sys.stderr)
        return 1

    stem = f"cross_backbone__res{a.res}"
    written = plot_backbone_panels(by_backbone, Path(a.fig_dir) / stem, readouts=readouts, res=a.res,
                                   titles=titles or None, patch_sizes=patch_sizes or None)
    report_dir = Path(a.report_fig_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for src in list(written):
        dst = report_dir / src.name
        if dst.resolve() != src.resolve():
            shutil.copyfile(src, dst)
            written.append(dst)
    tex_path = results_dir / f"cross_backbone_table__res{a.res}.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(cross_backbone_table_latex(by_backbone, readouts=readouts, res=a.res))
    written.append(tex_path)

    print()
    print(cross_backbone_table_text(by_backbone, readouts=readouts, res=a.res))
    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
