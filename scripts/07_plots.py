"""Step 7: blindness-curve figure and per-bin LaTeX table from results/*.json."""
import argparse
import shutil
import sys
from pathlib import Path

from lip.eval.plots import plot_blindness_curves, results_table_latex, results_table_text
from lip.eval.results import load_all_results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--fig-dir", default="results/figures")
    p.add_argument("--report-fig-dir", default="report/figures")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    a = p.parse_args(argv)

    results_dir = Path(a.results_dir)
    results = [
        r for r in load_all_results(results_dir)
        if str(r.get("backbone")) == a.backbone and int(r.get("res", -1)) == a.res
    ]
    if not results:
        print(f"no results for backbone={a.backbone} res={a.res} in {results_dir}", file=sys.stderr)
        return 1

    stem = f"blindness_curve__{a.backbone}__res{a.res}"
    fig_dir = Path(a.fig_dir)
    written = plot_blindness_curves(results, fig_dir / stem, res=a.res)
    report_dir = Path(a.report_fig_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for src in list(written):
        dst = report_dir / src.name
        if dst.resolve() != src.resolve():
            shutil.copyfile(src, dst)
            written.append(dst)

    tex = results_table_latex(
        results,
        floating=False,
        caption=(f"Macro AUROC per relative-size bin of the largest instance "
                 f"({a.backbone.replace('_', ' ')}, {a.res}px input). Best non-oracle read-out per column in bold."),
        label=f"tab:blindness_{a.backbone}_res{a.res}",
    )
    tex_path = results_dir / f"blindness_table__{a.backbone}__res{a.res}.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex)
    written.append(tex_path)

    print(f"{len(results)} read-outs for backbone={a.backbone} res={a.res}\n")
    print(results_table_text(results))
    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
