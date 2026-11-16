"""Step 10: paired bootstrap of the macro-AUROC difference between two read-outs.

Loads the saved test scores of both read-outs (``results/scores/``), resamples the
test images once per draw and evaluates both on every draw, so the CI of the
difference is paired. Writes ``results/compare__{a}_vs_{b}__{backbone}__res{R}.json``
and prints one line per bin plus the pooled AUROC and mAP.

    .venv/bin/python scripts/10_compare.py --a attn --b mean --backbone dinov2_vitb14_reg --res 224
    .venv/bin/python scripts/10_compare.py --pairs attn:mean,oracle:mean,attn:cls
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lip.data.labels import load_labels
from lip.eval.compare import format_compare_lines, paired_bootstrap_difference
from lip.eval.metrics import MIN_POS, N_BOOT
from lip.eval.results import save_results, scores_path


def compare_path(a: str, b: str, backbone: str, res: int, results_dir: Path | str) -> Path:
    """``results/compare__{a}_vs_{b}__{backbone}__res{R}.json``."""
    return Path(results_dir) / f"compare__{a}_vs_{b}__{backbone}__res{res}.json"


def parse_pairs(pairs: str) -> list[tuple[str, str]]:
    """``"attn:mean,oracle:mean"`` -> ``[("attn", "mean"), ("oracle", "mean")]``."""
    out = []
    for item in pairs.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"pair {item!r} must look like a:b")
        a, b = item.split(":", 1)
        out.append((a.strip(), b.strip()))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default="attn", help="read-out A (delta = A - B)")
    p.add_argument("--b", default="mean", help="read-out B")
    p.add_argument("--pairs", default="", help="comma-separated a:b pairs, overrides --a/--b")
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--split", default="test")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-pos", type=int, default=MIN_POS)
    p.add_argument("--labels", default="data/coco/labels.npz")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default=None, help="output directory (default: --results-dir)")
    a = p.parse_args(argv)

    pairs = parse_pairs(a.pairs) if a.pairs else [(a.a, a.b)]
    results_dir = Path(a.results_dir)
    out_dir = Path(a.out) if a.out else results_dir
    labels = load_labels(a.labels, split=a.split)
    y = np.asarray(labels["y"])
    max_box_frac = np.asarray(labels["max_box_frac"])

    cache: dict[str, np.ndarray] = {}

    def scores_of(name: str) -> np.ndarray:
        if name not in cache:
            path = scores_path(name, a.backbone, a.res, results_dir)
            if not path.exists():
                raise FileNotFoundError(f"no saved scores for read-out {name!r}: {path}")
            s = np.load(path).astype(np.float64)
            if s.shape != y.shape:
                raise ValueError(f"{path}: scores {s.shape} do not match labels {y.shape}")
            cache[name] = s
        return cache[name]

    status = 0
    for name_a, name_b in pairs:
        try:
            sa, sb = scores_of(name_a), scores_of(name_b)
        except (FileNotFoundError, ValueError) as e:
            print(f"skipping {name_a} vs {name_b}: {e}", file=sys.stderr)
            status = 1
            continue
        cmp = paired_bootstrap_difference(sa, sb, y, max_box_frac, n_boot=a.n_boot, seed=a.seed, min_pos=a.min_pos)
        cmp.update(a=name_a, b=name_b, backbone=a.backbone, res=int(a.res), split=a.split,
                   scores_a=str(scores_path(name_a, a.backbone, a.res, results_dir)),
                   scores_b=str(scores_path(name_b, a.backbone, a.res, results_dir)))
        path = save_results(cmp, compare_path(name_a, name_b, a.backbone, a.res, out_dir))
        print(f"{name_a} vs {name_b}  ({a.backbone}, res {a.res}, {cmp['n_images']} images, "
              f"{a.n_boot} paired draws, seed {a.seed})")
        for line in format_compare_lines(cmp, name_a, name_b):
            print("  " + line)
        print(f"wrote {path}\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
