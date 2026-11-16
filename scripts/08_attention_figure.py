"""Step 8: qualitative figure, attention maps of the learned head on small objects.

Picks ~6 positive (image, class) pairs from the two smallest size bins where the head
scores the class above a threshold, and draws two rows: the squash-resized image with
the GT box(es) on top, the class-specific attention heat map overlaid below.

    .venv/bin/python scripts/08_attention_figure.py --readout attn --backbone dinov2_vitb14_reg --res 224
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def pick_candidates(y, frac, scores, bins_of, image_ids, n, min_score, bins, rng):
    """Greedy pick of n (row, class) pairs: highest scores, distinct images and classes."""
    ii, cc = np.nonzero((y > 0) & np.isin(bins_of, bins) & (scores > min_score))
    if len(ii) == 0:
        return []
    order = np.lexsort((rng.random(len(ii)), -scores[ii, cc]))     # score desc, random ties
    picked, used_img, used_cls = [], set(), set()
    for relax in (False, True):                                   # 2nd pass: allow repeat classes
        for k in order:
            i, c = int(ii[k]), int(cc[k])
            if len(picked) >= n:
                break
            if int(image_ids[i]) in used_img or (not relax and c in used_cls):
                continue
            picked.append((i, c))
            used_img.add(int(image_ids[i]))
            used_cls.add(c)
    return picked[:n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--readout", default="attn")
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--bins", default="0,1", help="size-bin indices to draw from (smallest first)")
    p.add_argument("--split", default="test")
    p.add_argument("--manifest", default="data/coco/manifest.parquet")
    p.add_argument("--labels", default="data/coco/labels.npz")
    p.add_argument("--boxes", default="data/coco/boxes.parquet")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="report/figures/attention_examples.png")
    p.add_argument("--copy-to", default="results/figures")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle
    import pandas as pd
    from PIL import Image
    from lip.data.labels import load_labels
    from lip.eval.localization import _bins_impl
    from lip.models.train_head import attention_path

    _, assign_bins, bin_names, _ = _bins_impl()
    rng = np.random.default_rng(a.seed)
    res_root = Path(a.results)
    tag = f"{a.readout}__{a.backbone}__res{a.res}"
    apath = attention_path(a.readout, a.backbone, a.res, root=res_root / "attention")
    spath = res_root / "scores" / f"{tag}.npy"
    for q in (apath, spath):
        if not q.exists():
            print(f"[08] missing {q}; run scripts/05_train_head.py first. Nothing drawn.")
            return
    alpha = np.load(apath, mmap_mode="r")                 # (n, C, P) float16
    scores = np.load(spath)                               # (n, C) float32
    lab = load_labels(a.labels, split=a.split)
    y, frac, ids = lab["y"], lab["max_box_frac"], lab["image_ids"]
    names = [str(s) for s in lab["cat_names"]]
    n_img, C, P = alpha.shape
    G = int(round(np.sqrt(P)))
    if scores.shape != (n_img, C) or y.shape != (n_img, C):
        print(f"[08] shape mismatch: alpha {alpha.shape}, scores {scores.shape}, y {y.shape}")
        return
    man = pd.read_parquet(a.manifest).set_index("image_id")
    boxes = pd.read_parquet(a.boxes)
    groups = {k: g[["x0", "y0", "x1", "y1"]].to_numpy(dtype=float)
              for k, g in boxes.groupby(["image_id", "cat_idx"], sort=False)}

    bins_of = assign_bins(frac)
    want = [int(b) for b in a.bins.split(",")]
    picked = pick_candidates(y, frac, scores, bins_of, ids, a.n, a.min_score, want, rng)
    while len(picked) < a.n and max(want) + 1 < len(bin_names):
        want.append(max(want) + 1)
        print(f"[08] only {len(picked)} candidates; widening to bins {want}")
        picked = pick_candidates(y, frac, scores, bins_of, ids, a.n, a.min_score, want, rng)
    picked = [(i, c) for i, c in picked if int(ids[i]) in man.index
              and Path(man.loc[int(ids[i]), "path"]).exists()]
    if not picked:
        print("[08] no candidates with images on disk; nothing drawn.")
        return
    k = len(picked)
    print(f"[08] drawing {k} pairs: " + ", ".join(f"{int(ids[i])}/{names[c]}" for i, c in picked))

    R = a.res
    fig, axes = plt.subplots(2, k, figsize=(2.6 * k + 0.9, 5.6), squeeze=False)
    cmap = plt.get_cmap("Reds")
    for j, (i, c) in enumerate(picked):
        img = Image.open(man.loc[int(ids[i]), "path"]).convert("RGB").resize((R, R), Image.BILINEAR)
        img = np.asarray(img)
        bx = groups.get((int(ids[i]), c), np.zeros((0, 4)))
        heat = np.asarray(alpha[i, c], dtype=np.float32).reshape(G, G)
        heat = heat / max(float(heat.max()), 1e-12)
        top, bot = axes[0, j], axes[1, j]
        top.imshow(img)
        for (x0, y0, x1, y1) in bx:
            rect = Rectangle((x0 * R, y0 * R), (x1 - x0) * R, (y1 - y0) * R, fill=False,
                             edgecolor="white", linewidth=1.8)
            rect.set_path_effects([pe.withStroke(linewidth=3.2, foreground="black")])
            top.add_patch(rect)
        top.set_title(f"{names[c]}\np={scores[i, c]:.2f}, box {100 * frac[i, c]:.2f}% "
                      f"({bin_names[int(bins_of[i, c])]})", fontsize=8)
        bot.imshow(img)
        # opacity follows the attention value so the photo shows through where attention is low
        im = bot.imshow(heat, cmap=cmap, alpha=0.15 + 0.75 * heat, vmin=0, vmax=1,
                        extent=(0, R, R, 0), interpolation="nearest")
        for (x0, y0, x1, y1) in bx:
            rect = Rectangle((x0 * R, y0 * R), (x1 - x0) * R, (y1 - y0) * R, fill=False,
                             edgecolor="black", linewidth=1.0, linestyle="--")
            bot.add_patch(rect)
        for ax in (top, bot):
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(0, R); ax.set_ylim(R, 0)
    axes[0, 0].set_ylabel("image + GT box", fontsize=9)
    axes[1, 0].set_ylabel(f"attention ({a.readout})", fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.9, top=0.88, bottom=0.03, wspace=0.08, hspace=0.18)
    cax = fig.add_axes([0.915, 0.08, 0.015, 0.36])
    fig.colorbar(im, cax=cax).set_label("attention / max", fontsize=8)
    fig.suptitle(f"Class-specific attention on small objects, {a.backbone} @ {R}px, {a.split} split",
                 fontsize=10)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[08] saved {out}")
    if a.copy_to:
        cp = Path(a.copy_to) / out.name
        cp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cp, dpi=200)
        print(f"[08] saved {cp}")
    plt.close(fig)


if __name__ == "__main__":
    main()
