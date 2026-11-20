"""Step 4: oracle region-pooling read-out -> results JSON + test scores.

Builds (and caches under the feature store) the per-class GT-region means for
train/val/test, fits one logistic probe per class on the train split, scores the
test split and evaluates per size bin. ``--neg-mode matched`` (default) uses
class-matched random boxes for negatives (read-out ``oracle``);
``--neg-mode global_mean`` is the inflated ablation (read-out ``oracle_globalneg``).

    .venv/bin/python scripts/04_oracle.py --backbone dinov2_vitb14_reg --res 224
    .venv/bin/python scripts/04_oracle.py --neg-mode global_mean
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

SPLITS = ("train", "val", "test")
READOUT_NAME = {"matched": "oracle", "global_mean": "oracle_globalneg"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--neg-mode", choices=("matched", "global_mean"), default="matched")
    p.add_argument("--max-neg", type=int, default=3000, help="train negatives per class (seeded)")
    p.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of the per-class probe")
    p.add_argument("--batch-size", type=int, default=256, help="images per memmap block")
    p.add_argument("--features-root", default="features")
    p.add_argument("--labels", default="data/coco/labels.npz")
    p.add_argument("--boxes", default="data/coco/boxes.parquet")
    p.add_argument("--manifest", default="data/coco/manifest.parquet")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--readout-suffix", default="", help="appended to the read-out name, e.g. _s1 for seed repeats")
    p.add_argument("--force", action="store_true", help="rebuild cached oracle features")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    import pandas as pd
    from lip.data.labels import load_labels
    from lip.eval.results import (evaluate_readout, results_path, save_results, save_scores,
                                  scores_path)
    from lip.features.store import FeatureStore
    from lip.readouts.oracle import cache_oracle_features, fit_oracle_probes, positive_box_stats
    from lip.readouts.pooling import align_labels
    from lip.readouts.probe import macro_ap
    probes_script = import_module("03_probes")  # load_wh / print_bins helpers

    readout = READOUT_NAME[a.neg_mode] + a.readout_suffix
    stores = {s: FeatureStore(a.backbone, a.res, s, root=a.features_root) for s in SPLITS}
    labels = {s: align_labels(stores[s], load_labels(a.labels, split=s)) for s in SPLITS}
    n_classes = labels["train"]["y"].shape[1]
    boxes = pd.read_parquet(a.boxes)
    train_ids = set(int(i) for i in stores["train"].image_ids)
    stats = positive_box_stats(boxes[boxes["image_id"].isin(train_ids)], n_classes)
    print(f"[04] {readout}: {a.backbone} res{a.res}, grid {stores['train'].grid}; "
          f"{len(boxes)} boxes, negatives matched to TRAIN positives of each class")

    # per-split seeds derived from --seed so the three splits get independent box draws
    feats, seeds = {}, {}
    for k, s in enumerate(SPLITS):
        t0 = time.time()
        seeds[s] = a.seed + 1000 * k
        feats[s], _ = cache_oracle_features(stores[s], boxes, labels[s], stats, a.neg_mode,
                                            seed=seeds[s], batch_size=a.batch_size,
                                            force=a.force, verbose=a.verbose)
        print(f"    {s:5s} oracle features {tuple(feats[s].shape)} float16 ({time.time() - t0:.0f}s)")

    t0 = time.time()
    probes = fit_oracle_probes(feats["train"], labels["train"]["y"], max_neg=a.max_neg,
                               seed=a.seed, C=a.C, verbose=a.verbose)
    fit_s = time.time() - t0
    val_map = macro_ap(labels["val"]["y"], probes.predict(feats["val"]))
    scores = probes.predict(feats["test"])
    notes = (f"{readout}: neg_mode={a.neg_mode}; feature = mean of patch tokens in the union of "
             f"GT boxes (positives) / "
             + ("class-matched random box (negatives)" if a.neg_mode == "matched"
                else "global mean of all patches (negatives, inflated ablation)")
             + f"; one sklearn LogisticRegression(C={a.C}, lbfgs, max_iter=1000) per class on "
             f"standardised features; train negatives subsampled to <= {a.max_neg}/class; "
             f"all test images scored; val_map={val_map:.4f}; "
             f"n_pos/class min={int(probes.n_pos_.min())} max={int(probes.n_pos_.max())}; "
             f"seed={a.seed} feature_seeds={seeds}; fit_time_s={fit_s:.0f}")

    width, height = probes_script.load_wh(a.manifest, labels["test"]["image_ids"])
    res = evaluate_readout(scores, labels["test"], readout=readout, backbone=a.backbone,
                           res=a.res, split="test", train_split="train", notes=notes,
                           width=width, height=height, seed=a.seed)
    save_results(res, results_path(readout, a.backbone, a.res, a.out))
    save_scores(scores, scores_path(readout, a.backbone, a.res, a.out))
    probes_script.print_bins(readout, res)
    print(f"    notes: {notes}")


if __name__ == "__main__":
    main()
