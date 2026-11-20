"""Step 14 (chest X-ray arm): read-outs trained by patient-level cross-validation on the boxed images only.

The NIH training images have no boxes, so the oracle (which needs ground-truth regions for its
positives) cannot be trained on them. Following the motivating chest-radiograph study, we train
on the ~880 boxed images themselves with K-fold patient-disjoint cross-validation and score
each held-out fold, for cls / mean / max (pooled linear probes) and the oracle, so that all four
see exactly the same training images. Pairs that carry a finding label but no box are excluded
from oracle training (their region is undefined) and never act as negatives anywhere.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lip.data.labels import load_labels
from lip.eval.results import evaluate_readout, results_path, save_results, save_scores, scores_path
from lip.features.store import FeatureStore
from lip.readouts.oracle import cache_oracle_features, positive_box_stats
from lip.readouts.pooling import align_labels, cache_pooled


def folds_by_patient(image_ids: np.ndarray, k: int, seed: int) -> np.ndarray:
    patients = image_ids // 1000
    uniq = np.unique(patients)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {p: i % k for i, p in enumerate(uniq)}
    return np.array([fold_of[p] for p in patients])


def cv_scores(F_of_class, Y, boxed, folds, C=1.0, seed=0):
    """F_of_class(c) -> (n, d) features for class c. Returns (n, C) sigmoid scores."""
    n, n_cls = Y.shape
    out = np.full((n, n_cls), np.nan, dtype=np.float32)
    for c in range(n_cls):
        F = F_of_class(c).astype(np.float32)
        mu, sd = F.mean(0), F.std(0) + 1e-6
        F = (F - mu) / sd
        pos = boxed[:, c] == 1
        neg = Y[:, c] == 0
        for k in np.unique(folds):
            tr = (folds != k) & (pos | neg)
            te = folds == k
            ytr = pos[tr].astype(int)
            if ytr.sum() < 3 or (1 - ytr).sum() < 3:
                out[te, c] = 0.5
                continue
            clf = LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(F[tr], ytr)
            out[te, c] = clf.predict_proba(F[te])[:, 1]
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--features-root", default="features_cxr")
    p.add_argument("--labels", default="data/nih_cxr/labels.npz")
    p.add_argument("--boxes", default="data/nih_cxr/boxes.parquet")
    p.add_argument("--out", default="results_cxr")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-pos", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--region", choices=("inside", "outside"), default="inside", help="outside = context-only control")
    a = p.parse_args()

    store = FeatureStore(a.backbone, a.res, "test", root=a.features_root)
    lab = align_labels(store, load_labels(a.labels, split="test"))
    Y = lab["y"].astype(np.uint8)
    boxed = (lab["max_box_frac"] > 0).astype(np.uint8)
    boxes = pd.read_parquet(a.boxes)
    boxes = boxes[boxes["image_id"].isin(set(store.image_ids.tolist()))]
    folds = folds_by_patient(store.image_ids, a.folds, a.seed)
    print(f"[14] {a.backbone} res{a.res}: {len(store)} boxed test images, {a.folds}-fold patient CV, "
          f"boxed pairs={int(boxed.sum())}, labelled pairs={int(Y.sum())}")

    stats = positive_box_stats(boxes, Y.shape[1])
    F_or, _ = cache_oracle_features(store, boxes, {**lab, "y": boxed}, stats, neg_mode="matched",
                                    seed=a.seed, root=a.features_root, region=a.region)
    if a.region == "outside":
        readouts = [("context", lambda c: np.asarray(F_or[:, c, :]))]
    else:
        feats = {kind: cache_pooled(store, kind) for kind in ("cls", "mean", "max")}
        readouts = [("cls", lambda c: feats["cls"]), ("mean", lambda c: feats["mean"]),
                    ("max", lambda c: feats["max"]), ("oracle", lambda c: np.asarray(F_or[:, c, :]))]

    for name, fn in readouts:
        scores = cv_scores(fn, Y, boxed, folds, seed=a.seed)
        readout = f"{name}_cv{a.folds}"
        res = evaluate_readout(scores, lab, readout, a.backbone, a.res, split="test", train_split="test-cv",
                               min_pos=a.min_pos,
                               notes=f"{a.folds}-fold patient-disjoint CV on the boxed NIH images; per-class logistic "
                                     f"regression (C=1); positives = boxed pairs, negatives = label-absent; "
                                     f"min_pos={a.min_pos}; seed={a.seed}")
        Path(a.out).mkdir(parents=True, exist_ok=True)
        save_results(res, results_path(readout, a.backbone, a.res, a.out))
        save_scores(scores, scores_path(readout, a.backbone, a.res, a.out))
        pb = res["per_bin"]
        print(f"{readout:12s} " + " ".join(f"{k}={v['auroc_macro']:.3f}({v['n_classes']})" for k, v in pb.items())
              + f" | all auroc={res['overall']['auroc_macro']:.3f} mAP={res['overall']['map']:.3f}")
