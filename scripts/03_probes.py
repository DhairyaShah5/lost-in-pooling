"""Step 3: fixed read-outs -> results JSON + test scores (docs/CONTRACTS.md).

For each kind in ``--kinds``: build features for train/val/test from the feature
store, fit (linear probe on pooled features, or the Max-MIL head on streamed
tokens), predict the test split, evaluate per size bin and save
``results/{kind}__{backbone}__res{R}.json`` and ``results/scores/...npy``.

    .venv/bin/python scripts/03_probes.py --backbone dinov2_vitb14_reg --res 224 \
        --kinds cls,mean,cls_mean,max,maxmil
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


def load_wh(manifest: str, image_ids: np.ndarray):
    """Original (width, height) per image for COCO S/M/L; ``(None, None)`` if no manifest."""
    p = Path(manifest)
    if not p.exists():
        return None, None
    import pandas as pd
    m = pd.read_parquet(p, columns=["image_id", "width", "height"]).set_index("image_id")
    m = m.loc[np.asarray(image_ids, dtype=np.int64)]
    return m["width"].to_numpy(np.int64), m["height"].to_numpy(np.int64)


def fmt(v) -> str:
    return "  nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def print_bins(readout: str, res: dict) -> None:
    bins = " ".join(f"{name}={fmt(b['auroc_macro'])}" for name, b in res["per_bin"].items())
    o = res["overall"]
    print(f"{readout:14s} {bins} | overall auroc={fmt(o['auroc_macro'])} mAP={fmt(o['map'])}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--kinds", default="cls,mean,cls_mean,max,maxmil",
                   help="comma-separated subset of cls,mean,cls_mean,max,maxmil")
    p.add_argument("--epochs", type=int, default=40, help="linear-probe epochs")
    p.add_argument("--mil-epochs", type=int, default=15, help="max-MIL epochs (early-stopped)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512, help="linear-probe batch size")
    p.add_argument("--mil-batch-size", type=int, default=128, help="max-MIL images per batch")
    p.add_argument("--mil-topk", type=int, default=1, help="top-k mean pooling for max-MIL (1 = max)")
    p.add_argument("--mil-patience", type=int, default=3)
    p.add_argument("--features-root", default="features")
    p.add_argument("--labels", default="data/coco/labels.npz")
    p.add_argument("--manifest", default="data/coco/manifest.parquet")
    p.add_argument("--out", default="results")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--device", default="auto", help="auto (mps if available), mps or cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    # imports kept inside main so the module stays importable without the eval package
    import torch
    from lip.data.labels import load_labels
    from lip.eval.results import (evaluate_readout, results_path, save_results, save_scores,
                                  scores_path)
    from lip.features.store import FeatureStore
    from lip.readouts.maxmil import fit_maxmil
    from lip.readouts.pooling import POOL_KINDS, align_labels, cache_pooled
    from lip.readouts.probe import LinearProbe

    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    bad = [k for k in kinds if k not in POOL_KINDS and k != "maxmil"]
    if bad:
        raise SystemExit(f"unknown kinds {bad}; choose from {POOL_KINDS + ('maxmil',)}")

    stores = {s: FeatureStore(a.backbone, a.res, s, root=a.features_root) for s in SPLITS}
    labels = {s: align_labels(stores[s], load_labels(a.labels, split=s)) for s in SPLITS}
    width, height = load_wh(a.manifest, labels["test"]["image_ids"])
    print(f"[03] {a.backbone} res{a.res}: train {len(stores['train'])} / val {len(stores['val'])} "
          f"/ test {len(stores['test'])} images, grid {stores['train'].grid}, dim {stores['train'].dim}")

    for kind in kinds:
        t0 = time.time()
        if kind == "maxmil":
            model, hist = fit_maxmil(
                stores["train"], labels["train"]["y"], stores["val"], labels["val"]["y"],
                epochs=a.mil_epochs, lr=a.lr, weight_decay=a.weight_decay,
                batch_size=a.mil_batch_size, topk=a.mil_topk, patience=a.mil_patience,
                device=a.device, seed=a.seed, verbose=a.verbose)
            scores = model.predict_store(stores["test"], a.mil_batch_size)
            notes = (f"maxmil: per-patch nn.Linear, top{a.mil_topk} over patches, "
                     f"BCEWithLogits, AdamW lr={a.lr} wd={a.weight_decay} bs={a.mil_batch_size} "
                     f"cosine, epochs={a.mil_epochs} patience={a.mil_patience} "
                     f"ran={len(hist['epochs'])} best_epoch={hist['best_epoch']} "
                     f"val_map={hist['best_val_map']:.4f} seed={a.seed} "
                     f"train_time_s={time.time() - t0:.0f}")
            ck = Path(a.checkpoints) / f"maxmil__{a.backbone}__res{a.res}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "history": hist, "dim": stores["train"].dim, "topk": a.mil_topk}, ck)
        else:
            X = {s: cache_pooled(stores[s], kind) for s in SPLITS}
            probe = LinearProbe().fit(
                X["train"], labels["train"]["y"], X["val"], labels["val"]["y"],
                epochs=a.epochs, lr=a.lr, weight_decay=a.weight_decay,
                batch_size=a.batch_size, device=a.device, seed=a.seed, verbose=a.verbose)
            scores = probe.predict(X["test"])
            notes = (f"linear probe on {kind} (d={X['train'].shape[1]}), standardised, "
                     f"BCEWithLogits, AdamW, cosine: {probe.summary()} "
                     f"train_time_s={time.time() - t0:.0f}")

        res = evaluate_readout(scores, labels["test"], readout=kind, backbone=a.backbone,
                               res=a.res, split="test", train_split="train", notes=notes,
                               width=width, height=height, seed=a.seed)
        save_results(res, results_path(kind, a.backbone, a.res, a.out))
        save_scores(scores, scores_path(kind, a.backbone, a.res, a.out))
        print_bins(kind, res)
        print(f"    notes: {notes}")


if __name__ == "__main__":
    main()
