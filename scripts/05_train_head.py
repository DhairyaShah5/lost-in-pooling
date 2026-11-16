"""Step 5: train the gated attention MIL head from image-level labels, evaluate on test.

Trains `attn` / `attn_ms` (with `_ent{lam}` suffix when lam != 0) on the cached tokens,
predicts the test split, writes the results JSON + scores, saves the attention maps,
runs the box-based localization analysis and saves the checkpoint.

    .venv/bin/python scripts/05_train_head.py --backbone dinov2_vitb14_reg --res 224 \
        --kind attn --lam 0.0 --epochs 20 --lr 3e-4 --hidden 256 --batch-size 128
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def readout_name(kind: str, lam: float) -> str:
    return kind if lam == 0 else f"{kind}_ent{lam:g}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--kind", default="attn", choices=["attn", "attn_ms"])
    p.add_argument("--lam", type=float, default=0.0, help="entropy regulariser weight")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-pos", type=int, default=10, help="min positives per class per bin for the macro average")
    p.add_argument("--readout-suffix", default="", help="appended to the read-out name, e.g. _s1 for seed repeats")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results")
    p.add_argument("--features-root", default="features")
    p.add_argument("--labels", default="data/coco/labels.npz")
    p.add_argument("--boxes", default="data/coco/boxes.parquet")
    p.add_argument("--manifest", default="data/coco/manifest.parquet",
                   help="used for original width/height (COCO S/M/L block); optional")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--test-split", default="test")
    a = p.parse_args()

    import pandas as pd
    import torch
    from lip.data.labels import load_labels
    from lip.features.store import FeatureStore
    from lip.models.train_head import (attention_path, attention_store, checkpoint_path,
                                       fit_head, predict_store, save_attention, save_head)
    from lip.eval.localization import (format_summary, localization_path, localization_table,
                                       save_localization, summarise_localization)
    from lip.eval import results as R

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    readout = readout_name(a.kind, a.lam) + a.readout_suffix
    print(f"[05] readout={readout} backbone={a.backbone} res={a.res}")

    st_tr = FeatureStore(a.backbone, a.res, a.train_split, root=a.features_root)
    st_va = FeatureStore(a.backbone, a.res, a.val_split, root=a.features_root)
    st_te = FeatureStore(a.backbone, a.res, a.test_split, root=a.features_root)
    lab_tr = load_labels(a.labels, split=a.train_split)
    lab_va = load_labels(a.labels, split=a.val_split)
    lab_te = load_labels(a.labels, split=a.test_split)

    t0 = time.time()
    model, hist = fit_head(st_tr, lab_tr, st_va, lab_va, model_kind=a.kind, lam=a.lam,
                           epochs=a.epochs, lr=a.lr, weight_decay=a.weight_decay,
                           batch_size=a.batch_size, hidden=a.hidden, device=a.device,
                           seed=a.seed, verbose=True, patience=a.patience)
    train_secs = time.time() - t0
    n_params = hist["n_params"]
    notes = (f"gated attention MIL head ({a.kind}); epochs={a.epochs} lr={a.lr:g} "
             f"wd={a.weight_decay:g} lam={a.lam:g} hidden={a.hidden} batch={a.batch_size} "
             f"dropout=0.1 seed={a.seed} best_epoch={hist['best_epoch']} "
             f"val_map={hist['best_val_map']:.4f} params={n_params} "
             f"train_secs={train_secs:.0f} device={hist['config']['device']}")
    print(f"[05] {notes}")

    # checkpoint + history
    ckpt = save_head(model, checkpoint_path(readout, a.backbone, a.res, root=a.checkpoints), history=hist)
    hist_path = out / f"history__{readout}__{a.backbone}__res{a.res}.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(hist, indent=2))
    print(f"[05] checkpoint -> {ckpt}   history -> {hist_path}")

    # test predictions, results JSON + scores
    scores = predict_store(model, st_te, batch_size=256, device=a.device)
    width = height = None
    if Path(a.manifest).exists():
        man = pd.read_parquet(a.manifest).set_index("image_id")
        width = man.loc[lab_te["image_ids"], "width"].to_numpy()
        height = man.loc[lab_te["image_ids"], "height"].to_numpy()
    results = R.evaluate_readout(scores, lab_te, readout, a.backbone, a.res, min_pos=a.min_pos,
                                 split=a.test_split, train_split=a.train_split, notes=notes,
                                 width=width, height=height, seed=a.seed)
    rpath = R.save_results(results, R.results_path(readout, a.backbone, a.res, results_dir=out))
    spath = R.save_scores(scores, R.scores_path(readout, a.backbone, a.res, results_dir=out))
    per_bin = results["per_bin"]
    line = " | ".join(f"{b}: {v['auroc_macro']:.3f}" for b, v in per_bin.items()
                      if v.get("auroc_macro") is not None)
    ov = results["overall"]
    print(f"[05] test per-bin macro AUROC  {line}")
    print(f"[05] test overall AUROC {ov.get('auroc_macro', float('nan')):.4f}  "
          f"mAP {ov.get('map', float('nan')):.4f}  -> {rpath}, {spath}")

    # attention maps on the fine grid + localization analysis (boxes used only here)
    alpha = attention_store(model, st_te, batch_size=256, device=a.device)
    apath = save_attention(alpha, attention_path(readout, a.backbone, a.res, root=out / "attention"))
    print(f"[05] attention maps {alpha.shape} float16 -> {apath}")
    boxes = pd.read_parquet(a.boxes)
    boxes = boxes[boxes["image_id"].isin(set(int(i) for i in st_te.image_ids))]
    df = localization_table(alpha, boxes, lab_te, grid=st_te.grid, image_ids=st_te.image_ids)
    summary = summarise_localization(df, name=readout)
    summary.update({"readout": readout, "backbone": a.backbone, "res": a.res,
                    "split": a.test_split, "seed": a.seed, "notes": notes})
    lpath = save_localization(summary, localization_path(readout, a.backbone, a.res, root=out))
    print(format_summary(summary))
    print(f"[05] localization -> {lpath}")


if __name__ == "__main__":
    main()
