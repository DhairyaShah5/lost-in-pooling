"""Step 19: tiled-teacher distillation on cached tokens.

Teachers: (tiled) the R px head applied to the four quadrants of the 2R px tokens, max over quadrants;
          (hi) a head trained at 2R px. Students: R px heads trained on the images present in BOTH caches
with BCE(labels) + beta * BCE(teacher probabilities). Also evaluates the tiled teacher itself at test time
(`attn_tiled` at res 2R: inference-time tiling of the R px head, no training).

    .venv/bin/python scripts/19_distill.py --backbone dinov2_vitb14_reg --suffix _train5k --students tiled,hi,none
    .venv/bin/python scripts/19_distill.py --backbone dinov2_vitb14_reg --features-root features_cxr \
        --labels data/nih_cxr/labels.npz --manifest data/nih_cxr/manifest.parquet --boxes data/nih_cxr/boxes.parquet \
        --out results_cxr --checkpoints checkpoints_cxr --min-pos 5 --teacher-hi checkpoints_cxr/attn_full__dinov2_vitb14_reg__res448.pt \
        --students tiled,hi
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res-lo", type=int, default=224)
    p.add_argument("--res-hi", type=int, default=448)
    p.add_argument("--teacher-lo", default=None, help="checkpoint of the R px head (default checkpoints/attn__BB__resR.pt)")
    p.add_argument("--teacher-hi", default=None, help="checkpoint of the 2R px head (default checkpoints/attn__BB__res2R.pt if present)")
    p.add_argument("--students", default="tiled,hi,none", help="comma list of tiled, hi, none (none = plain head on the same rows)")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--suffix", default="", help="appended to student read-out names, e.g. _train5k")
    p.add_argument("--min-pos", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--features-root", default="features")
    p.add_argument("--labels", default="data/coco/labels.npz")
    p.add_argument("--manifest", default="data/coco/manifest.parquet")
    p.add_argument("--boxes", default="data/coco/boxes.parquet")
    p.add_argument("--out", default="results")
    p.add_argument("--checkpoints", default="checkpoints")
    a = p.parse_args()

    import pandas as pd
    import torch
    from lip.data.labels import load_labels
    from lip.eval import results as R
    from lip.eval.localization import (format_summary, localization_path, localization_table, save_localization,
                                       summarise_localization)
    from lip.features.store import FeatureStore
    from lip.models.distill import fit_student, head_probs, tiled_scores
    from lip.models.train_head import (attention_path, attention_store, checkpoint_path, labels_for_store, load_head,
                                       predict_store, save_attention, save_head)
    from lip.readouts.pooling import align_labels

    bb, lo, hi = a.backbone, a.res_lo, a.res_hi
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    st_lo = {s: FeatureStore(bb, lo, s, root=a.features_root) for s in ("train", "val", "test")}
    st_hi = {s: FeatureStore(bb, hi, s, root=a.features_root) for s in ("train", "test")}
    lab = {s: load_labels(a.labels, split=s) for s in ("train", "val", "test")}
    assert st_hi["train"].grid == 2 * st_lo["train"].grid, "hi grid must be twice the lo grid"

    ids_tr = [int(i) for i in st_hi["train"].image_ids if int(i) in st_lo["train"]._row]
    rows_lo = st_lo["train"].rows_for(ids_tr)
    rows_hi = st_hi["train"].rows_for(ids_tr)
    y_tr_all = labels_for_store(st_lo["train"], lab["train"])
    y_tr = y_tr_all[rows_lo]
    print(f"[19] {bb}: {len(ids_tr)} training images in both the {lo} and {hi} px caches; "
          f"grids {st_lo['train'].grid} -> {st_hi['train'].grid}")

    teacher_lo = a.teacher_lo or str(checkpoint_path("attn", bb, lo, root=a.checkpoints))
    head_lo = load_head(teacher_lo, device=a.device)
    man = pd.read_parquet(a.manifest).set_index("image_id") if Path(a.manifest).exists() else None
    boxes_all = pd.read_parquet(a.boxes)

    def wh(lab_te):
        if man is None:
            return None, None
        return man.loc[lab_te["image_ids"], "width"].to_numpy(), man.loc[lab_te["image_ids"], "height"].to_numpy()

    def evaluate(scores, store, readout, res, notes, seed, alpha=None):
        lab_te = align_labels(store, lab["test"])
        width, height = wh(lab_te)
        res_d = R.evaluate_readout(scores, lab_te, readout, bb, res, min_pos=a.min_pos, split="test",
                                   train_split="train", notes=notes, width=width, height=height, seed=seed)
        R.save_results(res_d, R.results_path(readout, bb, res, results_dir=out))
        R.save_scores(scores, R.scores_path(readout, bb, res, results_dir=out))
        pb = res_d["per_bin"]
        line = " | ".join(f"{b}: {v['auroc_macro']:.3f}" for b, v in pb.items() if v.get("auroc_macro") is not None)
        ov = res_d["overall"]
        print(f"[19] {readout:28s} {line} | all {ov.get('auroc_macro', float('nan')):.4f} mAP {ov.get('map', float('nan')):.4f}")
        if alpha is not None:
            save_attention(alpha, attention_path(readout, bb, res, root=out / "attention"))
            bx = boxes_all[boxes_all["image_id"].isin(set(int(i) for i in store.image_ids))]
            df = localization_table(alpha, bx, lab_te, grid=store.grid, image_ids=store.image_ids)
            summ = summarise_localization(df, name=readout)
            summ.update({"readout": readout, "backbone": bb, "res": res, "split": "test", "seed": seed, "notes": notes})
            save_localization(summ, localization_path(readout, bb, res, root=out))
            print(format_summary(summ))
        return res_d

    # ---- teachers --------------------------------------------------------------------------------------
    t0 = time.time()
    T_tiled_tr = tiled_scores(head_lo, st_hi["train"], device=a.device)[rows_hi]
    T_tiled_te = tiled_scores(head_lo, st_hi["test"], device=a.device)
    print(f"[19] tiled teacher scored train/test in {time.time() - t0:.0f}s")
    evaluate(T_tiled_te, st_hi["test"], "attn_tiled", hi,
             notes=f"inference-time tiling: the {lo} px head {Path(teacher_lo).name} applied to the four quadrants of the "
                   f"{hi} px tokens, max over quadrants; no training", seed=0)
    teachers = {"tiled": T_tiled_tr, "none": None}
    teacher_hi = a.teacher_hi or str(checkpoint_path("attn", bb, hi, root=a.checkpoints))
    if Path(teacher_hi).exists():
        head_hi = load_head(teacher_hi, device=a.device)
        teachers["hi"] = head_probs(head_hi, st_hi["train"], device=a.device)[rows_hi]
    else:
        print(f"[19] no hi-res teacher at {teacher_hi}; skipping the 'hi' student")

    # ---- students --------------------------------------------------------------------------------------
    summary = {"backbone": bb, "res_lo": lo, "res_hi": hi, "n_train": len(ids_tr), "beta": a.beta, "students": {}}
    for seed in [int(s) for s in a.seeds.split(",") if s != ""]:
        for student in [s for s in a.students.split(",") if s]:
            if student not in teachers:
                continue
            soft = teachers[student]
            beta = 0.0 if soft is None else a.beta
            name = ("attn" if soft is None else f"attn_kd_{student}") + a.suffix + (f"_s{seed}" if seed else "")
            torch.manual_seed(seed); np.random.seed(seed)
            t0 = time.time()
            model, hist = fit_student(st_lo["train"], rows_lo, y_tr, soft, st_lo["val"], lab["val"], model_kind="attn",
                                      beta=beta, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, device=a.device,
                                      seed=seed, verbose=False)
            notes = (f"{lo} px gated attention head trained on {len(ids_tr)} images with BCE(labels)"
                     + (f" + {beta:g} * BCE(teacher probs from {student} teacher)" if soft is not None else "")
                     + f"; epochs={a.epochs} lr={a.lr:g} seed={seed} best_epoch={hist['best_epoch']} "
                       f"val_map={hist['best_val_map']:.4f} train_secs={time.time() - t0:.0f}")
            save_head(model, checkpoint_path(name, bb, lo, root=a.checkpoints), history=hist)
            (out / f"history__{name}__{bb}__res{lo}.json").write_text(json.dumps(hist, indent=2))
            scores = predict_store(model, st_lo["test"], batch_size=256, device=a.device)
            alpha = attention_store(model, st_lo["test"], batch_size=256, device=a.device)
            res_d = evaluate(scores, st_lo["test"], name, lo, notes, seed, alpha=alpha)
            summary["students"].setdefault(student, []).append({"seed": seed, "readout": name,
                                                                "overall": res_d["overall"],
                                                                "per_bin": {b: v.get("auroc_macro") for b, v in res_d["per_bin"].items()}})
    (out / f"distill__{bb}__res{lo}.json").write_text(json.dumps(summary, indent=2))
    print(f"[19] summary -> {out / f'distill__{bb}__res{lo}.json'}")


if __name__ == "__main__":
    main()
