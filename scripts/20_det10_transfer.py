"""Step 20: evaluate NIH-trained read-outs on ChestX-Det10 (a second set of radiologist boxes, 10 findings).

Heads trained on the NIH image-level labels (checkpoints) and CLS/mean probes refitted on the NIH training
cache are scored on the Det10 tokens; the 8 findings shared with NIH are evaluated on the Det10 images whose
patient is NOT in our NIH train/val splits (identities recovered by pixel hash in step 18).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NIH_FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis",
                "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--heads", default="attn,attn_s1,attn_s2,attn_ms")
    p.add_argument("--probes", default="cls,mean")
    p.add_argument("--features-root", default="features_det10")
    p.add_argument("--nih-features-root", default="features_cxr")
    p.add_argument("--labels", default="data/chestx_det10/labels.npz")
    p.add_argument("--manifest", default="data/chestx_det10/manifest.parquet")
    p.add_argument("--boxes", default="data/chestx_det10/boxes.parquet")
    p.add_argument("--nih-labels", default="data/nih_cxr/labels.npz")
    p.add_argument("--nih-manifest", default="data/nih_cxr/manifest.parquet")
    p.add_argument("--checkpoints", default="checkpoints_cxr")
    p.add_argument("--out", default="results_det10")
    p.add_argument("--min-pos", type=int, default=5)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    import pandas as pd
    from lip.data.labels import load_labels
    from lip.eval import results as R
    from lip.eval.localization import (format_summary, localization_path, localization_table, save_localization,
                                       summarise_localization)
    from lip.features.store import FeatureStore
    from lip.models.train_head import attention_store, load_head, predict_store
    from lip.readouts.pooling import align_labels, cache_pooled
    from lip.readouts.probe import LinearProbe

    bb, res = a.backbone, a.res
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(bb, res, "test", root=a.features_root)
    lab = align_labels(store, load_labels(a.labels, split="test"))
    det_names = [str(c) for c in lab["cat_names"]]
    shared = [c for c in det_names if c in NIH_FINDINGS]
    det_idx = np.array([det_names.index(c) for c in shared]); nih_idx = np.array([NIH_FINDINGS.index(c) for c in shared])

    man = pd.read_parquet(a.manifest).set_index("image_id")
    nih_man = pd.read_parquet(a.nih_manifest)
    trainval_patients = set((nih_man.loc[nih_man["split"].isin(["train", "val"]), "image_id"] // 1000).tolist())
    pid = man.loc[lab["image_ids"], "patient_id"].to_numpy()
    matched = (man.loc[lab["image_ids"], "nih_file_name"].to_numpy() != "")
    eligible = ~np.isin(pid, list(trainval_patients))
    print(f"[20] {bb} res{res}: {len(store)} Det10 images; {int(matched.sum())} matched to NIH files; "
          f"{int((~eligible).sum())} excluded (patient in our NIH train/val); {int(eligible.sum())} evaluated; "
          f"shared findings: {shared}")

    def reduced(mask):
        return {"image_ids": lab["image_ids"][mask], "y": lab["y"][mask][:, det_idx],
                "max_box_frac": lab["max_box_frac"][mask][:, det_idx], "max_seg_frac": lab["max_seg_frac"][mask][:, det_idx],
                "cat_ids": np.arange(len(shared)), "cat_names": np.array(shared), "split": lab["split"][mask]}

    boxes = pd.read_parquet(a.boxes)
    remap = {int(d): k for k, d in enumerate(det_idx)}
    boxes = boxes[boxes["cat_idx"].isin(list(remap))].copy(); boxes["cat_idx"] = boxes["cat_idx"].map(remap).astype(np.int16)
    subsets = [("", eligible, reduced(eligible)), ("_m", eligible & matched, reduced(eligible & matched))]
    print(f"[20] evaluation subsets: all eligible = {int(eligible.sum())}, matched-only = {int((eligible & matched).sum())}")
    summary = {}

    def evaluate(scores14, name, notes, alpha=None):
        for tag, mask, labm in subsets:
            nm = name + tag
            s = scores14[:, nih_idx][mask]
            r = R.evaluate_readout(s, labm, nm, bb, res, min_pos=a.min_pos, split="det10", train_split="nih-train",
                                   notes=notes + (" [matched-to-NIH images only]" if tag else ""), seed=0)
            R.save_results(r, R.results_path(nm, bb, res, results_dir=out)); R.save_scores(s, R.scores_path(nm, bb, res, results_dir=out))
            pb = r["per_bin"]
            print(f"[20] {nm:18s} " + " ".join(f"{k}={v['auroc_macro']:.3f}({v['n_classes']})" for k, v in pb.items() if v.get("auroc_macro") is not None)
                  + f" | all auroc={r['overall']['auroc_macro']:.3f} mAP={r['overall']['map']:.3f} n={int(mask.sum())}")
            summary[nm] = {"overall": r["overall"], "per_bin": {k: v.get("auroc_macro") for k, v in pb.items()}, "n_images": int(mask.sum())}
            if alpha is not None:
                ids = labm["image_ids"]
                bx = boxes[boxes["image_id"].isin(set(int(i) for i in ids))]
                df = localization_table(alpha[mask][:, nih_idx], bx, labm, grid=store.grid, image_ids=ids)
                summ = summarise_localization(df, name=nm)
                summ.update({"readout": nm, "backbone": bb, "res": res, "split": "det10", "notes": notes})
                save_localization(summ, localization_path(nm, bb, res, root=out))
                print(format_summary(summ))

    for h in [x for x in a.heads.split(",") if x]:
        ck = Path(a.checkpoints) / f"{h}__{bb}__res{res}.pt"
        if not ck.exists():
            print(f"[20] no checkpoint {ck}; skipped"); continue
        model = load_head(ck, device=a.device)
        scores = predict_store(model, store, batch_size=256, device=a.device)
        alpha = attention_store(model, store, batch_size=256, device=a.device) if h.startswith("attn") and "ms" not in h else None
        evaluate(scores, f"{h}_nih", f"NIH-trained head {ck.name} applied to ChestX-Det10 tokens; evaluated on the 8 shared findings "
                                     f"over {int(eligible.sum())} images whose patient is not in our NIH train/val", alpha=alpha)

    if a.probes:
        nih_tr = FeatureStore(bb, res, "train", root=a.nih_features_root); nih_va = FeatureStore(bb, res, "val", root=a.nih_features_root)
        ytr = align_labels(nih_tr, load_labels(a.nih_labels, split="train"))["y"]; yva = align_labels(nih_va, load_labels(a.nih_labels, split="val"))["y"]
        for kind in [k for k in a.probes.split(",") if k]:
            Xtr, Xva = cache_pooled(nih_tr, kind), cache_pooled(nih_va, kind)
            probe = LinearProbe().fit(Xtr, ytr, Xva, yva, epochs=40, lr=1e-3, weight_decay=1e-4, batch_size=512, device="cpu", seed=0)
            scores = probe.predict(cache_pooled(store, kind))
            evaluate(scores, f"{kind}_nih", f"linear probe on NIH-train {kind} features (same recipe as step 3) applied to ChestX-Det10")
    (out / f"transfer__{bb}__res{res}.json").write_text(json.dumps({"backbone": bb, "res": res, "n_eval": int(eligible.sum()),
                                                                     "n_matched": int(matched.sum()), "shared": shared, "readouts": summary}, indent=2))


if __name__ == "__main__":
    main()
