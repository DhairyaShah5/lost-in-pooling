"""NIH ChestX-ray14 in the same contract as COCO (docs/CONTRACTS.md): manifest, labels, boxes.

Classes: the 14 findings. Size variable: bbox area / (1024*1024) from BBox_List_2017.csv, which
annotates 984 boxes (8 findings) on ~880 images. Test split = every boxed image on disk;
train/val = the remaining images, patient-disjoint from test, seeded. Pairs that carry the
finding label but no box get y=1 and max_box_frac=0, so they count as positives overall but
enter no size bin (the harness ignores bin -1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis",
            "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]
BBOX_ALIASES = {"Infiltrate": "Infiltration"}
SIDE = 1024


def image_id_of(name: str) -> int:
    stem = name.replace(".png", "")
    pid, fu = stem.split("_")
    return int(pid) * 1000 + int(fu)


def build(root: Path = Path("data/nih_cxr"), val_frac: float = 0.1, seed: int = 0) -> None:
    root = Path(root)
    on_disk = {p.name: p for p in root.glob("images/**/*.png")}
    entry = pd.read_csv(root / "Data_Entry_2017.csv")
    entry = entry[entry["Image Index"].isin(on_disk)].reset_index(drop=True)
    bbox = pd.read_csv(root / "BBox_List_2017.csv")
    bbox = bbox.iloc[:, :6]
    bbox.columns = ["Image Index", "Finding Label", "x", "y", "w", "h"]
    bbox["Finding Label"] = bbox["Finding Label"].replace(BBOX_ALIASES)
    bbox = bbox[bbox["Image Index"].isin(on_disk)].reset_index(drop=True)

    cidx = {f: k for k, f in enumerate(FINDINGS)}
    C = len(FINDINGS)
    boxed_images = set(bbox["Image Index"])
    test_patients = set(entry.loc[entry["Image Index"].isin(boxed_images), "Patient ID"])
    rest = entry[~entry["Patient ID"].isin(test_patients)]
    patients = np.array(sorted(rest["Patient ID"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val = int(round(val_frac * len(patients)))
    val_patients = set(patients[:n_val].tolist())

    def split_of(row):
        if row["Image Index"] in boxed_images:
            return "test"
        if row["Patient ID"] in test_patients:
            return "drop"  # same patient as a test image: excluded to keep splits patient-disjoint
        return "val" if row["Patient ID"] in val_patients else "train"

    entry["split"] = entry.apply(split_of, axis=1)
    entry = entry[entry["split"] != "drop"].reset_index(drop=True)
    entry["image_id"] = entry["Image Index"].map(image_id_of).astype(np.int64)
    manifest = pd.DataFrame({
        "image_id": entry["image_id"], "file_name": entry["Image Index"], "split": entry["split"],
        "width": np.int32(SIDE), "height": np.int32(SIDE),
        "path": entry["Image Index"].map(lambda n: str(on_disk[n].relative_to(root.parent.parent))),
    })
    N = len(manifest)
    row_of = {int(i): k for k, i in enumerate(manifest["image_id"])}

    y = np.zeros((N, C), dtype=np.uint8)
    for k, labs in enumerate(entry["Finding Labels"]):
        for f in str(labs).split("|"):
            if f in cidx:
                y[k, cidx[f]] = 1
    max_box = np.zeros((N, C), dtype=np.float32)
    rows = []
    for _, b in bbox.iterrows():
        r = row_of.get(image_id_of(b["Image Index"]))
        c = cidx.get(b["Finding Label"])
        if r is None or c is None:
            continue
        x0, y0 = max(0.0, b["x"] / SIDE), max(0.0, b["y"] / SIDE)
        x1, y1 = min(1.0, (b["x"] + b["w"]) / SIDE), min(1.0, (b["y"] + b["h"]) / SIDE)
        bf = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if bf <= 0:
            continue
        y[r, c] = 1
        max_box[r, c] = max(max_box[r, c], bf)
        rows.append((int(manifest["image_id"].iloc[r]), c, x0, y0, x1, y1, bf, bf, 0))
    boxes = pd.DataFrame(rows, columns=["image_id", "cat_idx", "x0", "y0", "x1", "y1", "box_frac", "seg_frac", "iscrowd"])
    boxes = boxes.astype({"image_id": np.int64, "cat_idx": np.int16, "x0": np.float32, "y0": np.float32,
                          "x1": np.float32, "y1": np.float32, "box_frac": np.float32, "seg_frac": np.float32,
                          "iscrowd": np.int8})

    manifest.to_parquet(root / "manifest.parquet", index=False)
    boxes.to_parquet(root / "boxes.parquet", index=False)
    np.savez(root / "labels.npz", image_ids=manifest["image_id"].to_numpy(), split=manifest["split"].to_numpy().astype(str),
             y=y, max_box_frac=max_box, max_seg_frac=max_box.copy(),
             cat_ids=np.arange(C, dtype=np.int64), cat_names=np.array(FINDINGS))
    sp = manifest["split"].to_numpy()
    print(f"manifest: {N} images  " + "  ".join(f"{s}={int((sp == s).sum())}" for s in ("train", "val", "test")))
    print(f"boxes: {len(boxes)} on {boxes['image_id'].nunique()} images; boxed (image, finding) pairs per finding:")
    tb = boxes.groupby("cat_idx")["image_id"].nunique()
    print("  " + ", ".join(f"{FINDINGS[c]}={int(n)}" for c, n in tb.items()))
    edges = [0, 0.001, 0.003, 0.01, 0.03, 0.10, 1.0]
    bt = max_box[sp == "test"]
    pos = bt[bt > 0]
    print("  boxed pairs per size bin:", {f"{lo*100:g}-{hi*100:g}%": int(((pos >= lo) & (pos < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])})
    print("  train positives per finding:", dict(zip(FINDINGS, y[sp == "train"].sum(0).tolist())))
