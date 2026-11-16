"""Build manifest.parquet, labels.npz and boxes.parquet from COCO annotations (docs/CONTRACTS.md)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lip.data.subset import N_TRAIN, load_subset


def _load(ann_path: Path) -> dict:
    with open(ann_path) as f:
        return json.load(f)


def build(root: Path = Path("data/coco")) -> None:
    root = Path(root)
    val = _load(root / "annotations" / "instances_val2017.json")
    train = _load(root / "annotations" / "instances_train2017.json")

    cats = sorted(val["categories"], key=lambda c: c["id"])
    cat_ids = np.array([c["id"] for c in cats], dtype=np.int64)
    cat_names = np.array([c["name"] for c in cats])
    cat_index = {int(cid): k for k, cid in enumerate(cat_ids)}
    C = len(cat_ids)

    subset = load_subset(root / "train_subset_ids.json")
    sub_ids = [int(i) for i in subset["ids"]]
    n_train = int(subset.get("n_train", N_TRAIN))
    split_of = {i: ("train" if k < n_train else "val") for k, i in enumerate(sub_ids)}

    rows = []
    img_meta = {}
    for img in val["images"]:
        iid = int(img["id"])
        rows.append((iid, img["file_name"], "test", img["width"], img["height"],
                     f"data/coco/images/val2017/{img['file_name']}"))
        img_meta[iid] = (img["width"], img["height"])
    train_imgs = {int(img["id"]): img for img in train["images"]}
    for iid in sub_ids:
        img = train_imgs[iid]
        p = root / "images" / "train2017" / img["file_name"]
        if not p.exists():
            continue  # failed download; dropped from the manifest
        rows.append((iid, img["file_name"], split_of[iid], img["width"], img["height"],
                     f"data/coco/images/train2017/{img['file_name']}"))
        img_meta[iid] = (img["width"], img["height"])

    manifest = pd.DataFrame(rows, columns=["image_id", "file_name", "split", "width", "height", "path"])
    manifest["image_id"] = manifest["image_id"].astype(np.int64)
    manifest["width"] = manifest["width"].astype(np.int32)
    manifest["height"] = manifest["height"].astype(np.int32)
    N = len(manifest)
    row_of = {int(i): k for k, i in enumerate(manifest["image_id"].to_numpy())}

    y = np.zeros((N, C), dtype=np.uint8)
    max_box = np.zeros((N, C), dtype=np.float32)
    max_seg = np.zeros((N, C), dtype=np.float32)
    box_rows = []
    for ann in list(val["annotations"]) + list(train["annotations"]):
        iid = int(ann["image_id"])
        r = row_of.get(iid)
        if r is None or ann.get("iscrowd", 0):
            continue
        c = cat_index[int(ann["category_id"])]
        W, H = img_meta[iid]
        x, yy, w, h = ann["bbox"]
        x0, y0 = max(0.0, x / W), max(0.0, yy / H)
        x1, y1 = min(1.0, (x + w) / W), min(1.0, (yy + h) / H)
        bf = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
        sf = float(ann.get("area", 0.0)) / (W * H)
        if bf <= 0:
            continue
        y[r, c] = 1
        max_box[r, c] = max(max_box[r, c], bf)
        max_seg[r, c] = max(max_seg[r, c], sf)
        box_rows.append((iid, c, x0, y0, x1, y1, bf, sf, 0))

    boxes = pd.DataFrame(box_rows, columns=["image_id", "cat_idx", "x0", "y0", "x1", "y1",
                                            "box_frac", "seg_frac", "iscrowd"])
    boxes = boxes.astype({"image_id": np.int64, "cat_idx": np.int16, "x0": np.float32, "y0": np.float32,
                          "x1": np.float32, "y1": np.float32, "box_frac": np.float32,
                          "seg_frac": np.float32, "iscrowd": np.int8})

    manifest.to_parquet(root / "manifest.parquet", index=False)
    boxes.to_parquet(root / "boxes.parquet", index=False)
    np.savez(root / "labels.npz", image_ids=manifest["image_id"].to_numpy(), split=manifest["split"].to_numpy().astype(str),
             y=y, max_box_frac=max_box, max_seg_frac=max_seg, cat_ids=cat_ids, cat_names=cat_names)

    print(f"manifest: {N} images  " + "  ".join(f"{s}={int((manifest.split == s).sum())}" for s in ("train", "val", "test")))
    print(f"boxes: {len(boxes)} non-crowd instances; positives (image,class) pairs: {int(y.sum())}")
