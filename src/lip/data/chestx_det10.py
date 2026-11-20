"""ChestX-Det10 (Liu et al., 2020): 3,543 NIH ChestX-ray14 images re-annotated by three radiologists with boxes
for 10 findings. Built into the COCO/NIH contract under data/chestx_det10/ as ONE boxed split ("test"), for
patient-disjoint cross-validated read-outs and for evaluating NIH-trained heads.

The Det10 files are renumbered (e.g. 36200.png), so NIH identities are recovered by matching a 32x32 pixel
hash against the NIH images on disk; matched images get their NIH image_id (patient * 1000 + follow-up), so the
NIH split of the same patient is known and the folds stay patient-disjoint. Unmatched images get a synthetic
patient id >= 100000.
"""
from __future__ import annotations

import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CATEGORIES = ["Atelectasis", "Calcification", "Consolidation", "Effusion", "Emphysema", "Fibrosis", "Fracture",
              "Mass", "Nodule", "Pneumothorax"]
SIDE = 1024
SYNTHETIC_PATIENT0 = 100000


def pixel_hash(path: str) -> str:
    im = Image.open(path).convert("L").resize((32, 32), Image.BOX)
    a = (np.asarray(im, dtype=np.uint8) >> 3).tobytes()          # 32 grey levels: robust to re-encoding
    return hashlib.sha1(a).hexdigest()


def hash_many(paths: list[str], workers: int = 4) -> list[str]:
    with Pool(workers) as pool:
        return list(pool.imap(pixel_hash, paths, chunksize=64))


def build(root: Path = Path("data/chestx_det10"), nih_root: Path = Path("data/nih_cxr"), workers: int = 4) -> None:
    root = Path(root); nih_root = Path(nih_root)
    ann = []
    for split in ("train", "test"):
        for e in json.loads((root / "raw" / f"{split}.json").read_text()):
            ann.append({"file_name": e["file_name"], "det10_split": split, "syms": e["syms"], "boxes": e["boxes"]})
    on_disk = {p.name: p for p in (root / "images_raw").glob("**/*.png")}
    ann = [e for e in ann if e["file_name"] in on_disk]
    print(f"[det10] {len(ann)} annotated images on disk")

    # ---- recover NIH identities by pixel hash -------------------------------------------------------------
    nih_paths = sorted((nih_root / "images").glob("**/*.png"))
    cache = root / "_hashes.json"
    if cache.exists():
        hc = json.loads(cache.read_text())
        nih_hash, det_hash = hc["nih"], [hc["det"][e["file_name"]] for e in ann]
        print(f"[det10] loaded cached hashes for {len(nih_hash)} NIH and {len(det_hash)} Det10 images")
    else:
        print(f"[det10] hashing {len(nih_paths)} NIH images and {len(ann)} Det10 images ({workers} workers)")
        nih_hashes = hash_many([str(p) for p in nih_paths], workers)
        det_hash = hash_many([str(on_disk[e["file_name"]]) for e in ann], workers)
        nih_hash = {}
        for h, p in zip(nih_hashes, nih_paths):
            nih_hash.setdefault(h, []).append(p.name)
        cache.write_text(json.dumps({"nih": nih_hash, "det": {e["file_name"]: h for e, h in zip(ann, det_hash)}}))
    # a hash bucket may hold several NIH files (near-identical follow-ups); confirm with an exact pixel comparison
    def same_pixels(a: Path, b: Path) -> bool:
        return np.array_equal(np.asarray(Image.open(a).convert("L")), np.asarray(Image.open(b).convert("L")))
    nih_by_name = {p.name: p for p in nih_paths}
    entry = pd.read_csv(nih_root / "Data_Entry_2017.csv").set_index("Image Index")
    nih_man = pd.read_parquet(nih_root / "manifest.parquet").set_index("file_name")
    from .nih_cxr import image_id_of
    n_match = 0
    synthetic = SYNTHETIC_PATIENT0
    used_nih = set()
    for e, h in zip(ann, det_hash):
        nih_name = None
        for cand in nih_hash.get(h, []):
            if cand not in used_nih and same_pixels(on_disk[e["file_name"]], nih_by_name[cand]):
                nih_name = cand
                break
        if nih_name is not None:
            used_nih.add(nih_name)
            n_match += 1
            e["nih_file_name"] = nih_name
            e["patient_id"] = int(entry.loc[nih_name, "Patient ID"])
            e["image_id"] = image_id_of(nih_name)
            e["nih_split"] = str(nih_man.loc[nih_name, "split"]) if nih_name in nih_man.index else "unused"
        else:
            e["nih_file_name"] = ""
            e["patient_id"] = synthetic
            e["image_id"] = synthetic * 1000
            e["nih_split"] = "none"
            synthetic += 1
    print(f"[det10] matched {n_match}/{len(ann)} images to NIH files on disk")

    # ---- manifest / labels / boxes -----------------------------------------------------------------------
    C = len(CATEGORIES); cidx = {c: k for k, c in enumerate(CATEGORIES)}
    N = len(ann)
    manifest = pd.DataFrame({
        "image_id": np.array([e["image_id"] for e in ann], dtype=np.int64),
        "file_name": [e["file_name"] for e in ann], "split": "test",
        "width": np.int32(SIDE), "height": np.int32(SIDE),
        "path": [str(on_disk[e["file_name"]].relative_to(root.parent.parent)) for e in ann],
        "det10_split": [e["det10_split"] for e in ann], "nih_file_name": [e["nih_file_name"] for e in ann],
        "patient_id": np.array([e["patient_id"] for e in ann], dtype=np.int64),
        "nih_split": [e["nih_split"] for e in ann],
    })
    assert manifest["image_id"].is_unique, "duplicate image ids (two Det10 files matched the same NIH image?)"
    y = np.zeros((N, C), dtype=np.uint8)
    max_box = np.zeros((N, C), dtype=np.float32)
    rows = []
    for k, e in enumerate(ann):
        for sym, b in zip(e["syms"], e["boxes"]):
            c = cidx.get(sym)
            if c is None:
                continue
            x0, y0 = max(0.0, b[0] / SIDE), max(0.0, b[1] / SIDE)
            x1, y1 = min(1.0, b[2] / SIDE), min(1.0, b[3] / SIDE)
            bf = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if bf <= 0:
                continue
            y[k, c] = 1
            max_box[k, c] = max(max_box[k, c], bf)
            rows.append((int(manifest["image_id"].iloc[k]), c, x0, y0, x1, y1, bf, bf, 0))
    boxes = pd.DataFrame(rows, columns=["image_id", "cat_idx", "x0", "y0", "x1", "y1", "box_frac", "seg_frac", "iscrowd"])
    boxes = boxes.astype({"image_id": np.int64, "cat_idx": np.int16, "x0": np.float32, "y0": np.float32,
                          "x1": np.float32, "y1": np.float32, "box_frac": np.float32, "seg_frac": np.float32,
                          "iscrowd": np.int8})
    manifest.to_parquet(root / "manifest.parquet", index=False)
    boxes.to_parquet(root / "boxes.parquet", index=False)
    np.savez(root / "labels.npz", image_ids=manifest["image_id"].to_numpy(), split=manifest["split"].to_numpy().astype(str),
             y=y, max_box_frac=max_box, max_seg_frac=max_box.copy(),
             cat_ids=np.arange(C, dtype=np.int64), cat_names=np.array(CATEGORIES))
    vc = manifest["nih_split"].value_counts().to_dict()
    print(f"[det10] manifest {N} images; NIH split of the matched images: {vc}")
    print(f"[det10] boxes {len(boxes)} on {boxes['image_id'].nunique()} images; boxed pairs per finding: "
          + ", ".join(f"{CATEGORIES[c]}={int(n)}" for c, n in boxes.groupby('cat_idx')['image_id'].nunique().items()))
    edges = [0, 0.001, 0.003, 0.01, 0.03, 0.10, 1.0]
    pos = max_box[max_box > 0]
    print("[det10] boxed pairs per size bin:", {f"{lo*100:g}-{hi*100:g}%": int(((pos >= lo) & (pos < hi)).sum())
                                              for lo, hi in zip(edges[:-1], edges[1:])})
