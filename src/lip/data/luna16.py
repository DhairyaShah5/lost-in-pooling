"""LUNA16 (Setio et al., 2017) as 2D axial slices in the COCO/NIH contract (data/luna16/).

Positives: for every annotated nodule, the axial slice through its centre and the neighbouring slices within
its radius, each with the nodule's square box (diameter in pixels). Negatives: lung-containing slices of the
same scans at least `neg_margin_mm` away in z from every nodule centre. Images are 8-bit lung-window PNGs.
image_id = scan_index * 1000 + slice_index, so `image_id // 1000` is the scan (the "patient" for folds).

Two label files share one image set: `labels.npz` puts every slice in "test" (scan-disjoint CV read-outs) and
`labels_head.npz` splits scans train/val/test for the image-level head. Both are built from `manifest.parquet`,
which carries both split columns; `02_extract.py` is pointed at one or the other through `--manifest`.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CATEGORIES = ["Nodule"]
LUNG_WINDOW = (-1000.0, 400.0)
DIAMETER_BINS = [(0, 5), (5, 8), (8, 12), (12, 20), (20, 100)]


def unzip_if_needed(zip_path: Path, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        return
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def to_uint8(slice_hu: np.ndarray) -> np.ndarray:
    lo, hi = LUNG_WINDOW
    return np.clip((slice_hu - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def build(root: Path = Path("data/luna16"), subsets=("subset0", "subset1"), neg_per_scan: int = 6,
          neg_margin_mm: float = 20.0, seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.3) -> None:
    import SimpleITK as sitk
    root = Path(root); raw = root / "raw"; img_dir = root / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(raw / "annotations.csv")
    seg_dir = raw / "seg-lungs-LUNA16"
    if (raw / "seg-lungs-LUNA16.zip").exists():
        unzip_if_needed(raw / "seg-lungs-LUNA16.zip", seg_dir)
    seg_files = {p.stem: p for p in seg_dir.glob("**/*.mhd")}
    mhds = []
    for s in subsets:
        if (raw / f"{s}.zip").exists():
            unzip_if_needed(raw / f"{s}.zip", raw / s)
        mhds += sorted((raw / s).glob("**/*.mhd"))
    print(f"[luna16] {len(mhds)} scans in {list(subsets)}; {len(ann)} annotated nodules overall; {len(seg_files)} lung masks")
    rng = np.random.default_rng(seed)
    rows, boxes = [], []
    for si, mhd in enumerate(mhds):
        uid = mhd.stem
        img = sitk.ReadImage(str(mhd))
        vol = sitk.GetArrayFromImage(img)                       # (Z, Y, X) in HU
        nz, ny, nx = vol.shape
        sx, sy, sz = img.GetSpacing()
        nod = ann[ann["seriesuid"] == uid]
        lung_z = None
        if uid in seg_files:
            seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_files[uid])))
            lung_z = np.where((seg > 0).reshape(nz, -1).sum(1) > 0.01 * ny * nx)[0]
        pos_slices = {}                                          # z -> list of (cx, cy, r_px, d_mm)
        centres_z_mm = []
        for _, n in nod.iterrows():
            ix, iy, iz = img.TransformPhysicalPointToIndex((float(n.coordX), float(n.coordY), float(n.coordZ)))
            if not (0 <= iz < nz and 0 <= iy < ny and 0 <= ix < nx):
                continue
            r_px = 0.5 * float(n.diameter_mm) / float(sx)
            half = int(max(0, np.floor(0.5 * float(n.diameter_mm) / float(sz) - 0.25)))
            centres_z_mm.append(float(n.coordZ))
            for dz in range(-half, half + 1):
                z = iz + dz
                if 0 <= z < nz:
                    pos_slices.setdefault(z, []).append((ix, iy, r_px, float(n.diameter_mm)))
        cand = lung_z if lung_z is not None and len(lung_z) else np.arange(int(0.2 * nz), int(0.8 * nz))
        far = []
        for z in cand:
            zmm = img.TransformIndexToPhysicalPoint((0, 0, int(z)))[2]
            if all(abs(zmm - c) >= neg_margin_mm for c in centres_z_mm):
                far.append(int(z))
        neg = rng.choice(far, size=min(neg_per_scan, len(far)), replace=False) if far else []
        for z, items in list(pos_slices.items()) + [(z, []) for z in neg]:
            image_id = si * 1000 + int(z)
            name = f"{si:03d}_{int(z):04d}.png"
            Image.fromarray(to_uint8(vol[int(z)])).save(img_dir / name)
            rows.append({"image_id": image_id, "file_name": name, "scan": uid, "scan_idx": si, "z": int(z),
                         "width": int(nx), "height": int(ny), "path": str((img_dir / name).relative_to(root.parent.parent)),
                         "n_nodules": len(items), "max_diameter_mm": max([d for *_, d in items], default=0.0)})
            for (cx, cy, r, d) in items:
                x0, y0 = max(0.0, (cx - r) / nx), max(0.0, (cy - r) / ny)
                x1, y1 = min(1.0, (cx + r) / nx), min(1.0, (cy + r) / ny)
                bf = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                if bf > 0:
                    boxes.append((image_id, 0, x0, y0, x1, y1, bf, bf, 0, d))
        if si % 10 == 0:
            print(f"[luna16] scan {si + 1}/{len(mhds)}: {len(pos_slices)} nodule slices, {len(neg)} negatives")
    man = pd.DataFrame(rows)
    man["split"] = "test"
    scans = np.array(sorted(man["scan_idx"].unique())); rng.shuffle(scans)
    n_te = int(round(test_frac * len(scans))); n_va = int(round(val_frac * len(scans)))
    split_of = {int(s): ("test" if i < n_te else "val" if i < n_te + n_va else "train") for i, s in enumerate(scans)}
    man["split_head"] = man["scan_idx"].map(split_of)
    bx = pd.DataFrame(boxes, columns=["image_id", "cat_idx", "x0", "y0", "x1", "y1", "box_frac", "seg_frac", "iscrowd", "diameter_mm"])
    y = (man["n_nodules"].to_numpy() > 0).astype(np.uint8)[:, None]
    max_box = np.zeros((len(man), 1), dtype=np.float32)
    if len(bx):
        mb = bx.groupby("image_id")["box_frac"].max()
        max_box[:, 0] = man["image_id"].map(mb).fillna(0.0).to_numpy(dtype=np.float32)
    man.to_parquet(root / "manifest.parquet", index=False)
    man_head = man.drop(columns=["split"]).rename(columns={"split_head": "split"})
    man_head.to_parquet(root / "manifest_head.parquet", index=False)
    bx.to_parquet(root / "boxes.parquet", index=False)
    for fname, split in (("labels.npz", man["split"]), ("labels_head.npz", man["split_head"])):
        np.savez(root / fname, image_ids=man["image_id"].to_numpy(), split=split.to_numpy().astype(str), y=y,
                 max_box_frac=max_box, max_seg_frac=max_box.copy(), cat_ids=np.arange(1, dtype=np.int64),
                 cat_names=np.array(CATEGORIES))
    print(f"[luna16] {len(man)} slices ({int(y.sum())} with nodules) from {len(mhds)} scans; {len(bx)} boxes; "
          f"head split: {man['split_head'].value_counts().to_dict()}")
    d = bx["diameter_mm"].to_numpy()
    print("[luna16] nodule boxes per diameter bin:", {f"{lo}-{hi}mm": int(((d >= lo) & (d < hi)).sum()) for lo, hi in DIAMETER_BINS})
    pos = max_box[max_box > 0]
    edges = [0, 0.001, 0.003, 0.01, 0.03, 0.10, 1.0]
    print("[luna16] slices per relative-area bin:", {f"{lo*100:g}-{hi*100:g}%": int(((pos >= lo) & (pos < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])})
