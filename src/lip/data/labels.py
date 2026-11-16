"""Load `data/coco/labels.npz` (see docs/CONTRACTS.md), optionally restricted to a split."""
from __future__ import annotations

from pathlib import Path

import numpy as np

LABELS_PATH = Path("data/coco/labels.npz")
SPLITS = ("train", "val", "test")


def load_labels(path: Path | str = LABELS_PATH, split: str | None = None) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if split is not None:
        m = d["split"] == split
        for k in ("image_ids", "split", "y", "max_box_frac", "max_seg_frac"):
            d[k] = d[k][m]
    return d
