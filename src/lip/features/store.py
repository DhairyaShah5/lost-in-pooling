"""Read access to cached frozen features. See docs/CONTRACTS.md (Feature store)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURES_ROOT = Path("features")


class FeatureStore:
    """Memory-mapped view of `features/{backbone}/res{R}/{split}/`.

    Never materialise `.patch` fully; index it in batches (machine has 16 GB RAM).
    """

    def __init__(self, backbone: str = "dinov2_vitb14_reg", res: int = 224,
                 split: str = "test", root: Path | str = FEATURES_ROOT):
        self.dir = Path(root) / backbone / f"res{res}" / split
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.cls = np.load(self.dir / "cls.npy", mmap_mode="r")
        self.patch = np.load(self.dir / "patch.npy", mmap_mode="r")
        self.image_ids = np.load(self.dir / "image_ids.npy")
        self._row = {int(i): k for k, i in enumerate(self.image_ids)}

    @property
    def grid(self) -> int:
        return int(self.meta["grid"])

    @property
    def dim(self) -> int:
        return int(self.meta["dim"])

    @property
    def n_patches(self) -> int:
        return int(self.patch.shape[1])

    def __len__(self) -> int:
        return int(len(self.image_ids))

    def rows_for(self, image_ids) -> np.ndarray:
        return np.array([self._row[int(i)] for i in image_ids], dtype=np.int64)

    def patch_batch(self, rows) -> np.ndarray:
        """Gather patch tokens for arbitrary rows (sorted access under the hood)."""
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(rows)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        out = np.asarray(self.patch[rows[order]])
        return out[inv]

    def iter_patch_batches(self, batch_size: int = 256):
        """Yield (row_slice, patch_block) sequentially over the whole split."""
        n = len(self)
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            yield slice(s, e), np.asarray(self.patch[s:e])


def write_meta(dir: Path, **meta) -> None:
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "meta.json").write_text(json.dumps(meta, indent=2))
