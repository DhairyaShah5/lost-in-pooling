"""Fixed global-pooling read-outs over cached tokens (CONTRACTS: read-out names).

Each read-out turns the per-image tokens (CLS (D,), patches (P, D)) into ONE
vector per image; a joint linear probe (`lip.readouts.probe`) then maps it to
80 class scores. Kinds:

* ``cls``       the CLS token                                   (n, D)
* ``mean``      mean over patch tokens                          (n, D)
* ``cls_mean``  concat(cls, mean)                               (n, 2D)
* ``max``       per-dimension max over patch tokens             (n, D)

The patch memmap is never materialised; it is streamed with
``FeatureStore.iter_patch_batches`` and each block is cast to float32.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lip.features.store import FeatureStore

POOL_KINDS = ("cls", "mean", "cls_mean", "max")

#: label keys whose first axis is the image axis (see docs/CONTRACTS.md, labels.npz)
_PER_IMAGE_KEYS = ("image_ids", "split", "y", "max_box_frac", "max_seg_frac")


def pooled_features(store: FeatureStore, kind: str, batch_size: int = 256) -> np.ndarray:
    """Return the pooled read-out ``kind`` for every image in ``store`` as float32."""
    if kind not in POOL_KINDS:
        raise ValueError(f"unknown pooling kind {kind!r}; choose from {POOL_KINDS}")
    if kind == "cls":
        return np.asarray(store.cls, dtype=np.float32)
    if kind == "cls_mean":
        return np.concatenate(
            [pooled_features(store, "cls"), pooled_features(store, "mean", batch_size)], axis=1
        )
    n, d = len(store), int(store.patch.shape[-1])
    out = np.empty((n, d), dtype=np.float32)
    reduce = np.mean if kind == "mean" else np.max
    for sl, block in store.iter_patch_batches(batch_size):
        out[sl] = reduce(block.astype(np.float32, copy=False), axis=1)
    return out


def pooled_path(store: FeatureStore, kind: str, root: Path | str | None = None) -> Path:
    """``{root}/{backbone}/res{R}/{split}/pooled_{kind}.npy``.

    ``root=None`` means the root the store was opened from (``features/`` by default).
    """
    if root is None:
        return store.dir / f"pooled_{kind}.npy"
    backbone, res_dir, split = store.dir.parts[-3:]
    return Path(root) / backbone / res_dir / split / f"pooled_{kind}.npy"


def cache_pooled(store: FeatureStore, kind: str, root: Path | str | None = None,
                 batch_size: int = 256, force: bool = False) -> np.ndarray:
    """Load the pooled array from disk if cached, else compute, save and return it."""
    path = pooled_path(store, kind, root)
    if path.exists() and not force:
        x = np.load(path)
        if x.shape[0] == len(store):
            return x.astype(np.float32, copy=False)
    x = pooled_features(store, kind, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x


def align_labels(store: FeatureStore, labels: dict) -> dict:
    """Reorder a ``load_labels(split=...)`` dict so its per-image rows follow ``store`` order.

    Both are in manifest order per the contract, so this is normally the identity; it
    is still asserted so a mismatch between a feature cache and labels fails loudly.
    """
    pos = {int(i): k for k, i in enumerate(labels["image_ids"])}
    try:
        idx = np.array([pos[int(i)] for i in store.image_ids], dtype=np.int64)
    except KeyError as e:  # pragma: no cover - defensive
        raise ValueError(f"image id {e} in the feature store has no label row") from None
    out = dict(labels)
    for k in _PER_IMAGE_KEYS:
        if k in out:
            out[k] = np.asarray(out[k])[idx]
    return out
