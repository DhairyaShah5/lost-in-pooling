"""Oracle region-pooling read-out: the "told where to look" upper bound.

Motivation. In the chest-X-ray study that motivates this project, a frozen
encoder's global embedding missed small findings, yet a probe fitted on the
embedding of the *annotated region* recovered them: the evidence was in the
tokens, the pooling threw it away. The oracle reproduces that upper bound on
COCO: for class ``c`` and an image where ``y[i, c] = 1`` the feature is the mean
of the patch tokens inside the union of the ground-truth boxes of class ``c``
(``box_to_patch_mask`` semantics from docs/CONTRACTS.md; a sub-patch box maps to
the single patch containing its centre). Nothing is learned about *where* to
look, so the gap between the oracle and the global read-outs is the amount of
small-object evidence that pooling discards.

Matched negatives. A probe would trivially separate "mean of a few patches" from
"mean of all patches" by box statistics alone, so for ``y[i, c] = 0`` the
feature is the mean inside a RANDOM box whose (area fraction, aspect ratio) pair
is drawn (seeded) from the empirical distribution of class ``c``'s positive
instances in the TRAIN split (joint draw: one train instance of ``c`` is picked
uniformly and its ``box_frac`` and aspect are copied), placed uniformly at
random so that it lies inside the image. ``neg_mode="global_mean"`` is the
ablation where negatives use the mean of ALL patches instead (read-out name
``oracle_globalneg``); it shows the inflation the matched design avoids.

Because the feature depends on the class, ONE binary probe is trained per class
(``OracleProbes``: sklearn ``LogisticRegression`` on standardised features, see
its docstring for why), using all train positives and at most ``max_neg``
seeded train negatives; prediction uses ALL test images.

Cache layout (chosen for simplicity of per-class slicing):
``features/{backbone}/res{R}/{split}/oracle_{neg_mode}.npy`` float16 (n, 80, d)
where ``[i, c]`` is the GT-region mean if ``y[i, c] = 1`` else the negative-box
mean; and ``oracle_{neg_mode}_boxes.npy`` float32 (n, 80, 4) with the sampled
negative box for every (image, class) pair (also filled for positives, where it
is unused). Sizes at D=768: train 1.1 GB, val 0.12 GB, test 0.61 GB per mode.
Open with ``mmap_mode="r"`` and slice ``[:, c, :]`` or row blocks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lip.features.store import FeatureStore

N_CLASSES = 80
NEG_MODES = ("matched", "global_mean")


def _as_y(labels) -> np.ndarray:
    return np.asarray(labels["y"] if isinstance(labels, dict) else labels)


# --------------------------------------------------------------------- masks
def box_masks(boxes, grid: int) -> np.ndarray:
    """Vectorised contract helper: (k, 4) normalised xyxy boxes -> (k, G*G) bool.

    A patch (r, c) with centre ((c+0.5)/G, (r+0.5)/G) is inside the box if the
    centre lies in ``[x0, x1) x [y0, y1)`` (same convention as
    ``lip.eval.bins.box_to_patch_mask``). A box containing no patch centre maps
    to the single patch containing its own centre. Row-major grid, top-left first.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    G = int(grid)
    centres = (np.arange(G, dtype=np.float64) + 0.5) / G
    cx_in = (centres[None, :] >= b[:, 0:1]) & (centres[None, :] < b[:, 2:3])  # (k, G) cols
    cy_in = (centres[None, :] >= b[:, 1:2]) & (centres[None, :] < b[:, 3:4])  # (k, G) rows
    m = (cy_in[:, :, None] & cx_in[:, None, :]).reshape(len(b), G * G)
    empty = ~m.any(axis=1)
    if empty.any():
        bx = 0.5 * (b[empty, 0] + b[empty, 2])
        by = 0.5 * (b[empty, 1] + b[empty, 3])
        col = np.clip(np.floor(bx * G), 0, G - 1).astype(np.int64)
        row = np.clip(np.floor(by * G), 0, G - 1).astype(np.int64)
        m[np.flatnonzero(empty), row * G + col] = True
    return m


def union_patch_mask(boxes, grid: int) -> np.ndarray:
    """Union of ``box_masks`` over (k, 4) boxes -> (G*G,) bool."""
    return box_masks(boxes, grid).any(axis=0)


# ------------------------------------------------------- box statistics
@dataclass
class BoxStats:
    """Per-class empirical (box_frac, aspect=w/h) pairs of TRAIN positives."""
    per_class: list[np.ndarray] = field(default_factory=list)  # each (k_c, 2)
    pooled: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))

    def pairs(self, class_idx: int) -> np.ndarray:
        p = self.per_class[class_idx]
        if len(p) == 0:
            p = self.pooled
        if len(p) == 0:  # nothing at all: fall back to a 10% square
            p = np.array([[0.1, 1.0]], dtype=np.float32)
        return p


def positive_box_stats(boxes_df: pd.DataFrame, n_classes: int = N_CLASSES) -> BoxStats:
    """Build ``BoxStats`` from the TRAIN rows of ``data/coco/boxes.parquet``."""
    w = (boxes_df["x1"] - boxes_df["x0"]).to_numpy(np.float64)
    h = (boxes_df["y1"] - boxes_df["y0"]).to_numpy(np.float64)
    frac = boxes_df["box_frac"].to_numpy(np.float64) if "box_frac" in boxes_df else w * h
    ok = (w > 0) & (h > 0) & (frac > 0)
    cat = boxes_df["cat_idx"].to_numpy(np.int64)
    pairs = np.stack([frac, w / np.where(h > 0, h, 1.0)], axis=1).astype(np.float32)
    per_class = [pairs[ok & (cat == c)] for c in range(n_classes)]
    return BoxStats(per_class=per_class, pooled=pairs[ok])


def sample_matched_boxes(stats: BoxStats, class_idx: int, n: int,
                         rng: np.random.Generator) -> np.ndarray:
    """``n`` random boxes (n, 4) xyxy in [0,1] with class-matched (area, aspect) pairs."""
    pairs = stats.pairs(class_idx)
    pick = pairs[rng.integers(0, len(pairs), size=n)]
    frac, aspect = pick[:, 0].astype(np.float64), pick[:, 1].astype(np.float64)
    # w*h = frac, w/h = aspect; if an extreme aspect pushes one side past the
    # image edge, clamp that side to 1 and stretch the other to keep the area.
    w = np.minimum(np.sqrt(frac * aspect), 1.0)
    h = np.minimum(frac / w, 1.0)
    w = np.minimum(frac / h, 1.0)
    x0 = rng.uniform(0.0, 1.0, size=n) * (1.0 - w)
    y0 = rng.uniform(0.0, 1.0, size=n) * (1.0 - h)
    return np.stack([x0, y0, x0 + w, y0 + h], axis=1).astype(np.float32)


# -------------------------------------------------------- GT box lookup
def _gt_pairs(store: FeatureStore, boxes_df: pd.DataFrame, n_classes: int):
    """Union GT masks per (row, class) pair present in ``store``.

    Returns ``(keys, U)``: sorted int64 keys ``row * n_classes + cat`` and the
    matching (n_pairs, P) bool union masks.
    """
    keep = boxes_df["image_id"].isin(set(int(i) for i in store.image_ids)).to_numpy()
    df = boxes_df[keep]
    P = store.grid * store.grid
    if len(df) == 0:
        return np.zeros(0, np.int64), np.zeros((0, P), bool)
    rows = store.rows_for(df["image_id"].to_numpy())
    cats = df["cat_idx"].to_numpy(np.int64)
    boxes = df[["x0", "y0", "x1", "y1"]].to_numpy(np.float32)
    keys_all = rows * n_classes + cats
    keys, inv = np.unique(keys_all, return_inverse=True)
    counts = np.zeros((len(keys), P), dtype=np.int32)
    np.add.at(counts, inv, box_masks(boxes, store.grid).astype(np.int32))
    return keys, counts > 0


def _normalise(w: np.ndarray) -> np.ndarray:
    s = w.sum(axis=-1, keepdims=True)
    return w / np.maximum(s, 1.0)


# ------------------------------------------------------- feature builders
def build_oracle_features(store: FeatureStore, boxes_df: pd.DataFrame, labels, class_idx: int,
                          rng: np.random.Generator, stats: BoxStats | None = None,
                          neg_mode: str = "matched", batch_size: int = 256) -> np.ndarray:
    """Oracle features for ONE class: (n, d) float32.

    Positives (``y[:, class_idx] = 1``): mean of patch tokens inside the union of
    that class's GT boxes. Negatives: mean inside a matched random box (``stats``
    from the TRAIN split; defaults to this split's own boxes) or, with
    ``neg_mode="global_mean"``, the mean of all patches.
    """
    if neg_mode not in NEG_MODES:
        raise ValueError(f"neg_mode must be one of {NEG_MODES}")
    y = _as_y(labels)[:, class_idx].astype(bool)
    n, P, d = len(store), store.grid * store.grid, int(store.patch.shape[-1])
    assert len(y) == n, "labels must follow store row order"
    stats = stats if stats is not None else positive_box_stats(boxes_df, _as_y(labels).shape[1])
    n_classes = _as_y(labels).shape[1]

    keys, U = _gt_pairs(store, boxes_df, n_classes)
    W = np.zeros((n, P), dtype=np.float32)
    if neg_mode == "matched":
        W[:] = box_masks(sample_matched_boxes(stats, class_idx, n, rng), store.grid)
    else:
        W[:] = 1.0
    sel = (keys % n_classes) == class_idx
    pos_rows = keys[sel] // n_classes
    W[pos_rows] = U[sel]
    missing = y & ~np.isin(np.arange(n), pos_rows)
    if missing.any():
        raise ValueError(f"{missing.sum()} positives of class {class_idx} have no GT box rows")
    W = _normalise(W)

    out = np.empty((n, d), dtype=np.float32)
    for sl, block in store.iter_patch_batches(batch_size):
        x = block.astype(np.float32, copy=False)
        out[sl] = np.einsum("bp,bpd->bd", W[sl], x)
    return out


def build_oracle_features_all(store: FeatureStore, boxes_df: pd.DataFrame, labels,
                              stats: BoxStats, neg_mode: str = "matched", seed: int = 0,
                              batch_size: int = 256, out_path: Path | str | None = None,
                              verbose: bool = False, region: str = "inside") -> tuple[np.ndarray, np.ndarray]:
    """One pass over the memmap -> features (n, C, d) float16 and neg boxes (n, C, 4).

    If ``out_path`` is given the feature array is an ``open_memmap`` written in
    place (never fully resident); otherwise an in-memory float16 array is returned.
    """
    if neg_mode not in NEG_MODES:
        raise ValueError(f"neg_mode must be one of {NEG_MODES}")
    Y = _as_y(labels).astype(bool)
    n, C = Y.shape
    G = store.grid
    P, d = G * G, int(store.patch.shape[-1])
    assert n == len(store), "labels must follow store row order"
    rng = np.random.default_rng(seed)

    neg_boxes = np.empty((n, C, 4), dtype=np.float32)
    for c in range(C):  # deterministic per-class draw order
        neg_boxes[:, c] = sample_matched_boxes(stats, c, n, rng)

    keys, U = _gt_pairs(store, boxes_df, C)
    pair_rows = keys // C
    pair_cats = keys % C
    covered = np.zeros((n, C), dtype=bool)
    covered[pair_rows, pair_cats] = True
    if (Y & ~covered).any():
        bad = int((Y & ~covered).sum())
        raise ValueError(f"{bad} positive (image, class) pairs have no GT boxes in boxes_df")

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        feats = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, C, d))
    else:
        feats = np.empty((n, C, d), dtype=np.float16)

    t0 = time.time()
    for sl, block in store.iter_patch_batches(batch_size):
        s, e = sl.start, sl.stop
        b = e - s
        if neg_mode == "matched":
            W = box_masks(neg_boxes[s:e].reshape(-1, 4), G).reshape(b, C, P).astype(np.float32)
        else:
            W = np.ones((b, C, P), dtype=np.float32)
        lo, hi = np.searchsorted(keys, [s * C, e * C])
        W[pair_rows[lo:hi] - s, pair_cats[lo:hi]] = U[lo:hi]
        if region == "outside":          # context-only control: every token the box read-out does NOT see
            W = (W <= 0).astype(np.float32)
        W = _normalise(W)
        x = block.astype(np.float32, copy=False)
        feats[s:e] = np.matmul(W, x).astype(np.float16)
        if verbose:
            print(f"[oracle] {e}/{n} images ({time.time() - t0:.1f}s)", end="\r")
    if verbose:
        print()
    if out_path is not None:
        feats.flush()
    return feats, neg_boxes


def oracle_paths(store: FeatureStore, neg_mode: str, root: Path | str | None = None, region: str = "inside"):
    base = store.dir if root is None else Path(root).joinpath(*store.dir.parts[-3:])
    tag = "" if region == "inside" else f"_{region}"
    return base / f"oracle_{neg_mode}{tag}.npy", base / f"oracle_{neg_mode}{tag}_boxes.npy"


def cache_oracle_features(store: FeatureStore, boxes_df: pd.DataFrame, labels, stats: BoxStats,
                          neg_mode: str = "matched", seed: int = 0, root=None,
                          batch_size: int = 256, force: bool = False, verbose: bool = False,
                          region: str = "inside"):
    """Build (once) and return the memmapped (n, C, d) float16 oracle features + neg boxes.
    ``region="outside"`` builds the context-only control (mean of the tokens outside the box)."""
    fp, bp = oracle_paths(store, neg_mode, root, region)
    n, C = _as_y(labels).shape
    if fp.exists() and bp.exists() and not force:
        f = np.load(fp, mmap_mode="r")
        if f.shape[:2] == (n, C):
            return f, np.load(bp)
    _, neg_boxes = build_oracle_features_all(store, boxes_df, labels, stats, neg_mode, seed,
                                             batch_size, out_path=fp, verbose=verbose, region=region)
    np.save(bp, neg_boxes)
    return np.load(fp, mmap_mode="r"), neg_boxes


# ---------------------------------------------------------------- probes
class OracleProbes:
    """One standardised logistic-regression probe per class on oracle features.

    sklearn ``LogisticRegression(C=1.0, lbfgs, max_iter=1000)`` was chosen over a
    torch ``nn.Linear(d, 1)`` loop: with <= ~7000 x 768 samples per class LBFGS
    converges in well under a second per class on CPU, is deterministic, and needs
    no device shuffling; 80 classes take about a minute. Weights are stored as
    arrays so ``predict`` is one einsum over row blocks of the (n, C, d) memmap.
    """

    def __init__(self):
        self.W: np.ndarray | None = None      # (C, d)
        self.b: np.ndarray | None = None      # (C,)
        self.mu: np.ndarray | None = None     # (C, d)
        self.sd: np.ndarray | None = None     # (C, d)
        self.n_pos_: np.ndarray | None = None
        self.n_neg_: np.ndarray | None = None
        self.config_: dict = {}

    def fit(self, F_train, Y_train, max_neg: int = 3000, seed: int = 0, C: float = 1.0,
            max_iter: int = 1000, verbose: bool = False) -> "OracleProbes":
        Y = _as_y(Y_train).astype(bool)
        n, n_classes = Y.shape
        d = int(F_train.shape[-1])
        rng = np.random.default_rng(seed)
        self.W = np.zeros((n_classes, d), np.float32)
        self.b = np.zeros(n_classes, np.float32)
        self.mu = np.zeros((n_classes, d), np.float32)
        self.sd = np.ones((n_classes, d), np.float32)
        self.n_pos_ = np.zeros(n_classes, np.int64)
        self.n_neg_ = np.zeros(n_classes, np.int64)
        self.config_ = dict(max_neg=max_neg, seed=seed, C=C, max_iter=max_iter, solver="lbfgs")
        t0 = time.time()
        for c in range(n_classes):
            pos = np.flatnonzero(Y[:, c])
            neg = np.flatnonzero(~Y[:, c])
            if len(neg) > max_neg:
                neg = np.sort(rng.choice(neg, size=max_neg, replace=False))
            self.n_pos_[c], self.n_neg_[c] = len(pos), len(neg)
            if len(pos) == 0 or len(neg) == 0:
                continue  # score stays at sigmoid(0)=0.5 for an untrainable class
            rows = np.sort(np.concatenate([pos, neg]))
            X = np.asarray(F_train[rows, c, :], dtype=np.float32)
            y = Y[rows, c].astype(np.int64)
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
            clf.fit((X - mu) / sd, y)
            self.W[c] = clf.coef_[0]
            self.b[c] = clf.intercept_[0]
            self.mu[c], self.sd[c] = mu, sd
            if verbose:
                print(f"[oracle] class {c:2d}: {len(pos)} pos / {len(neg)} neg "
                      f"({time.time() - t0:.1f}s)", end="\r")
        if verbose:
            print()
        return self

    def predict(self, F, block: int = 512) -> np.ndarray:
        """(n, C, d) features (memmap ok) -> (n, C) sigmoid scores, float32."""
        if self.W is None:
            raise RuntimeError("call fit() first")
        n = F.shape[0]
        out = np.empty((n, self.W.shape[0]), dtype=np.float32)
        for s in range(0, n, block):
            x = (np.asarray(F[s:s + block], dtype=np.float32) - self.mu) / self.sd
            z = np.einsum("ncd,cd->nc", x, self.W) + self.b
            out[s:s + block] = 1.0 / (1.0 + np.exp(-z))
        return out


def fit_oracle_probes(F_train, Y_train, max_neg: int = 3000, seed: int = 0, C: float = 1.0,
                      verbose: bool = False) -> OracleProbes:
    """Train one logistic probe per class on (n, C, d) train oracle features."""
    return OracleProbes().fit(F_train, Y_train, max_neg=max_neg, seed=seed, C=C, verbose=verbose)
