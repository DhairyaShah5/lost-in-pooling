"""Synthetic-data tests for the fixed read-outs (pooling, linear probe, Max-MIL, oracle).

A tiny FeatureStore (grid 4, dim 16) is written to a tmp dir. Each image has 1-2
objects from a handful of "active" classes; a planted unit direction per class is
added to the patch tokens inside the object's box, so the label is linearly
decodable from the right patches. Everything runs on CPU in seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from lip.features.store import FeatureStore
from lip.readouts import oracle as orc
from lip.readouts.maxmil import MaxMIL, fit_maxmil
from lip.readouts.pooling import POOL_KINDS, align_labels, cache_pooled, pooled_features
from lip.readouts.probe import LinearProbe, macro_ap

GRID, DIM, N_CLASSES = 4, 16, 80
ACTIVE = (0, 7, 23, 41, 62, 79)
BACKBONE, RES = "synthetic_vit", 56  # grid 4 -> "res" 56 = 4 * 14
AMP = 6.0


def _ref_mask(box, grid):
    """Independent, loop-based contract mask used to cross-check `oracle.box_masks`."""
    x0, y0, x1, y1 = box
    m = np.zeros(grid * grid, bool)
    for r in range(grid):
        for c in range(grid):
            cx, cy = (c + 0.5) / grid, (r + 0.5) / grid
            if x0 <= cx < x1 and y0 <= cy < y1:
                m[r * grid + c] = True
    if not m.any():
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        m[min(grid - 1, int(cy * grid)) * grid + min(grid - 1, int(cx * grid))] = True
    return m


def _make_split(root: Path, split: str, n: int, id_base: int, V: np.ndarray, rng):
    """Write one synthetic feature store + labels dict + boxes DataFrame."""
    P = GRID * GRID
    patch = rng.normal(0.0, 1.0, size=(n, P, DIM)).astype(np.float32)
    y = np.zeros((n, N_CLASSES), np.uint8)
    mbf = np.zeros((n, N_CLASSES), np.float32)
    rows = []
    image_ids = id_base + np.arange(n, dtype=np.int64)
    for i in range(n):
        k = int(rng.integers(1, 3))
        for c in rng.choice(ACTIVE, size=k, replace=False):
            w = float(rng.uniform(0.3, 2.5) / GRID)
            h = float(rng.uniform(0.3, 2.5) / GRID)
            x0 = float(rng.uniform(0, 1 - w))
            y0 = float(rng.uniform(0, 1 - h))
            box = (x0, y0, x0 + w, y0 + h)
            patch[i, _ref_mask(box, GRID)] += AMP * V[c]
            y[i, c] = 1
            mbf[i, c] = max(mbf[i, c], w * h)
            rows.append(dict(image_id=int(image_ids[i]), cat_idx=np.int16(c), x0=box[0], y0=box[1],
                             x1=box[2], y1=box[3], box_frac=w * h, seg_frac=0.8 * w * h))
    cls = patch.mean(axis=1) + rng.normal(0.0, 0.3, size=(n, DIM)).astype(np.float32)

    d = root / BACKBONE / f"res{RES}" / split
    d.mkdir(parents=True)
    np.save(d / "cls.npy", cls.astype(np.float16))
    pm = np.lib.format.open_memmap(d / "patch.npy", mode="w+", dtype=np.float16, shape=(n, P, DIM))
    pm[:] = patch.astype(np.float16)
    pm.flush()
    del pm
    np.save(d / "image_ids.npy", image_ids)
    (d / "meta.json").write_text(json.dumps(dict(
        backbone=BACKBONE, res=RES, patch_size=14, grid=GRID, dim=DIM, n=n, resize="squash",
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
    labels = dict(image_ids=image_ids, split=np.array([split] * n), y=y, max_box_frac=mbf,
                  max_seg_frac=0.8 * mbf, cat_ids=np.arange(N_CLASSES) + 1,
                  cat_names=np.array([f"c{i}" for i in range(N_CLASSES)]))
    boxes = pd.DataFrame(rows)
    for col in ("x0", "y0", "x1", "y1", "box_frac", "seg_frac"):
        boxes[col] = boxes[col].astype(np.float32)
    return labels, boxes, V


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("features")
    rng = np.random.default_rng(0)
    V = rng.normal(size=(N_CLASSES, DIM))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    out = {"root": root, "V": V}
    for split, n, base in (("train", 200, 1), ("val", 100, 10_001), ("test", 100, 20_001)):
        labels, boxes, _ = _make_split(root, split, n, base, V, rng)
        store = FeatureStore(BACKBONE, RES, split, root=root)
        out[split] = dict(store=store, labels=labels, boxes=boxes)
    return out


# ------------------------------------------------------------------ pooling
def test_pooled_features_shapes_and_values(synth):
    st = synth["train"]["store"]
    n = len(st)
    patch = np.asarray(st.patch, dtype=np.float32)
    for kind, width in (("cls", DIM), ("mean", DIM), ("cls_mean", 2 * DIM), ("max", DIM)):
        x = pooled_features(st, kind, batch_size=64)
        assert x.shape == (n, width) and x.dtype == np.float32
    assert np.allclose(pooled_features(st, "mean", 64), patch.mean(1), atol=1e-3)
    assert np.allclose(pooled_features(st, "max", 64), patch.max(1), atol=1e-3)
    assert np.allclose(pooled_features(st, "cls"), np.asarray(st.cls, np.float32))
    cm = pooled_features(st, "cls_mean", 64)
    assert np.allclose(cm[:, :DIM], np.asarray(st.cls, np.float32))
    assert set(POOL_KINDS) == {"cls", "mean", "cls_mean", "max"}
    with pytest.raises(ValueError):
        pooled_features(st, "median")


def test_cache_pooled_roundtrip(synth):
    st = synth["val"]["store"]
    x = cache_pooled(st, "mean", batch_size=32)
    p = st.dir / "pooled_mean.npy"
    assert p.exists()
    x2 = cache_pooled(st, "mean")
    assert np.array_equal(x, x2)
    x3 = cache_pooled(st, "max", root=synth["root"])
    assert (synth["root"] / BACKBONE / f"res{RES}" / "val" / "pooled_max.npy").exists()
    assert x3.shape == (len(st), DIM)


def test_align_labels_identity_and_reorder(synth):
    st, lab = synth["test"]["store"], synth["test"]["labels"]
    a = align_labels(st, lab)
    assert np.array_equal(a["y"], lab["y"])
    perm = np.random.default_rng(1).permutation(len(st))
    shuffled = {k: (v[perm] if k in ("image_ids", "split", "y", "max_box_frac", "max_seg_frac") else v)
                for k, v in lab.items()}
    b = align_labels(st, shuffled)
    assert np.array_equal(b["y"], lab["y"]) and np.array_equal(b["image_ids"], st.image_ids)


# -------------------------------------------------------------- linear probe
def test_linear_probe_learns_planted_signal(synth):
    tr, va, te = synth["train"], synth["val"], synth["test"]
    Xtr = pooled_features(tr["store"], "mean")
    Xva = pooled_features(va["store"], "mean")
    Xte = pooled_features(te["store"], "mean")
    probe = LinearProbe().fit(Xtr, tr["labels"]["y"], Xva, va["labels"]["y"], epochs=30,
                              lr=1e-2, batch_size=64, device="cpu", seed=0)
    assert probe.best_val_map_ > 0.8, probe.history_
    assert 1 <= probe.best_epoch_ <= 30
    S = probe.predict(Xte)
    assert S.shape == (len(te["store"]), N_CLASSES) and S.dtype == np.float32
    assert S.min() >= 0 and S.max() <= 1
    assert macro_ap(te["labels"]["y"], S) > 0.8
    # standardisation statistics are stored and reused
    assert probe.mean_.shape == (DIM,) and probe.std_.shape == (DIM,)
    assert np.all(probe.std_ > 0)


def test_linear_probe_is_deterministic(synth):
    tr, va = synth["train"], synth["val"]
    Xtr = pooled_features(tr["store"], "cls")
    Xva = pooled_features(va["store"], "cls")
    kw = dict(epochs=5, batch_size=64, device="cpu", seed=0)
    p1 = LinearProbe().fit(Xtr, tr["labels"]["y"], Xva, va["labels"]["y"], **kw)
    p2 = LinearProbe().fit(Xtr, tr["labels"]["y"], Xva, va["labels"]["y"], **kw)
    assert np.allclose(p1.predict(Xva), p2.predict(Xva))


# ------------------------------------------------------------------ max-mil
def test_maxmil_learns_planted_signal(synth):
    tr, va, te = synth["train"], synth["val"], synth["test"]
    model, hist = fit_maxmil(tr["store"], tr["labels"]["y"], va["store"], va["labels"]["y"],
                             epochs=10, lr=1e-2, batch_size=32, n_stats=50, device="cpu",
                             seed=0, patience=4)
    assert isinstance(model, MaxMIL)
    assert hist["best_val_map"] > 0.8, hist
    S = model.predict_store(te["store"], batch_size=32)
    assert S.shape == (len(te["store"]), N_CLASSES) and S.dtype == np.float32
    assert macro_ap(te["labels"]["y"], S) > 0.8
    rows = np.array([5, 1, 42])
    ps = model.patch_scores(te["store"], rows)
    assert ps.shape == (3, GRID * GRID, N_CLASSES)
    # the max over patches of the per-patch logits reproduces the image logit
    img = model(model._to_device(te["store"].patch_batch(rows))).detach().cpu().numpy()
    assert np.allclose(ps.max(axis=1), img, atol=1e-4)
    # the top-scoring patch for a present class lies inside that object's box
    te_boxes = te["boxes"]
    hits, total = 0, 0
    for r in rows:
        img_id = int(te["store"].image_ids[r])
        for c in np.flatnonzero(te["labels"]["y"][r]):
            b = te_boxes[(te_boxes.image_id == img_id) & (te_boxes.cat_idx == c)]
            m = orc.union_patch_mask(b[["x0", "y0", "x1", "y1"]].to_numpy(), GRID)
            hits += bool(m[int(ps[list(rows).index(r), :, c].argmax())])
            total += 1
    assert hits == total


def test_maxmil_topk_pooling():
    torch = pytest.importorskip("torch")
    m = MaxMIL(DIM, N_CLASSES, topk=3)
    x = torch.randn(2, 5, DIM)
    pl = m.patch_logits(x)
    ref = pl.topk(3, dim=1).values.mean(1)
    assert torch.allclose(m(x), ref)


# ------------------------------------------------------------------- oracle
def test_box_masks_match_contract():
    rng = np.random.default_rng(3)
    for grid in (4, 16):
        boxes = rng.uniform(0, 1, size=(200, 4))
        boxes = np.concatenate([np.minimum(boxes[:, :2], boxes[:, 2:]),
                                np.maximum(boxes[:, :2], boxes[:, 2:])], axis=1)
        # add sub-patch boxes that contain no patch centre
        tiny = np.array([[0.05, 0.05, 0.06, 0.06], [0.95, 0.5, 0.96, 0.52], [0.30, 0.30, 0.31, 0.31]])
        boxes = np.concatenate([boxes, tiny])
        got = orc.box_masks(boxes, grid)
        ref = np.stack([_ref_mask(b, grid) for b in boxes])
        assert got.shape == (len(boxes), grid * grid)
        assert np.array_equal(got, ref)
        assert got.sum(1).min() >= 1
    # sub-patch -> exactly one patch: the one containing the box centre
    m = orc.box_masks([[0.05, 0.05, 0.06, 0.06]], 4)[0]
    assert m.sum() == 1 and m[0]
    m = orc.box_masks([[0.0, 0.0, 1.0, 1.0]], 4)[0]
    assert m.all()


def test_matched_negative_sampler_matches_area_and_aspect(synth):
    boxes_tr = synth["train"]["boxes"]
    stats = orc.positive_box_stats(boxes_tr, N_CLASSES)
    c = ACTIVE[1]
    pos = boxes_tr[boxes_tr.cat_idx == c]
    pos_frac = pos.box_frac.to_numpy(np.float64)
    pos_aspect = ((pos.x1 - pos.x0) / (pos.y1 - pos.y0)).to_numpy(np.float64)
    rng = np.random.default_rng(0)
    nb = orc.sample_matched_boxes(stats, c, 4000, rng)
    assert nb.shape == (4000, 4)
    assert (nb >= 0).all() and (nb <= 1).all()
    assert (nb[:, 2] > nb[:, 0]).all() and (nb[:, 3] > nb[:, 1]).all()
    area = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
    aspect = (nb[:, 2] - nb[:, 0]) / (nb[:, 3] - nb[:, 1])
    # every sampled (area, aspect) is a copy of an empirical positive pair
    assert np.abs(area[:, None] - pos_frac[None, :]).min(1).max() < 1e-4
    assert np.abs(np.log(aspect)[:, None] - np.log(pos_aspect)[None, :]).min(1).max() < 1e-2
    # and the distribution matches (KS distance to the empirical cdf is small)
    for samp, emp in ((area, pos_frac), (aspect, pos_aspect)):
        grid = np.sort(emp)
        ks = np.abs(np.searchsorted(np.sort(samp), grid, side="right") / len(samp)
                    - np.arange(1, len(grid) + 1) / len(grid)).max()
        assert ks < 0.06, ks
    # positions are uniform: the box centre covers the whole image, not one corner
    cx = 0.5 * (nb[:, 0] + nb[:, 2])
    assert cx.min() < 0.2 and cx.max() > 0.8
    # a class with no train positives falls back to the pooled distribution
    nb2 = orc.sample_matched_boxes(stats, 3, 10, rng)
    assert nb2.shape == (10, 4)
    # deterministic given the seed
    assert np.array_equal(orc.sample_matched_boxes(stats, c, 5, np.random.default_rng(7)),
                          orc.sample_matched_boxes(stats, c, 5, np.random.default_rng(7)))


def test_oracle_features_separate_classes(synth):
    tr = synth["train"]
    V = synth["V"]
    stats = orc.positive_box_stats(tr["boxes"], N_CLASSES)
    c = ACTIVE[2]
    F = orc.build_oracle_features(tr["store"], tr["boxes"], tr["labels"], c,
                                  np.random.default_rng(0), stats=stats, batch_size=64)
    assert F.shape == (len(tr["store"]), DIM) and F.dtype == np.float32
    y = tr["labels"]["y"][:, c]
    proj = F @ V[c]
    assert roc_auc_score(y, proj) > 0.95
    # positives carry about AMP along the planted direction, negatives ~0
    assert proj[y == 1].mean() > 0.6 * AMP and abs(proj[y == 0].mean()) < 1.0
    # positives are computed from GT patches only: recompute one by hand
    i = int(np.flatnonzero(y)[0])
    img_id = int(tr["store"].image_ids[i])
    b = tr["boxes"][(tr["boxes"].image_id == img_id) & (tr["boxes"].cat_idx == c)]
    m = orc.union_patch_mask(b[["x0", "y0", "x1", "y1"]].to_numpy(), GRID)
    ref = np.asarray(tr["store"].patch[i], np.float32)[m].mean(0)
    assert np.allclose(F[i], ref, atol=1e-3)
    # global-mean negatives equal the mean over all patches
    Fg = orc.build_oracle_features(tr["store"], tr["boxes"], tr["labels"], c,
                                   np.random.default_rng(0), stats=stats, neg_mode="global_mean")
    j = int(np.flatnonzero(y == 0)[0])
    assert np.allclose(Fg[j], np.asarray(tr["store"].patch[j], np.float32).mean(0), atol=1e-3)


def test_oracle_all_classes_cache_and_probes(synth):
    tr, va, te = synth["train"], synth["val"], synth["test"]
    V = synth["V"]
    stats = orc.positive_box_stats(tr["boxes"], N_CLASSES)
    Ftr, nb_tr = orc.cache_oracle_features(tr["store"], tr["boxes"], tr["labels"], stats,
                                           neg_mode="matched", seed=0, batch_size=64)
    assert Ftr.shape == (len(tr["store"]), N_CLASSES, DIM) and Ftr.dtype == np.float16
    assert nb_tr.shape == (len(tr["store"]), N_CLASSES, 4)
    assert (tr["store"].dir / "oracle_matched.npy").exists()
    # second call loads the cache
    Ftr2, _ = orc.cache_oracle_features(tr["store"], tr["boxes"], tr["labels"], stats, "matched")
    assert np.array_equal(np.asarray(Ftr), np.asarray(Ftr2))
    # the one-pass builder agrees with the per-class builder (same seed stream per class)
    c = ACTIVE[0]
    rng = np.random.default_rng(0)
    Fc = orc.build_oracle_features(tr["store"], tr["boxes"], tr["labels"], c, rng, stats=stats,
                                   batch_size=64)
    assert np.allclose(np.asarray(Ftr[:, c, :], np.float32), Fc, atol=2e-2)

    Fte, _ = orc.cache_oracle_features(te["store"], te["boxes"], te["labels"], stats, "matched",
                                       seed=2, batch_size=64)
    probes = orc.fit_oracle_probes(Ftr, tr["labels"]["y"], max_neg=50, seed=0)
    assert probes.n_neg_.max() <= 50
    S = probes.predict(Fte, block=33)
    assert S.shape == (len(te["store"]), N_CLASSES) and S.dtype == np.float32
    assert S.min() >= 0 and S.max() <= 1
    yte = te["labels"]["y"]
    assert macro_ap(yte, S) > 0.9
    for c in ACTIVE:
        assert roc_auc_score(yte[:, c], S[:, c]) > 0.95
    # inactive classes are untrainable and score a constant 0.5
    inactive = [c for c in range(N_CLASSES) if c not in ACTIVE]
    assert np.allclose(S[:, inactive], 0.5)

    # global-mean ablation: shapes only (the inflation is a property of real data)
    Fg, _ = orc.cache_oracle_features(va["store"], va["boxes"], va["labels"], stats,
                                      neg_mode="global_mean", seed=1, batch_size=64)
    assert Fg.shape == (len(va["store"]), N_CLASSES, DIM)
    assert (va["store"].dir / "oracle_global_mean.npy").exists()
