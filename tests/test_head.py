"""Synthetic-data tests for the gated attention head, its trainer and the localization analysis.

A tiny FeatureStore is written to a tmp dir (grid 4 -> 16 patches, dim 16, 300 train +
100 val images).  Every positive (image, class) pair plants a class signal in exactly
ONE patch, so mean pooling dilutes it 16x while attention can find it.  No real
dataset, no dependency on the evaluation harness modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from lip.features.store import FeatureStore, write_meta
from lip.models.attention_mil import (GatedAttentionHead, MultiScaleGatedAttentionHead,
                                      attention_entropy, build_head, mil_loss, n_params)
from lip.models.train_head import (attention_store, fit_head, labels_for_store, load_head,
                                   predict_store, save_head)
from lip.eval.localization import (localization_table, save_localization,
                                   summarise_localization)

G, D, C = 4, 16, 6          # grid, token dim, classes
P = G * G
BACKBONE, RES = "synth", 56  # 56 / 14 = grid 4
SIGNAL = 5.0


def _make_split(root, split, n, seed, id_offset):
    rng = np.random.default_rng(seed)
    d = root / BACKBONE / f"res{RES}" / split
    d.mkdir(parents=True)
    patch = np.lib.format.open_memmap(d / "patch.npy", mode="w+", dtype=np.float16, shape=(n, P, D))
    y = (rng.random((n, C)) < 0.3).astype(np.uint8)
    planted = np.full((n, C), -1, dtype=np.int64)
    for i in range(n):
        tok = rng.standard_normal((P, D)).astype(np.float32)
        for c in np.nonzero(y[i])[0]:
            p = int(rng.integers(P))
            planted[i, c] = p
            tok[p, c] += SIGNAL          # class c lives along token dimension c
        patch[i] = tok.astype(np.float16)
    patch.flush()
    del patch
    np.save(d / "cls.npy", rng.standard_normal((n, D)).astype(np.float16))
    ids = id_offset + np.arange(n, dtype=np.int64)
    np.save(d / "image_ids.npy", ids)
    write_meta(d, backbone=BACKBONE, res=RES, patch_size=14, grid=G, dim=D, n=n,
               resize="squash", mean=[0.0] * 3, std=[1.0] * 3)
    frac = np.where(y > 0, 1.0 / P, 0.0).astype(np.float32)
    labels = {"image_ids": ids, "split": np.array([split] * n), "y": y,
              "max_box_frac": frac, "max_seg_frac": frac,
              "cat_ids": np.arange(C), "cat_names": np.array([f"class{c}" for c in range(C)])}
    rows = []
    for i, c in zip(*np.nonzero(y)):
        p = planted[i, c]
        r, col = divmod(int(p), G)
        rows.append({"image_id": int(ids[i]), "cat_idx": int(c), "x0": col / G, "y0": r / G,
                     "x1": (col + 1) / G, "y1": (r + 1) / G, "box_frac": 1.0 / P, "seg_frac": 1.0 / P})
    boxes = pd.DataFrame(rows)
    return labels, boxes, planted


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("features")
    lab_tr, box_tr, pl_tr = _make_split(root, "train", 300, seed=0, id_offset=1000)
    lab_va, box_va, pl_va = _make_split(root, "val", 100, seed=1, id_offset=5000)
    st_tr = FeatureStore(BACKBONE, RES, "train", root=root)
    st_va = FeatureStore(BACKBONE, RES, "val", root=root)
    return dict(root=root, store_train=st_tr, labels_train=lab_tr, store_val=st_va,
                labels_val=lab_va, boxes_val=box_va, planted_val=pl_va)


@pytest.fixture(scope="module")
def trained(synth):
    model, hist = fit_head(synth["store_train"], synth["labels_train"], synth["store_val"],
                           synth["labels_val"], model_kind="attn", lam=0.0, epochs=30, lr=1e-2,
                           batch_size=32, hidden=32, device="cpu", seed=0, verbose=False,
                           patience=8, n_stat_images=100)
    return model, hist


# ------------------------------------------------------------------ forward
def test_forward_shapes_and_alpha_normalised():
    torch.manual_seed(0)
    m = GatedAttentionHead(dim=D, n_classes=C, hidden=8)
    z = torch.randn(5, P, D)
    logits, alpha = m(z)
    assert logits.shape == (5, C)
    assert alpha.shape == (5, C, P)
    assert torch.allclose(alpha.sum(-1), torch.ones(5, C), atol=1e-5)
    assert (alpha >= 0).all()


def test_param_count_formula():
    m = GatedAttentionHead(dim=768, n_classes=80, hidden=256)
    expected = 2 * (768 * 256 + 256) + 256 * 80 + 80 * 768 + 80
    assert n_params(m) == expected
    assert 4.5e5 < expected < 5.0e5


def test_entropy_bounds_and_limits():
    uni = torch.full((2, C, P), 1.0 / P)
    assert torch.allclose(attention_entropy(uni), torch.ones(2, C), atol=1e-6)
    hot = torch.zeros(2, C, P)
    hot[..., 3] = 1.0
    assert torch.allclose(attention_entropy(hot), torch.zeros(2, C), atol=1e-6)
    rnd = torch.softmax(torch.randn(4, C, P) * 3, dim=-1)
    e = attention_entropy(rnd)
    assert (e >= 0).all() and (e <= 1 + 1e-6).all()


def test_mil_loss_regularises_present_classes_only():
    torch.manual_seed(0)
    logits = torch.zeros(3, C)
    alpha = torch.softmax(torch.randn(3, C, P), dim=-1)
    y = torch.zeros(3, C)
    loss0, bce, ent0 = mil_loss(logits, alpha, y, lam=0.5)
    assert float(ent0) == 0.0 and torch.isclose(loss0, bce)   # no positives: no entropy term
    y[0, 1] = 1
    loss1, bce1, ent1 = mil_loss(logits, alpha, y, lam=0.5)
    assert torch.isclose(ent1, attention_entropy(alpha)[0, 1])
    assert torch.isclose(loss1, bce1 + 0.5 * ent1)


def test_multiscale_head_forward():
    torch.manual_seed(0)
    m = MultiScaleGatedAttentionHead(dim=D, n_classes=C, hidden=8, grid=G)
    z = torch.randn(3, P, D)
    logits, alpha = m(z)
    n_tok = P + (G // 2) ** 2 + (G // 4) ** 2
    assert logits.shape == (3, C) and alpha.shape == (3, C, n_tok)
    assert torch.allclose(alpha.sum(-1), torch.ones(3, C), atol=1e-5)
    fine = m.fine_attention(alpha)
    assert fine.shape == (3, C, P)
    assert torch.allclose(fine.sum(-1), torch.ones(3, C), atol=1e-5)
    assert build_head("attn_ms", dim=D, n_classes=C, hidden=8, grid=G).kind == "attn_ms"


# ----------------------------------------------------------------- training
def test_fit_head_learns_planted_signal(synth, trained):
    model, hist = trained
    assert hist["best_val_map"] > 0.8, hist
    assert len(hist["epochs"]) <= 30 and hist["best_epoch"] >= 1
    scores = predict_store(model, synth["store_val"], batch_size=64, device="cpu")
    assert scores.shape == (100, C) and scores.dtype == np.float32
    assert (scores >= 0).all() and (scores <= 1).all()


def test_attention_lands_on_planted_patch(synth, trained):
    model, _ = trained
    alpha = attention_store(model, synth["store_val"], batch_size=64, device="cpu")
    assert alpha.shape == (100, C, P) and alpha.dtype == np.float16
    assert np.allclose(alpha.astype(np.float32).sum(-1), 1.0, atol=2e-3)
    y = synth["labels_val"]["y"]
    pl = synth["planted_val"]
    ii, cc = np.nonzero(y)
    mass = alpha[ii, cc, pl[ii, cc]].astype(np.float32)
    chance = 1.0 / P
    assert mass.mean() > 8 * chance, mass.mean()       # wide margin over uniform
    assert mass.mean() > 0.5


def test_localization_table_and_summary(synth, trained, tmp_path):
    model, _ = trained
    alpha = attention_store(model, synth["store_val"], batch_size=64, device="cpu")
    df = localization_table(alpha, synth["boxes_val"], synth["labels_val"], grid=G)
    assert len(df) == int(synth["labels_val"]["y"].sum())
    for col in ["image_id", "cat_idx", "mass_in_box", "chance_mass", "pointing_hit", "size_bin"]:
        assert col in df.columns
    assert np.allclose(df["chance_mass"], 1.0 / P)
    assert (df["size_bin"] == 4).all()                   # 1/16 = 6.25% -> "3-10%"
    assert df["mass_in_box"].mean() > 0.5
    assert df["pointing_hit"].mean() > 0.8

    summ = summarise_localization(df, name="attn")
    assert set(summ) >= {"overall", "per_bin", "uniform", "bins", "n_pairs"}
    keys = {"mass_in_box", "chance_mass", "mass_ratio", "pointing_acc", "n"}
    assert set(summ["overall"]) == keys
    assert set(summ["per_bin"]) == set(summ["bins"]["names"]) and len(summ["per_bin"]) == 6
    assert set(summ["per_bin"]["3-10%"]) == keys and summ["per_bin"]["3-10%"]["n"] == len(df)
    assert summ["per_bin"]["<0.1%"]["n"] == 0
    assert summ["overall"]["mass_ratio"] > 8
    assert summ["uniform"]["overall"]["mass_ratio"] == 1.0

    # per-patch logits path: softmaxed inside, same table shape
    logits = np.random.default_rng(0).standard_normal(alpha.shape).astype(np.float32)
    df2 = localization_table(logits, synth["boxes_val"], synth["labels_val"], grid=G, is_logits=True)
    assert len(df2) == len(df) and (df2["mass_in_box"] <= 1.0 + 1e-6).all()

    out = save_localization(summ, tmp_path / "loc.json")
    assert out.exists() and out.stat().st_size > 0


def test_save_load_roundtrip(synth, trained, tmp_path):
    model, hist = trained
    path = save_head(model, tmp_path / "ckpt" / "attn.pt", history=hist)
    m2 = load_head(path)
    s1 = predict_store(model, synth["store_val"], device="cpu")
    s2 = predict_store(m2, synth["store_val"], device="cpu")
    assert np.allclose(s1, s2, atol=1e-6)
    assert torch.allclose(m2.in_mean, model.in_mean.cpu())


def test_labels_alignment_by_image_id(synth):
    st, lab = synth["store_val"], synth["labels_val"]
    perm = np.random.default_rng(3).permutation(len(st))
    shuffled = {"image_ids": lab["image_ids"][perm], "y": lab["y"][perm]}
    y = labels_for_store(st, shuffled)
    assert np.array_equal(y, lab["y"])
