"""Tiled-teacher distillation over cached tokens (phase 7 of the plan).

Teacher, tiled : the head trained on the R px grid (G x G tokens) applied to each of the four quadrants of the
                 2R px grid (2G x 2G tokens) and max-pooled over quadrants. It needs no training and sees the
                 image at twice the effective resolution the student will see.
Teacher, hi    : a head trained on the 2R px tokens, applied to the 2R px tokens.
Student        : a head on the R px tokens trained on the same images with
                 BCE(labels) + beta * BCE(teacher probabilities)  (+ lam * attention entropy, as in `fit_head`).

Everything runs on cached tokens; the backbone is never called.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .attention_mil import attention_entropy, build_head, n_params
from .train_head import (_prefetch, _to_device, estimate_input_stats, labels_for_store, macro_ap,
                         predict_store, resolve_device)


@torch.no_grad()
def tiled_scores(head_lo, store_hi, tiles: int = 2, batch_size: int = 64, device=None,
                 reduce: str = "max") -> np.ndarray:
    """Apply a head trained on a g x g grid to the (tiles*g) x (tiles*g) grid of `store_hi`, one tile at a time.

    Returns teacher probabilities (N, C): max (or mean) over tiles of sigmoid(logits).
    """
    device = resolve_device(device)
    head_lo = head_lo.to(device).eval()
    G = store_hi.grid
    assert G % tiles == 0, f"grid {G} not divisible by tiles={tiles}"
    g = G // tiles
    out = []
    for _, block in store_hi.iter_patch_batches(batch_size):
        z = _to_device(block, device).float()                   # (B, G*G, D)
        B, _, D = z.shape
        z = z.view(B, tiles, g, tiles, g, D).permute(0, 1, 3, 2, 4, 5).reshape(B * tiles * tiles, g * g, D)
        logits, _ = head_lo(z)                                  # (B*T, C)
        probs = torch.sigmoid(logits).view(B, tiles * tiles, -1)
        probs = probs.max(dim=1).values if reduce == "max" else probs.mean(dim=1)
        out.append(probs.float().cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def head_probs(head, store, batch_size: int = 128, device=None) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-predict_store(head, store, batch_size=batch_size, device=device)))


def _batches_rows(store, rows: np.ndarray, y: np.ndarray, soft, batch_size: int, rng: np.random.Generator):
    """Yield (patch fp16 (B,P,D), labels (B,C), soft targets (B,C) or None) over a permutation of `rows`."""
    perm = rng.permutation(len(rows))
    for s in range(0, len(perm), batch_size):
        k = perm[s:s + batch_size]
        yield store.patch_batch(rows[k]), y[k], (None if soft is None else soft[k])


def fit_student(store_train, rows_train: np.ndarray, y_train: np.ndarray, soft_train, store_val, labels_val,
                model_kind: str = "attn", beta: float = 1.0, lam: float = 0.0, epochs: int = 20,
                lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 128, hidden: int = 256,
                device=None, seed: int = 0, verbose: bool = True, patience: int = 5, dropout: float = 0.1,
                n_stat_images: int = 1000):
    """Train a head on `rows_train` of `store_train` with hard labels `y_train` (K, C) and optional soft
    teacher probabilities `soft_train` (K, C). `beta` weights the soft term; beta=0 or soft=None is the
    plain head on the same rows. Returns (model, history) like `fit_head`."""
    device = resolve_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y_train = y_train.astype(np.uint8)
    n_classes = int(y_train.shape[1])
    if soft_train is not None:
        soft_train = soft_train.astype(np.float32)
    y_va = labels_for_store(store_val, labels_val).astype(np.uint8) if store_val is not None else None

    model = build_head(model_kind, dim=store_train.dim, n_classes=n_classes, hidden=hidden,
                       dropout=dropout, grid=store_train.grid)
    mean, std = estimate_input_stats(store_train, n_images=n_stat_images, seed=seed)
    model.set_input_stats(mean, std)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = int(np.ceil(len(rows_train) / batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_epoch))
    config = {"model_kind": model_kind, "beta": beta, "lam": lam, "epochs": epochs, "lr": lr,
              "weight_decay": weight_decay, "batch_size": batch_size, "hidden": hidden, "dropout": dropout,
              "seed": seed, "device": str(device), "patience": patience, "n_train": int(len(rows_train)),
              "n_val": len(store_val) if store_val is not None else 0,
              "teacher": "none" if soft_train is None else "soft"}
    history = {"epochs": [], "best_epoch": -1, "best_val_map": -1.0, "n_params": n_params(model), "config": config}
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    if verbose:
        print(f"[fit_student] kind={model_kind} beta={beta} n_train={len(rows_train)} steps/epoch={steps_per_epoch} device={device}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot = {"loss": 0.0, "bce": 0.0, "kd": 0.0, "entropy": 0.0}
        n_steps = 0
        for xb, yb, sb in _prefetch(_batches_rows(store_train, rows_train, y_train, soft_train, batch_size, rng), depth=2):
            x = _to_device(xb, device)
            y = _to_device(yb, device).float()
            logits, alpha = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y)
            kd = torch.zeros((), device=device)
            if sb is not None and beta > 0:
                kd = F.binary_cross_entropy_with_logits(logits, _to_device(sb, device))
            ent = attention_entropy(alpha).mean()
            loss = bce + beta * kd + lam * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot["loss"] += float(loss.detach()); tot["bce"] += float(bce.detach())
            tot["kd"] += float(kd.detach()); tot["entropy"] += float(ent.detach())
            n_steps += 1
        val_map = float("nan")
        if store_val is not None:
            val_map = macro_ap(y_va, predict_store(model, store_val, batch_size=max(batch_size, 256), device=device))
        rec = {"epoch": epoch, **{k: v / max(n_steps, 1) for k, v in tot.items()}, "val_map": val_map,
               "lr": float(opt.param_groups[0]["lr"]), "seconds": time.time() - t0}
        history["epochs"].append(rec)
        if verbose:
            print(f"[fit_student] epoch {epoch:3d}/{epochs}  loss {rec['loss']:.4f}  bce {rec['bce']:.4f}  "
                  f"kd {rec['kd']:.4f}  val mAP {val_map:.4f}  {rec['seconds']:.1f}s")
        improved = (store_val is None) or (np.isfinite(val_map) and val_map > history["best_val_map"])
        if improved:
            history["best_val_map"] = val_map if store_val is not None else float("nan")
            history["best_epoch"] = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"[fit_student] early stop at epoch {epoch} (best {history['best_epoch']})")
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history
