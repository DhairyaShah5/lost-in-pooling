"""Max-MIL read-out: a per-patch linear scorer, image logit = max over patches.

``MaxMIL`` applies ``nn.Linear(d, C)`` to every (standardised) patch token and
takes the max (or top-k mean, k=1 by default) over the P patches per class. It is
trained on image-level labels only, streaming patch tokens from the fp16 memmap
in random mini-batches (rows pre-shuffled per epoch, sorted within a batch for IO).
Token standardisation uses mean/std estimated once from a 1000-image subsample.

Memory: one batch of 128 x 256 x 768 fp32 is ~100 MB, so this fits on 16 GB.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lip.features.store import FeatureStore
from lip.readouts.probe import auto_device, macro_ap


def _as_y(labels) -> np.ndarray:
    """Accept a ``load_labels`` dict or a (n, C) array."""
    return np.asarray(labels["y"] if isinstance(labels, dict) else labels)


def estimate_token_stats(store: FeatureStore, n_images: int = 1000, seed: int = 0,
                         chunk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean/std of patch tokens over a seeded subsample of images."""
    rng = np.random.default_rng(seed)
    n = len(store)
    rows = np.sort(rng.choice(n, size=min(n_images, n), replace=False))
    d = int(store.patch.shape[-1])
    s = np.zeros(d, dtype=np.float64)
    ss = np.zeros(d, dtype=np.float64)
    cnt = 0
    for i in range(0, len(rows), chunk):
        x = store.patch_batch(rows[i:i + chunk]).astype(np.float32).reshape(-1, d)
        s += x.sum(axis=0, dtype=np.float64)
        ss += np.square(x, dtype=np.float64).sum(axis=0)
        cnt += x.shape[0]
    mu = s / cnt
    var = np.maximum(ss / cnt - mu * mu, 0.0)
    sd = np.sqrt(var)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


class MaxMIL(nn.Module):
    """Per-patch linear scorer with max (top-k mean) pooling over patches."""

    def __init__(self, dim: int, n_classes: int = 80, topk: int = 1):
        super().__init__()
        self.scorer = nn.Linear(dim, n_classes)
        self.topk = int(topk)
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.history_: dict = {}

    @property
    def device(self) -> torch.device:
        return self.scorer.weight.device

    def set_standardisation(self, mu: np.ndarray, sd: np.ndarray) -> None:
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
        self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32))

    def patch_logits(self, x: torch.Tensor) -> torch.Tensor:
        """(b, P, d) raw tokens -> (b, P, C) per-patch logits."""
        return self.scorer((x - self.mu) / self.sd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(b, P, d) -> (b, C) image logits = top-k mean over patches (k=1: max)."""
        pl = self.patch_logits(x)
        k = min(self.topk, pl.shape[1])
        if k <= 1:
            return pl.max(dim=1).values
        return pl.topk(k, dim=1).values.mean(dim=1)

    def _to_device(self, block: np.ndarray) -> torch.Tensor:
        """fp16 block -> float32 tensor on the model device (copy if it is a read-only memmap view)."""
        block = np.ascontiguousarray(block)
        if not block.flags.writeable:
            block = block.copy()
        return torch.from_numpy(block).to(self.device).float()

    @torch.no_grad()
    def predict_store(self, store: FeatureStore, batch_size: int = 128) -> np.ndarray:
        """Sigmoid image scores for every image of ``store``: (n, C) float32."""
        self.eval()
        out = np.empty((len(store), self.scorer.out_features), dtype=np.float32)
        for sl, block in store.iter_patch_batches(batch_size):
            out[sl] = torch.sigmoid(self(self._to_device(block))).cpu().numpy()
        return out

    @torch.no_grad()
    def patch_scores(self, store: FeatureStore, rows) -> np.ndarray:
        """Per-patch logits for the given rows: (b, P, C) float32 (for localisation analysis)."""
        self.eval()
        x = self._to_device(store.patch_batch(rows))
        return self.patch_logits(x).cpu().numpy()


def fit_maxmil(store_train: FeatureStore, labels_train, store_val: FeatureStore, labels_val,
               epochs: int = 15, lr: float = 1e-3, weight_decay: float = 1e-4,
               batch_size: int = 128, topk: int = 1, patience: int = 3, n_stats: int = 1000,
               device=None, seed: int = 0, verbose: bool = False) -> tuple[MaxMIL, dict]:
    """Train a ``MaxMIL`` head on memmapped tokens; early-stop on val macro-AP.

    Returns ``(model, history)`` with the best-val weights loaded; ``history`` has
    ``epochs`` (list of per-epoch records), ``best_epoch``, ``best_val_map``.
    """
    Y_tr = _as_y(labels_train).astype(np.float32)
    Y_va = _as_y(labels_val)
    n, d = len(store_train), int(store_train.patch.shape[-1])
    n_classes = Y_tr.shape[1]
    assert Y_tr.shape[0] == n, "labels_train must follow store_train row order"

    dev = auto_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = MaxMIL(d, n_classes, topk=topk).to(dev)
    mu, sd = estimate_token_stats(store_train, n_images=n_stats, seed=seed)
    model.set_standardisation(mu, sd)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    records = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, batch_size):
            rows = np.sort(perm[s:s + batch_size])  # sorted within batch for IO
            xb = model._to_device(store_train.patch_batch(rows))
            yb = torch.from_numpy(Y_tr[rows]).to(dev)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(rows)
        sched.step()

        vmap = macro_ap(Y_va, model.predict_store(store_val, batch_size))
        vcmp = -1.0 if np.isnan(vmap) else vmap
        rec = dict(epoch=ep, train_loss=tot / n, val_map=vmap, lr=sched.get_last_lr()[0],
                   sec=time.time() - t0)
        records.append(rec)
        if verbose:
            print(f"[maxmil] epoch {ep:3d}/{epochs} loss {rec['train_loss']:.4f} "
                  f"val mAP {vmap:.4f} ({rec['sec']:.1f}s)")
        if vcmp > best:
            best, best_epoch, bad = vcmp, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"[maxmil] early stop at epoch {ep} (best {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history = dict(epochs=records, best_epoch=best_epoch,
                   best_val_map=best if best >= 0 else float("nan"),
                   config=dict(epochs=epochs, lr=lr, weight_decay=weight_decay,
                               batch_size=batch_size, topk=topk, patience=patience,
                               n_stats=n_stats, seed=seed))
    model.history_ = history
    return model, history


def predict_store(model: MaxMIL, store: FeatureStore, batch_size: int = 128) -> np.ndarray:
    """Functional alias of ``MaxMIL.predict_store``."""
    return model.predict_store(store, batch_size)


def patch_scores(model: MaxMIL, store: FeatureStore, rows) -> np.ndarray:
    """Functional alias of ``MaxMIL.patch_scores``: (b, P, C) per-patch logits."""
    return model.patch_scores(store, rows)
