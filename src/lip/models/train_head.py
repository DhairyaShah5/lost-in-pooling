"""Train / apply the gated attention head on memory-mapped patch tokens.

Streams random mini-batches from `FeatureStore.patch_batch` (never materialises the
full memmap), AdamW + cosine LR, early stopping on validation macro mAP.

Checkpoints go to `checkpoints/` (gitignored); attention maps to `results/attention/`.
"""
from __future__ import annotations

import copy
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from lip.models.attention_mil import build_head, mil_loss, n_params

CHECKPOINT_ROOT = Path("checkpoints")
ATTENTION_ROOT = Path("results/attention")


# ----------------------------------------------------------------------- helpers
def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None and device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def labels_for_store(store, labels) -> np.ndarray:
    """(n_store, C) uint8 label matrix aligned to `store.image_ids`.

    `labels` is either an aligned (n, C) array or the dict from `load_labels(split=...)`
    (aligned through `image_ids`, so the order need not match the store).
    """
    if isinstance(labels, np.ndarray):
        if len(labels) != len(store):
            raise ValueError(f"label rows {len(labels)} != store rows {len(store)}")
        return labels
    ids, y = labels["image_ids"], labels["y"]
    if len(ids) == len(store) and np.array_equal(ids, store.image_ids):
        return np.asarray(y)
    idx = {int(i): k for k, i in enumerate(ids)}
    rows = np.array([idx[int(i)] for i in store.image_ids], dtype=np.int64)
    return np.asarray(y)[rows]


def estimate_input_stats(store, n_images: int = 1000, seed: int = 0, chunk: int = 250):
    """Per-dimension token mean/std from a random image subsample (float64 accumulation)."""
    rng = np.random.default_rng(seed)
    n = min(n_images, len(store))
    rows = np.sort(rng.choice(len(store), size=n, replace=False))
    D = store.dim
    s = np.zeros(D, dtype=np.float64)
    ss = np.zeros(D, dtype=np.float64)
    count = 0
    for i in range(0, n, chunk):
        blk = store.patch_batch(rows[i:i + chunk]).astype(np.float64).reshape(-1, D)
        s += blk.sum(0)
        ss += (blk * blk).sum(0)
        count += blk.shape[0]
    mean = s / count
    var = np.maximum(ss / count - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32) + 1e-6


def macro_ap(y: np.ndarray, scores: np.ndarray) -> float:
    """Macro average precision over classes with >= 1 positive (sklearn AP per class)."""
    y = np.asarray(y)
    aps = [average_precision_score(y[:, c], scores[:, c])
           for c in range(y.shape[1]) if y[:, c].any()]
    return float(np.mean(aps)) if aps else float("nan")


def _prefetch(gen, depth: int = 2):
    """Run a generator in a background thread, keeping `depth` items ahead."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()

    def worker():
        try:
            for item in gen:
                q.put(item)
        except BaseException as e:  # propagate to the consumer
            q.put(e)
        finally:
            q.put(done)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _batches(store, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    """Yield (patch fp16 (B,P,D), labels uint8 (B,C)) over a random permutation."""
    perm = rng.permutation(len(store))
    for s in range(0, len(perm), batch_size):
        rows = perm[s:s + batch_size]
        yield store.patch_batch(rows), y[rows]


def _to_device(x: np.ndarray, device: torch.device) -> torch.Tensor:
    x = np.ascontiguousarray(x)
    if not x.flags.writeable:       # memmap slices are read-only; we only read them
        x = x.copy() if not x.flags.owndata else x
        try:
            x.flags.writeable = True
        except ValueError:
            x = x.copy()
    return torch.from_numpy(x).to(device, non_blocking=True)


# ------------------------------------------------------------------------- fit
def fit_head(store_train, labels_train, store_val, labels_val, model_kind: str = "attn",
             lam: float = 0.0, epochs: int = 20, lr: float = 3e-4, weight_decay: float = 1e-4,
             batch_size: int = 128, hidden: int = 256, device=None, seed: int = 0,
             verbose: bool = True, patience: int = 5, dropout: float = 0.1,
             n_stat_images: int = 1000):
    """Train an attention head from image-level labels. Returns (model, history).

    `history` = {"epochs": [ {epoch, loss, bce, entropy, val_map, lr, seconds}, ...],
                 "best_epoch", "best_val_map", "n_params", "config"}.
    The returned model carries the best validation weights (early stopping on val mAP;
    training stops after `patience` epochs without improvement).
    """
    device = resolve_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    y_tr = labels_for_store(store_train, labels_train).astype(np.uint8)
    y_va = labels_for_store(store_val, labels_val).astype(np.uint8) if store_val is not None else None
    n_classes = int(y_tr.shape[1])

    model = build_head(model_kind, dim=store_train.dim, n_classes=n_classes, hidden=hidden,
                       dropout=dropout, grid=store_train.grid)
    mean, std = estimate_input_stats(store_train, n_images=n_stat_images, seed=seed)
    model.set_input_stats(mean, std)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = int(np.ceil(len(store_train) / batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_epoch))

    config = {"model_kind": model_kind, "lam": lam, "epochs": epochs, "lr": lr,
              "weight_decay": weight_decay, "batch_size": batch_size, "hidden": hidden,
              "dropout": dropout, "seed": seed, "device": str(device), "patience": patience,
              "n_train": len(store_train), "n_val": len(store_val) if store_val is not None else 0}
    history = {"epochs": [], "best_epoch": -1, "best_val_map": -1.0,
               "n_params": n_params(model), "config": config}
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    if verbose:
        print(f"[fit_head] kind={model_kind} params={history['n_params']:,} device={device} "
              f"n_train={len(store_train)} steps/epoch={steps_per_epoch} lam={lam}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss = tot_bce = tot_ent = 0.0
        n_steps = 0
        for xb, yb in _prefetch(_batches(store_train, y_tr, batch_size, rng), depth=2):
            x = _to_device(xb, device)
            y = _to_device(yb, device)
            logits, alpha = model(x)
            loss, bce, ent = mil_loss(logits, alpha, y, lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += float(loss.detach())
            tot_bce += float(bce.detach())
            tot_ent += float(ent.detach())
            n_steps += 1

        val_map = float("nan")
        if store_val is not None:
            scores = predict_store(model, store_val, batch_size=max(batch_size, 256), device=device)
            val_map = macro_ap(y_va, scores)
        secs = time.time() - t0
        rec = {"epoch": epoch, "loss": tot_loss / max(n_steps, 1), "bce": tot_bce / max(n_steps, 1),
               "entropy": tot_ent / max(n_steps, 1), "val_map": val_map,
               "lr": float(opt.param_groups[0]["lr"]), "seconds": secs}
        history["epochs"].append(rec)
        if verbose:
            print(f"[fit_head] epoch {epoch:3d}/{epochs}  loss {rec['loss']:.4f}  "
                  f"val mAP {val_map:.4f}  ent {rec['entropy']:.3f}  {secs:.1f}s")

        improved = (store_val is None) or (np.isfinite(val_map) and val_map > history["best_val_map"])
        if improved:
            history["best_val_map"] = val_map if store_val is not None else float("nan")
            history["best_epoch"] = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"[fit_head] early stop at epoch {epoch} (best {history['best_epoch']})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


# --------------------------------------------------------------------- apply
@torch.no_grad()
def _iter_forward(model, store, batch_size: int, device=None):
    device = resolve_device(device)
    model = model.to(device).eval()
    for sl, blk in _prefetch(store.iter_patch_batches(batch_size), depth=2):
        logits, alpha = model(_to_device(blk, device))
        yield sl, logits, alpha


def predict_store(model, store, batch_size: int = 256, device=None) -> np.ndarray:
    """Sigmoid class scores for every image in the store: (n, C) float32, store order."""
    out = np.zeros((len(store), model.n_classes), dtype=np.float32)
    for sl, logits, _ in _iter_forward(model, store, batch_size, device):
        out[sl] = torch.sigmoid(logits).float().cpu().numpy()
    return out


def attention_store(model, store, batch_size: int = 256, device=None) -> np.ndarray:
    """Attention on the fine G x G grid for every image: (n, C, P) float16, store order."""
    P = store.n_patches
    out = np.zeros((len(store), model.n_classes, P), dtype=np.float16)
    for sl, _, alpha in _iter_forward(model, store, batch_size, device):
        out[sl] = model.fine_attention(alpha).half().cpu().numpy()
    return out


def attention_path(readout: str, backbone: str, res: int, root: Path | str = ATTENTION_ROOT) -> Path:
    return Path(root) / f"{readout}__{backbone}__res{res}.npy"


def save_attention(alpha: np.ndarray, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(alpha, dtype=np.float16))
    return path


# --------------------------------------------------------------- checkpoints
def checkpoint_path(readout: str, backbone: str, res: int, root: Path | str = CHECKPOINT_ROOT) -> Path:
    return Path(root) / f"{readout}__{backbone}__res{res}.pt"


def save_head(model, path: Path | str, history: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"config": model.config, "state_dict": state, "history": history}, path)
    return path


def load_head(path: Path | str, device=None):
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    model = build_head(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if device is not None:
        model.to(resolve_device(device))
    return model
