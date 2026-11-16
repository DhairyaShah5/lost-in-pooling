"""Joint multi-label linear probe: one ``nn.Linear(d, C)`` trained with BCE-with-logits.

Inputs are standardised with mean/std fitted on the training set (stored on the
probe so ``predict`` applies the same transform). AdamW + cosine LR; the epoch with
the best validation macro-AP is kept. Data is small (at most 9000 x 1536) so the
whole design matrix lives on the device.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def auto_device(device: str | torch.device | None = None) -> torch.device:
    """``None``/``"auto"`` -> MPS when available, else CPU."""
    if device is None or device == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device)


def macro_ap(Y: np.ndarray, S: np.ndarray) -> float:
    """Macro average precision (sklearn) over classes with at least one positive.

    Classes without positives are skipped rather than counted as 0, so the number is
    comparable between splits with different class coverage. NaN if no class has a positive.
    """
    Y = np.asarray(Y)
    S = np.asarray(S, dtype=np.float64)
    keep = Y.sum(axis=0) > 0
    if not keep.any():
        return float("nan")
    return float(average_precision_score(Y[:, keep], S[:, keep], average="macro"))


class LinearProbe:
    """Standardise -> ``nn.Linear(d, C)`` -> sigmoid, chosen at the best-val-mAP epoch."""

    def __init__(self):
        self.model: nn.Linear | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.device: torch.device = torch.device("cpu")
        self.history_: list[dict] = []
        self.best_epoch_: int = 0
        self.best_val_map_: float = float("nan")
        self.config_: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X_train, Y_train, X_val, Y_val, epochs: int = 40, lr: float = 1e-3,
            weight_decay: float = 1e-4, batch_size: int = 512, device=None, seed: int = 0,
            verbose: bool = False) -> "LinearProbe":
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        Y_train = np.asarray(Y_train, dtype=np.float32)
        Y_val_np = np.asarray(Y_val)
        n, d = X_train.shape
        n_classes = Y_train.shape[1]
        self.config_ = dict(epochs=epochs, lr=lr, weight_decay=weight_decay,
                            batch_size=batch_size, seed=seed, d=d, n_train=n)

        # standardisation fitted on train only
        self.mean_ = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        self.std_ = np.where(std < 1e-6, 1.0, std).astype(np.float32)

        self.device = dev = auto_device(device)
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)

        Xt = torch.from_numpy((X_train - self.mean_) / self.std_).to(dev)
        Yt = torch.from_numpy(Y_train).to(dev)
        Xv = torch.from_numpy((X_val - self.mean_) / self.std_).to(dev)

        model = nn.Linear(d, n_classes).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

        best, best_state, best_epoch = -1.0, None, 0
        self.history_ = []
        for ep in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            perm = torch.randperm(n, generator=gen).to(dev)
            tot = torch.zeros((), device=dev)  # accumulate on device: one host sync per epoch
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                logits = model(Xt[idx])
                loss = F.binary_cross_entropy_with_logits(logits, Yt[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.detach() * len(idx)
            sched.step()
            tot = float(tot)

            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xv)).cpu().numpy()
            vmap = macro_ap(Y_val_np, pv)
            vcmp = -1.0 if np.isnan(vmap) else vmap
            if vcmp > best:
                best, best_epoch = vcmp, ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            rec = dict(epoch=ep, train_loss=tot / n, val_map=vmap, lr=sched.get_last_lr()[0],
                       sec=time.time() - t0)
            self.history_.append(rec)
            if verbose:
                print(f"[probe] epoch {ep:3d}/{epochs} loss {rec['train_loss']:.4f} "
                      f"val mAP {vmap:.4f} lr {rec['lr']:.2e} ({rec['sec']:.1f}s)")

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        self.best_epoch_ = best_epoch
        self.best_val_map_ = best if best >= 0 else float("nan")
        return self

    # -------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, X, batch_size: int = 4096) -> np.ndarray:
        """Sigmoid probabilities, shape (n, C), float32."""
        if self.model is None:
            raise RuntimeError("call fit() first")
        X = (np.asarray(X, dtype=np.float32) - self.mean_) / self.std_
        self.model.eval()
        out = np.empty((X.shape[0], self.model.out_features), dtype=np.float32)
        for s in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[s:s + batch_size]).to(self.device)
            out[s:s + batch_size] = torch.sigmoid(self.model(xb)).cpu().numpy()
        return out

    def summary(self) -> str:
        c = self.config_
        return (f"epochs={c.get('epochs')} lr={c.get('lr')} wd={c.get('weight_decay')} "
                f"bs={c.get('batch_size')} best_epoch={self.best_epoch_} "
                f"val_map={self.best_val_map_:.4f} seed={c.get('seed')}")
