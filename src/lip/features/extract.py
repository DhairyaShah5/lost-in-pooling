"""Frozen DINOv2 feature extraction -> fp16 memmaps (docs/CONTRACTS.md, Feature store)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lip.features.store import write_meta

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 14


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_backbone(name: str = "dinov2_vitb14_reg", device=None):
    model = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True)
    model.eval().requires_grad_(False)
    return model.to(device or get_device())


class SquashDataset(Dataset):
    """Resize every image to res x res without cropping (aspect ratio is not preserved)."""

    def __init__(self, paths: list[str], res: int):
        self.paths, self.res = paths, res
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize((self.res, self.res), Image.BICUBIC)
        x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std


@torch.no_grad()
def extract_split(model, backbone: str, manifest: pd.DataFrame, split: str, res: int,
                  out_root: Path = Path("features"), batch_size: int = 64, workers: int = 6,
                  limit: int | None = None, device=None) -> Path:
    device = device or get_device()
    sub = manifest[manifest.split == split].reset_index(drop=True)
    if limit:
        sub = sub.iloc[:limit]
    n = len(sub)
    grid = res // PATCH
    P = grid * grid
    D = int(model.embed_dim)
    out = out_root / backbone / f"res{res}" / split
    out.mkdir(parents=True, exist_ok=True)
    cls_mm = np.lib.format.open_memmap(out / "cls.npy", mode="w+", dtype=np.float16, shape=(n, D))
    patch_mm = np.lib.format.open_memmap(out / "patch.npy", mode="w+", dtype=np.float16, shape=(n, P, D))

    dl = DataLoader(SquashDataset(sub["path"].tolist(), res), batch_size=batch_size,
                    num_workers=workers, shuffle=False, pin_memory=False)
    t0, done = time.time(), 0
    for x in dl:
        x = x.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type != "cpu")):
            f = model.forward_features(x)
        cls = f["x_norm_clstoken"].float().cpu().numpy().astype(np.float16)
        pt = f["x_norm_patchtokens"].float().cpu().numpy().astype(np.float16)
        b = cls.shape[0]
        cls_mm[done:done + b] = cls
        patch_mm[done:done + b] = pt
        done += b
        if done % (batch_size * 10) == 0 or done == n:
            el = time.time() - t0
            print(f"  {split}: {done}/{n}  {done / el:5.1f} img/s  elapsed {el:5.0f}s", flush=True)
    cls_mm.flush()
    patch_mm.flush()
    np.save(out / "image_ids.npy", sub["image_id"].to_numpy().astype(np.int64))
    write_meta(out, backbone=backbone, res=res, patch_size=PATCH, grid=grid, dim=D, n=n,
               resize="squash", mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD),
               img_per_s=round(n / (time.time() - t0), 2), tokens="x_norm_clstoken + x_norm_patchtokens",
               autocast="float16" if device.type != "cpu" else "none")
    return out
