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

from lip.features.backbones import IMAGENET_MEAN, IMAGENET_STD, Backbone, get_device, load_backbone  # noqa: F401
from lip.features.store import write_meta


class SquashDataset(Dataset):
    """Resize every image to res x res without cropping (aspect ratio is not preserved)."""

    def __init__(self, paths: list[str], res: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.paths, self.res = paths, res
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize((self.res, self.res), Image.BICUBIC)
        x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std


@torch.no_grad()
def extract_split(bb: Backbone, backbone: str, manifest: pd.DataFrame, split: str, res: int,
                  out_root: Path = Path("features"), batch_size: int = 64, workers: int = 6,
                  limit: int | None = None, device=None) -> Path:
    device = device or get_device()
    sub = manifest[manifest.split == split].reset_index(drop=True)
    if limit:
        sub = sub.iloc[:limit]
    n = len(sub)
    grid = res // bb.patch_size
    P = grid * grid
    D = int(bb.embed_dim)
    out = out_root / backbone / f"res{res}" / split
    out.mkdir(parents=True, exist_ok=True)
    cls_mm = np.lib.format.open_memmap(out / "cls.npy", mode="w+", dtype=np.float16, shape=(n, D))
    patch_mm = np.lib.format.open_memmap(out / "patch.npy", mode="w+", dtype=np.float16, shape=(n, P, D))

    dl = DataLoader(SquashDataset(sub["path"].tolist(), res, bb.mean, bb.std), batch_size=batch_size,
                    num_workers=workers, shuffle=False, pin_memory=False)
    t0, done = time.time(), 0
    for x in dl:
        x = x.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type != "cpu")):
            c, pt = bb.tokens(x)
        assert pt.shape[1] == P, f"expected {P} patch tokens, got {pt.shape[1]}"
        cls = c.float().cpu().numpy().astype(np.float16)
        pt = pt.float().cpu().numpy().astype(np.float16)
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
    write_meta(out, backbone=backbone, res=res, patch_size=int(bb.patch_size), grid=grid, dim=D, n=n,
               resize="squash", mean=list(bb.mean), std=list(bb.std),
               img_per_s=round(n / (time.time() - t0), 2), tokens=f"global={bb.global_token}; patches=final tokens",
               autocast="float16" if device.type != "cpu" else "none")
    return out
