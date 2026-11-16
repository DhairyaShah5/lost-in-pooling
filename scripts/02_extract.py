"""Step 2: extract frozen DINOv2 features for the requested splits into fp16 memmaps."""
import argparse
import time
from pathlib import Path

import pandas as pd

from lip.features.extract import extract_split, get_device, load_backbone

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--splits", default="test,val,train")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="benchmark: only first N images per split")
    p.add_argument("--manifest", default="data/coco/manifest.parquet")
    a = p.parse_args()

    dev = get_device()
    t0 = time.time()
    model = load_backbone(a.backbone, dev)
    print(f"loaded {a.backbone} on {dev} in {time.time()-t0:.0f}s; embed_dim={model.embed_dim} patch={model.patch_size}")
    manifest = pd.read_parquet(a.manifest)
    for split in a.splits.split(","):
        t1 = time.time()
        out = extract_split(model, a.backbone, manifest, split, a.res, Path("features"),
                            a.batch_size, a.workers, a.limit, dev)
        print(f"{split} -> {out}  ({time.time()-t1:.0f}s)")
