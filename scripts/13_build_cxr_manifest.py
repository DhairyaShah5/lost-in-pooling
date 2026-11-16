"""Step 13: NIH ChestX-ray14 manifest/labels/boxes in the COCO contract (data/nih_cxr/)."""
import argparse
from pathlib import Path

from lip.data.nih_cxr import build

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/nih_cxr")
    p.add_argument("--val-frac", type=float, default=0.1)
    a = p.parse_args()
    build(Path(a.root), a.val_frac)
