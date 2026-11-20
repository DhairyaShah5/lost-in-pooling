"""Step 18: ChestX-Det10 manifest/labels/boxes in the COCO contract (data/chestx_det10/), with NIH identities
recovered by pixel hash so folds and transfer evaluations stay patient-disjoint."""
import argparse
from pathlib import Path

from lip.data.chestx_det10 import build

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/chestx_det10")
    p.add_argument("--nih-root", default="data/nih_cxr")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    build(Path(a.root), Path(a.nih_root), a.workers)
