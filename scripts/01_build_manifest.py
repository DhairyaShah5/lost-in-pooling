"""Step 1: build manifest.parquet, labels.npz, boxes.parquet from the downloaded COCO files."""
import argparse
from pathlib import Path

from lip.data.manifest import build

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/coco")
    a = p.parse_args()
    build(Path(a.root))
