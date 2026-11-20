"""Step 21: LUNA16 nodule slices (subsets on disk) in the COCO contract under data/luna16/."""
import argparse
from pathlib import Path

from lip.data.luna16 import build

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/luna16")
    p.add_argument("--subsets", default="subset0,subset1")
    p.add_argument("--neg-per-scan", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build(Path(a.root), tuple(s for s in a.subsets.split(",") if s), a.neg_per_scan, seed=a.seed)
