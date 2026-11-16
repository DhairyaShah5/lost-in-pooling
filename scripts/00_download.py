"""Step 0: download COCO 2017 annotations, val2017, and a seeded 10k train2017 subset."""
import argparse
from pathlib import Path

from lip.data.download import download_all
from lip.data.subset import N_TRAIN_SUBSET

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/coco")
    p.add_argument("--n-train", type=int, default=N_TRAIN_SUBSET)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    download_all(Path(a.root), a.n_train, a.workers)
