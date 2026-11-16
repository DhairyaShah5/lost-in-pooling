"""Seeded selection of the train2017 subset. Shared by download + manifest so they agree."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_TRAIN_SUBSET = 10_000
N_TRAIN = 9_000  # first N_TRAIN of the subset -> split "train", rest -> "val"
SEED = 0


def select_train_subset(images: list[dict], n: int = N_TRAIN_SUBSET, seed: int = SEED) -> list[int]:
    """`images` is the COCO `images` list. Returns chosen image ids in permutation order."""
    ids = sorted(int(img["id"]) for img in images)
    perm = np.random.default_rng(seed).permutation(len(ids))[:n]
    return [ids[i] for i in perm]


def save_subset(ids: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": SEED, "n": len(ids), "n_train": N_TRAIN, "ids": ids}))


def load_subset(path: Path) -> dict:
    return json.loads(Path(path).read_text())
