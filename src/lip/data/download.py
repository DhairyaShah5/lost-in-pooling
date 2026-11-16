"""COCO 2017 downloader: annotations, full val2017, and a seeded subset of train2017."""
from __future__ import annotations

import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from lip.data.subset import save_subset, select_train_subset

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
IMG_URL = "http://images.cocodataset.org/train2017/{}"


def stream_download(url: str, dst: Path, chunk: int = 1 << 20) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    tmp = dst.with_name(dst.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            for c in r.iter_content(chunk):
                f.write(c)
                bar.update(len(c))
    tmp.rename(dst)
    return dst


def unzip_once(zip_path: Path, dest: Path, marker: Path) -> None:
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def _fetch_one(name: str, dst_dir: Path, retries: int = 4):
    dst = dst_dir / name
    if dst.exists() and dst.stat().st_size > 0:
        return name, True
    for k in range(retries):
        try:
            r = requests.get(IMG_URL.format(name), timeout=30)
            if r.status_code == 200 and r.content:
                tmp = dst.with_name(name + ".part")
                tmp.write_bytes(r.content)
                tmp.rename(dst)
                return name, True
        except Exception:
            pass
        time.sleep(1.5 * (k + 1))
    return name, False


def fetch_many(names: list[str], dst_dir: Path, workers: int = 32) -> list[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_fetch_one, n, dst_dir) for n in names]
        for f in tqdm(as_completed(futs), total=len(futs), desc="train2017 subset"):
            name, ok = f.result()
            if not ok:
                failed.append(name)
    return failed


def download_all(root: Path, n_train: int, workers: int) -> None:
    root = Path(root)
    t0 = time.time()
    ann_zip = stream_download(ANN_URL, root / "annotations_trainval2017.zip")
    unzip_once(ann_zip, root, root / "annotations" / "instances_val2017.json")
    print(f"[{time.time()-t0:6.0f}s] annotations ready")

    val_zip = stream_download(VAL_URL, root / "val2017.zip")
    unzip_once(val_zip, root / "images", root / "images" / "val2017" / "000000000139.jpg")
    print(f"[{time.time()-t0:6.0f}s] val2017 ready")

    with open(root / "annotations" / "instances_train2017.json") as f:
        images = json.load(f)["images"]
    ids = select_train_subset(images, n=n_train)
    save_subset(ids, root / "train_subset_ids.json")
    id2name = {int(img["id"]): img["file_name"] for img in images}
    names = [id2name[i] for i in ids]
    del images
    failed = fetch_many(names, root / "images" / "train2017", workers=workers)
    print(f"[{time.time()-t0:6.0f}s] train subset ready, failed={len(failed)}")
    if failed:
        (root / "train_subset_failed.json").write_text(json.dumps(failed))
