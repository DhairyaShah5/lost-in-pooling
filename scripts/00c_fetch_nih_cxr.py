#!/usr/bin/env python
"""Selective download of NIH ChestX-ray14 PNGs from the Hugging Face mirror
alkzar90/NIH-Chest-X-ray-dataset, whose images are stored as 12 ZIP archives
(data/images/images_001.zip ... images_012.zip, ~42 GB total).

ZIP central directories are read with HTTP Range requests, and each wanted PNG
is then fetched with a single Range request covering its local header + data.
No full archive is ever downloaded.

Subcommands:
  index  --out INDEX_CSV                         build filename -> (zip, offset, size, crc) index
  plan   --index INDEX_CSV --data-dir DIR --n N --seed S --out PLAN_CSV
  fetch  --index INDEX_CSV --plan PLAN_CSV --out-dir IMG_DIR [--workers W] [--deadline EPOCH]
  verify --plan PLAN_CSV --out-dir IMG_DIR [--sample K]
  labels --data-dir DIR                          subset_plan.csv + Data_Entry labels + on-disk status -> subset_labels.csv

Originally data/nih_cxr/_scripts/hf_zip_fetch.py (2026-11-15); moved into the repo so the fetch is reproducible.
Provenance and counts: docs/NIH_CXR_SOURCES.md. Driver: scripts/00c_download_nih_cxr.sh.
"""
import argparse
import csv
import io
import os
import random
import struct
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

REPO = "alkzar90/NIH-Chest-X-ray-dataset"
ZIP_NAMES = [f"images_{i:03d}.zip" for i in range(1, 13)]
RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/main/data/images/{name}"
UA = "lost-in-pooling-data-fetch/0.1 (python-requests)"

_tls = threading.local()


def sess():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = UA
        a = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8)
        s.mount("https://", a)
        _tls.s = s
    return s


class UrlCache:
    """Resolve HF 'resolve' URLs to signed CDN URLs once; refresh on 403."""

    def __init__(self):
        self.lock = threading.Lock()
        self.urls = {}
        self.sizes = {}

    def get(self, name, force=False):
        with self.lock:
            if not force and name in self.urls:
                return self.urls[name], self.sizes[name]
            for attempt in range(6):
                try:
                    r = sess().head(RESOLVE_URL.format(repo=REPO, name=name),
                                    allow_redirects=False, timeout=30)
                    if r.status_code in (301, 302, 303, 307, 308):
                        loc = r.headers["Location"]
                        size = int(r.headers.get("x-linked-size") or r.headers.get("content-length") or 0)
                        self.urls[name] = loc
                        self.sizes[name] = size
                        return loc, size
                    if r.status_code == 200:
                        self.urls[name] = r.url
                        self.sizes[name] = int(r.headers["content-length"])
                        return r.url, self.sizes[name]
                    raise RuntimeError(f"resolve {name}: HTTP {r.status_code}")
                except Exception as e:  # noqa
                    if attempt == 5:
                        raise
                    time.sleep(1.5 * (attempt + 1))


URLS = UrlCache()


def fetch_range(name, start, end, retries=6):
    """Fetch inclusive byte range [start, end] of archive `name`."""
    last = None
    for attempt in range(retries):
        url, _ = URLS.get(name, force=(attempt > 0 and last == 403))
        try:
            r = sess().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 120))
            last = r.status_code
            if r.status_code == 206:
                if len(r.content) != end - start + 1:
                    raise IOError(f"short range: got {len(r.content)} want {end-start+1}")
                return r.content
            if r.status_code == 200:
                return r.content[start:end + 1]
            if r.status_code in (429, 500, 502, 503, 504, 403):
                time.sleep(min(30, 2 ** attempt))
                continue
            raise IOError(f"HTTP {r.status_code} for range {start}-{end} of {name}")
        except (requests.RequestException, IOError) as e:
            last = getattr(e, "errno", last)
            if attempt == retries - 1:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise IOError(f"range fetch failed for {name} {start}-{end} (last={last})")


class RemoteFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP Range requests, with a
    chunked cache. Only used to let zipfile parse the central directory."""

    CHUNK = 1 << 20

    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.pos = 0
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def _chunk(self, idx):
        if idx not in self.cache:
            start = idx * self.CHUNK
            end = min(self.size, start + self.CHUNK) - 1
            self.cache[idx] = fetch_range(self.name, start, end)
        return self.cache[idx]

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, self.CHUNK)
            c = self._chunk(idx)
            take = min(n, len(c) - off)
            out += c[off:off + take]
            self.pos += take
            n -= take
        return bytes(out)


def cmd_index(args):
    rows = []
    for name in ZIP_NAMES:
        url, size = URLS.get(name)
        rf = RemoteFile(name, size)
        t0 = time.time()
        zf = zipfile.ZipFile(rf)
        infos = zf.infolist()
        n_png = 0
        for zi in infos:
            if zi.is_dir():
                continue
            rows.append((os.path.basename(zi.filename), name, zi.filename, zi.header_offset,
                         zi.compress_size, zi.file_size, zi.compress_type, zi.CRC))
            n_png += 1
        print(f"{name}: size={size} entries={len(infos)} files={n_png} "
              f"cd_requests={len(rf.cache)} t={time.time()-t0:.1f}s", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["basename", "zip", "member", "header_offset", "compress_size",
                    "file_size", "compress_type", "crc"])
        w.writerows(rows)
    print(f"index: {len(rows)} files -> {args.out}", flush=True)


def load_index(path):
    idx = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            idx[r["basename"]] = r
    return idx


def cmd_plan(args):
    import pandas as pd
    idx = load_index(args.index)
    bbox = pd.read_csv(os.path.join(args.data_dir, "BBox_List_2017.csv"))
    de_path = os.path.join(args.data_dir, "Data_Entry_2017_v2020.csv")
    if not os.path.exists(de_path):
        de_path = os.path.join(args.data_dir, "Data_Entry_2017.csv")
    de = pd.read_csv(de_path)
    boxed = sorted(set(bbox["Image Index"]))
    missing_boxed = [b for b in boxed if b not in idx]
    print(f"boxed unique images: {len(boxed)}; missing from index: {len(missing_boxed)}", flush=True)
    all_imgs = [x for x in de["Image Index"].tolist() if x in idx]
    print(f"Data_Entry images present in index: {len(all_imgs)} / {len(de)}", flush=True)
    boxed_set = set(boxed)
    rest = [x for x in all_imgs if x not in boxed_set]
    rng = random.Random(args.seed)
    rng.shuffle(rest)
    rest = rest[: args.n]
    plan = [(b, 1, i) for i, b in enumerate(boxed)] + [(r, 0, len(boxed) + i) for i, r in enumerate(rest)]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image Index", "in_bbox", "rank", "zip"])
        for name, inb, rank in plan:
            w.writerow([name, inb, rank, idx[name]["zip"]])
    tot = sum(int(idx[n]["compress_size"]) for n, _, _ in plan)
    print(f"plan: {len(plan)} images ({len(boxed)} boxed + {len(rest)} random, seed={args.seed}), "
          f"{tot/1e9:.2f} GB -> {args.out}", flush=True)


def fetch_member(row):
    name = row["zip"]
    off = int(row["header_offset"])
    csize = int(row["compress_size"])
    usize = int(row["file_size"])
    ctype = int(row["compress_type"])
    crc = int(row["crc"])
    guess = 30 + len(row["member"].encode("utf-8")) + 64
    buf = fetch_range(name, off, off + guess + csize - 1)
    sig, _, _, comp, _, _, _, _, _, n, m = struct.unpack("<IHHHHHIIIHH", buf[:30])
    if sig != 0x04034B50:
        raise IOError(f"bad local header sig for {row['basename']}")
    ds = 30 + n + m
    need = ds + csize
    if need > len(buf):
        buf += fetch_range(name, off + len(buf), off + need - 1)
    data = buf[ds:need]
    if ctype == 8:
        data = zlib.decompress(data, -15)
    elif ctype != 0:
        raise IOError(f"unsupported compression {ctype}")
    if len(data) != usize or (zlib.crc32(data) & 0xFFFFFFFF) != crc:
        raise IOError(f"CRC/size mismatch for {row['basename']}")
    return data


def cmd_fetch(args):
    idx = load_index(args.index)
    plan = []
    with open(args.plan, newline="") as f:
        for r in csv.DictReader(f):
            plan.append(r["Image Index"])
    os.makedirs(args.out_dir, exist_ok=True)
    have = set(os.listdir(args.out_dir))
    todo = [p for p in plan if p not in have]
    print(f"plan={len(plan)} on_disk={len(plan)-len(todo)} todo={len(todo)} workers={args.workers}", flush=True)
    done = 0
    fail = 0
    nbytes = 0
    t0 = time.time()
    lock = threading.Lock()
    stop = threading.Event()

    def work(bn):
        if stop.is_set():
            return bn, None, "skipped"
        try:
            data = fetch_member(idx[bn])
            tmp = os.path.join(args.out_dir, "." + bn + ".part")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, os.path.join(args.out_dir, bn))
            return bn, len(data), None
        except Exception as e:  # noqa
            return bn, None, repr(e)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        it = iter(todo)
        # keep a bounded number in flight so the deadline check is responsive
        inflight = set()
        def submit_next():
            try:
                bn = next(it)
            except StopIteration:
                return False
            fu = ex.submit(work, bn)
            inflight.add(fu)
            return True
        for _ in range(args.workers * 2):
            if not submit_next():
                break
        while inflight:
            fu = next(as_completed(inflight))
            inflight.discard(fu)
            bn, n, err = fu.result()
            if err == "skipped":
                continue
            if err:
                fail += 1
                print(f"FAIL {bn}: {err}", flush=True)
            else:
                done += 1
                nbytes += n
            if (done + fail) % 200 == 0:
                el = time.time() - t0
                print(f"[{el:7.0f}s] done={done} fail={fail} {nbytes/1e6:.0f} MB "
                      f"{nbytes/1e6/el:.2f} MB/s", flush=True)
            if args.deadline and time.time() > args.deadline:
                if not stop.is_set():
                    print("deadline reached; draining in-flight requests", flush=True)
                    stop.set()
                continue
            submit_next()
    el = time.time() - t0
    print(f"FINISHED done={done} fail={fail} {nbytes/1e6:.0f} MB in {el:.0f}s "
          f"({nbytes/1e6/max(el,1):.2f} MB/s); on_disk={len(os.listdir(args.out_dir))}", flush=True)


def cmd_verify(args):
    from PIL import Image
    plan = []
    with open(args.plan, newline="") as f:
        for r in csv.DictReader(f):
            plan.append(r)
    on_disk = {x for x in os.listdir(args.out_dir) if x.endswith(".png")}
    boxed = [r["Image Index"] for r in plan if r["in_bbox"] == "1"]
    rnd = [r["Image Index"] for r in plan if r["in_bbox"] == "0"]
    print(f"png on disk: {len(on_disk)}")
    print(f"boxed images on disk: {sum(1 for b in boxed if b in on_disk)} / {len(boxed)}")
    print(f"random-subset images on disk: {sum(1 for b in rnd if b in on_disk)} / {len(rnd)}")
    rng = random.Random(0)
    sample = rng.sample(sorted(on_disk), min(args.sample, len(on_disk)))
    bad = 0
    sizes = {}
    modes = {}
    for bn in sample:
        try:
            im = Image.open(os.path.join(args.out_dir, bn))
            im.load()
            sizes[im.size] = sizes.get(im.size, 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
        except Exception as e:  # noqa
            bad += 1
            print(f"BAD {bn}: {e!r}")
    print(f"PIL check on {len(sample)} files: bad={bad} sizes={sizes} modes={modes}")


def cmd_labels(args):
    """Merge subset_plan.csv with the Data_Entry labels and on-disk status -> subset_labels.csv."""
    import pandas as pd
    D = args.data_dir
    plan = pd.read_csv(os.path.join(D, "subset_plan.csv"))
    de = pd.read_csv(os.path.join(D, "Data_Entry_2017_v2020.csv"))
    bbox = pd.read_csv(os.path.join(D, "BBox_List_2017.csv"))
    on_disk = set(os.listdir(os.path.join(D, "images")))
    test = set(open(os.path.join(D, "test_list.txt")).read().split())
    df = plan.merge(de, on="Image Index", how="left")
    df["on_disk"] = df["Image Index"].isin(on_disk).astype(int)
    df["official_split"] = df["Image Index"].map(lambda x: "test" if x in test else "train_val")
    df["n_bbox"] = df["Image Index"].map(bbox["Image Index"].value_counts()).fillna(0).astype(int)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df.to_csv(os.path.join(D, "subset_labels.csv"), index=False)
    print("rows:", len(df), "on_disk:", int(df.on_disk.sum()), "boxed:", int(df.in_bbox.sum()),
          "random:", int((df.in_bbox == 0).sum()))
    print("official split of subset:", df.official_split.value_counts().to_dict())
    print("boxed images sharing a Patient ID with random subset:",
          int(df[df.in_bbox == 1]["Patient ID"].isin(df[df.in_bbox == 0]["Patient ID"]).sum()))
    print("no-finding fraction in random subset:",
          round(float((df[df.in_bbox == 0]["Finding Labels"] == "No Finding").mean()), 3))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("index"); a.add_argument("--out", required=True)
    a = sp.add_parser("plan"); a.add_argument("--index", required=True); a.add_argument("--data-dir", required=True)
    a.add_argument("--n", type=int, default=15000); a.add_argument("--seed", type=int, default=42); a.add_argument("--out", required=True)
    a = sp.add_parser("fetch"); a.add_argument("--index", required=True); a.add_argument("--plan", required=True)
    a.add_argument("--out-dir", required=True); a.add_argument("--workers", type=int, default=24)
    a.add_argument("--deadline", type=float, default=0.0)
    a = sp.add_parser("verify"); a.add_argument("--plan", required=True); a.add_argument("--out-dir", required=True)
    a.add_argument("--sample", type=int, default=50)
    a = sp.add_parser("labels"); a.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    {"index": cmd_index, "plan": cmd_plan, "fetch": cmd_fetch, "verify": cmd_verify, "labels": cmd_labels}[args.cmd](args)


if __name__ == "__main__":
    main()
