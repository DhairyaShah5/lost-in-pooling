# NIH ChestX-ray14 subset: sources and provenance

Dataset: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on
Weakly-Supervised Classification and Localization of Common Thorax Diseases", CVPR 2017.
Official home: https://nihcc.app.box.com/v/ChestXray-NIHCC

Acquired on 2026-11-15 (PST) from the Hugging Face Hub mirror
`alkzar90/NIH-Chest-X-ray-dataset` (public, no auth). Kaggle and the NIH Box share
were not used.

## Label / metadata files (downloaded whole)

| Local file | Source URL | Bytes | Rows (excl. header) |
|---|---|---|---|
| `BBox_List_2017.csv` | https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/BBox_List_2017.csv | 92,416 | 984 (880 unique images) |
| `Data_Entry_2017_v2020.csv` | https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/Data_Entry_2017_v2020.csv | 9,003,499 | 112,120 |
| `Data_Entry_2017.csv` | copy of `Data_Entry_2017_v2020.csv` (name kept for code compatibility) | 9,003,499 | 112,120 |
| `train_val_list.txt` | https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/train_val_list.txt | 1,470,907 | 86,524 |
| `test_list.txt` | https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/test_list.txt | 435,131 | 25,596 |

Note: the mirror hosts the NIH-revised `Data_Entry_2017_v2020.csv` (the 2020 re-release of the
labels file from the official Box share; same images/labels, corrected patient-age values).
Columns: Image Index, Finding Labels, Follow-up #, Patient ID, Patient Age, Patient Gender,
View Position, OriginalImage[Width,Height], OriginalImagePixelSpacing[x,y].

`BBox_List_2017.csv` columns: Image Index, Finding Label, Bbox [x,y,w,h] (4 numeric columns,
pixel coordinates in the 1024x1024 image), plus 3 empty trailing columns.

## Images (selectively extracted, no archive downloaded in full)

The mirror stores the 112,120 PNGs inside 12 ZIP archives at
`https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images/images_0NN.zip`
(NN = 01..12; sizes 2.0 to 4.2 GB each, 46.1 GB total; members are deflate-compressed
`images/<Image Index>.png`, 1024x1024). The HF CDN supports HTTP Range requests, so
`scripts/00c_fetch_nih_cxr.py` (orchestrated by `scripts/00c_download_nih_cxr.sh`):

1. `index`: reads each archive's central directory by Range request
   (`data/nih_cxr/_index/hf_zip_index.csv`, 147,123 members incl. `__MACOSX/._*` junk; 112,120 real PNGs).
2. `plan`: selects all 880 boxed images, then a seeded random permutation
   (`random.Random(42).shuffle`) of the remaining 111,240 images, truncated to N
   (`data/nih_cxr/subset_plan.csv`, committed as `docs/data/nih_cxr_subset_plan.csv`; `rank` is the download order, so any prefix is a uniform random sample).
3. `fetch`: one Range request per PNG (local header + deflate stream), inflates, checks
   CRC32 against the central directory, writes `images/<Image Index>.png`.
4. `verify`: opens a random sample with PIL and reports sizes/modes.

Throughput observed: ~16-18 MB/s with 24 workers, 0 failures.

Images on disk are byte-identical to the archive members (CRC-verified). A small number of
NIH images are stored as RGBA/RGB rather than 8-bit grayscale (known dataset quirk);
convert with `Image.open(p).convert("L")`.

## Counts on disk (2026-11-15 fetch)

See the bottom of this file (filled in after the fetch finished).

### Final counts (2026-11-15 17:42 PST)

- `images/`: 40,880 PNGs, 15 GB (du -sh), all CRC32-verified against the archive central
  directories; PNG magic verified on all tranche-1 files; PIL opened 600+ sampled files,
  all 1024x1024 (~0.75% are RGBA instead of L).
- Boxed images: 880 / 880 unique images on disk, covering all 984 rows of `BBox_List_2017.csv`.
- Random subset: 40,000 / 40,000 images (seed 42; `subset_plan.csv` rank 880..40879 is the
  seeded order, so e.g. ranks 880..15879 are the first 15,000 random images).
- `subset_labels.csv`: 40,880 rows = `subset_plan.csv` joined with `Data_Entry_2017_v2020.csv`,
  plus `on_disk`, `official_split` (train_val / test per the NIH lists: 31,091 / 9,789),
  and `n_bbox`. 758 of the 880 boxed images share a Patient ID with at least one random-subset
  image, so split by `Patient ID` if boxed images are held out.
- Fetch logs: `_logs/fetch_tranche1_15880.log` (15,880 files, 6.39 GB, 347 s, 18.4 MB/s),
  `_logs/fetch_tranche2_40880.log` (25,000 files, 10.05 GB, 498 s, 20.2 MB/s). 0 failures.
- Total bytes pulled from the HF CDN: ~16.5 GB (vs 46.1 GB for the full archives).
