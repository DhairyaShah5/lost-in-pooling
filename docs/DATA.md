# Data sources and provenance

All raw data lives under `data/` (gitignored); every derived file is rebuilt by the scripts named below.

| Arm | Source | What we use | Builder |
|---|---|---|---|
| MS-COCO 2017 | cocodataset.org (val2017 + a seeded 10k subset of train2017; the chosen ids are also committed as `docs/data/coco_train_subset_ids.json`) | 5,000 test images, 9,000/1,000 train/val, instance boxes and masks for 80 classes | `scripts/00_download.py`, `01_build_manifest.py` |
| NIH ChestX-ray14 | Hugging Face mirror `alkzar90/NIH-Chest-X-ray-dataset` (zip range-reads, 40,880 images; exact list committed as `docs/data/nih_cxr_subset_plan.csv`, provenance in `docs/NIH_CXR_SOURCES.md`) | 880 boxed test images (984 radiologist boxes), 20,721 / 2,238 patient-disjoint train/val images with 14 image-level findings | `scripts/00c_download_nih_cxr.sh`, `13_build_cxr_manifest.py` |
| ChestX-Det10 | github.com/Deepwise-AILab/ChestX-Det10-Dataset (annotations) and resource.deepwise.com (images) | 3,543 NIH images re-annotated by three radiologists with 8,339 boxes for 10 findings; NIH identities recovered by a pixel hash (1,357 matched) so folds and transfer evaluations stay patient-disjoint | `scripts/00b_download_extras.sh`, `18_build_det10_manifest.py` |
| LUNA16 | zenodo.org/records/3723295 (subsets 0 and 1, annotations.csv, lung masks) | 2D axial lung-window slices: every slice through an annotated nodule (with its box) and six nodule-free lung slices per scan; scan-disjoint folds and a scan-level train/val/test split for the head | `scripts/00b_download_extras.sh`, `21_build_luna16.py` |

Licences: COCO (CC BY 4.0), NIH ChestX-ray14 (NIH Clinical Center, free for research), ChestX-Det10 (research use, cite Liu et al. 2020), LUNA16 (CC BY 4.0 via LIDC-IDRI).

## Expected layout under `data/`

Produced by the download and build steps in the README; nothing here is committed.

```
data/coco/          annotations/ (6 COCO json), images/ (15,000 jpg: val2017 + the 10k train2017 subset),
                    val2017.zip, annotations_trainval2017.zip, train_subset_ids.json,
                    manifest.parquet, labels.npz, boxes.parquet
data/nih_cxr/       BBox_List_2017.csv, Data_Entry_2017_v2020.csv (+ a copy named Data_Entry_2017.csv),
                    train_val_list.txt, test_list.txt, images/ (40,880 png, 1024x1024),
                    subset_plan.csv, subset_labels.csv, _index/hf_zip_index.csv,
                    manifest.parquet, labels.npz, boxes.parquet
data/chestx_det10/  raw/{train.json, test.json, README.md, train_data.zip, test_data.zip},
                    images_raw/ (3,543 png), _hashes.json (pixel hashes, cached),
                    manifest.parquet, labels.npz, boxes.parquet
data/luna16/        raw/{annotations.csv, seg-lungs-LUNA16.zip, subset0.zip, subset1.zip} and the folders the
                    builder unzips them into, images/ (2,507 png slices),
                    manifest.parquet, labels.npz, boxes.parquet (scan-disjoint CV arm),
                    manifest_head.parquet, labels_head.npz (scan-level train/val/test split for the head)
```

## Disk and time budget (measured 2026-11-20, Apple M4 16 GB, before the caches were cleared)

| Path | Size | Contents |
|---|---|---|
| `data/` | 57 GB | LUNA16 35 GB (12.5 GB of zips plus the extracted scans), NIH 15 GB, COCO 4 GB, ChestX-Det10 3 GB |
| `features/` | 78 GB | COCO tokens: DINOv2 at 224/336/448 px (40 GB), DINO ViT-B/16 at 224/448 (22 GB), CLIP 8 GB, SigLIP 8 GB |
| `features_cxr/` | 88 GB | NIH tokens, 44 GB per backbone; the full 20,721-image 448 px training cache alone is 30 GB |
| `features_det10/` | 13 GB | ChestX-Det10 tokens, both backbones at 224 and 448 px |
| `features_luna/`, `features_luna_head/` | 9 GB | LUNA16 tokens for the CV arm and the head arm |
| `results/attention/`, `results_cxr/attention/` | 6.5 GB | per-image attention maps from the localisation audits |

About 255 GB in all (272 GB in the decimal units Finder shows). Each `patch.npy` is an fp16 memmap of
images x tokens x 768; a 448 px grid has four times the tokens of a 224 px grid, so the 448 px caches dominate.
Everything in the table is gitignored and rebuilt by `scripts/02_extract.py` and the download scripts; the report,
slides, poster and `scripts/demo.py` (which loads `checkpoints/`) need none of it. The committed repository is 240 MB.
Extraction throughput on the M4: roughly 11 images/s at 448 px for NIH (31 min for the 20,721-image cache) and 25 to 30
images/s at 224 px; the whole plan took about two days of wall-clock GPU time, one job at a time.
