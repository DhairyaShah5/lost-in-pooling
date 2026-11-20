# Data sources and provenance

All raw data lives under `data/` (gitignored); every derived file is rebuilt by the scripts named below.

| Arm | Source | What we use | Builder |
|---|---|---|---|
| MS-COCO 2017 | cocodataset.org (val2017 + a seeded 10k subset of train2017) | 5,000 test images, 9,000/1,000 train/val, instance boxes and masks for 80 classes | `scripts/00_download.py`, `01_build_manifest.py` |
| NIH ChestX-ray14 | Hugging Face mirror `alkzar90/NIH-Chest-X-ray-dataset` (zip range-reads, 40,880 images) | 880 boxed test images (984 radiologist boxes), 20,721 / 2,238 patient-disjoint train/val images with 14 image-level findings | `scripts/13_build_cxr_manifest.py` |
| ChestX-Det10 | github.com/Deepwise-AILab/ChestX-Det10-Dataset (annotations) and resource.deepwise.com (images) | 3,543 NIH images re-annotated by three radiologists with 8,339 boxes for 10 findings; NIH identities recovered by a pixel hash (1,357 matched) so folds and transfer evaluations stay patient-disjoint | `scripts/00b_download_extras.sh`, `18_build_det10_manifest.py` |
| LUNA16 | zenodo.org/records/3723295 (subsets 0 and 1, annotations.csv, lung masks) | 2D axial lung-window slices: every slice through an annotated nodule (with its box) and six nodule-free lung slices per scan; scan-disjoint folds and a scan-level train/val/test split for the head | `scripts/00b_download_extras.sh`, `21_build_luna16.py` |

Licences: COCO (CC BY 4.0), NIH ChestX-ray14 (NIH Clinical Center, free for research), ChestX-Det10 (research use, cite Liu et al. 2020), LUNA16 (CC BY 4.0 via LIDC-IDRI).
