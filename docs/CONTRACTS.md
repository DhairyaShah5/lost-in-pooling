# Data and feature contracts

Every module reads and writes exactly these files. Scripts are numbered and run in order.
Paths are relative to the repo root. Python env: `.venv/bin/python`. Import as `lip.*`
(`src/` layout; `pip install -e .` or pytest's `pythonpath = ["src"]`).

## Dataset: MS-COCO 2017

* `data/coco/annotations/instances_{train,val}2017.json` (official).
* `data/coco/images/val2017/*.jpg`  all 5,000 images  -> split `test`.
* `data/coco/images/train2017/*.jpg` a seeded 10,000-image subset -> splits `train` (9,000) and `val` (1,000).
  Subset: sort train2017 image ids, `numpy.random.default_rng(0).permutation`, take first 10,000;
  first 9,000 of those are `train`, the rest `val`.

Crowd annotations (`iscrowd=1`) are ignored everywhere.
80 categories, indexed 0..79 in ascending COCO `category_id` order.

## `data/coco/manifest.parquet`  (one row per image, N = 15,000)

| column     | dtype  | meaning                                           |
|------------|--------|---------------------------------------------------|
| image_id   | int64  | COCO image id                                     |
| file_name  | str    | e.g. `000000000139.jpg`                           |
| split      | str    | `train`, `val`, or `test`                         |
| width      | int32  | original pixel width                              |
| height     | int32  | original pixel height                             |
| path       | str    | repo-relative path to the jpg                     |

Row order is the canonical order. All per-image arrays below use this order.

## `data/coco/labels.npz`

| key            | shape    | dtype   | meaning                                                        |
|----------------|----------|---------|----------------------------------------------------------------|
| image_ids      | (N,)     | int64   | same order as the manifest                                     |
| split          | (N,)     | str     | same as manifest                                               |
| y              | (N, 80)  | uint8   | 1 if at least one non-crowd instance of the class is present   |
| max_box_frac   | (N, 80)  | float32 | largest instance's bbox area / image area; 0 when absent       |
| max_seg_frac   | (N, 80)  | float32 | largest instance's segmentation `area` / image area; 0 absent  |
| cat_ids        | (80,)    | int64   | COCO category ids in ascending order                           |
| cat_names      | (80,)    | str     | category names                                                 |

**Size variable** used for binning is `max_box_frac` (relative area of the largest
instance of that class in that image). `max_seg_frac` exists so the COCO small/medium/large
convention (segmentation area < 32^2, < 96^2, else large, in original pixels) can be reported too.

## `data/coco/boxes.parquet`  (one row per non-crowd instance)

| column   | dtype   | meaning                                    |
|----------|---------|--------------------------------------------|
| image_id | int64   |                                            |
| cat_idx  | int16   | 0..79                                      |
| x0,y0,x1,y1 | float32 | normalized to [0,1] by image width/height |
| box_frac | float32 | (x1-x0)*(y1-y0)                            |
| seg_frac | float32 | segmentation area / (width*height)         |

## Feature store: `features/{backbone}/res{R}/{split}/`

Backbone names: `dinov2_vitb14_reg` (primary), `dinov2_vits14_reg` (fallback).
R is the square input resolution (224 primary, 448 ablation). Images are **squash-resized**
to R x R (no crop, aspect ratio not preserved) so every box maps into the token grid, then
ImageNet-normalized. Grid is G = R/14 patches per side, P = G*G tokens, row-major from top-left.

| file           | shape       | dtype   | notes                                   |
|----------------|-------------|---------|-----------------------------------------|
| cls.npy        | (n, D)      | float16 | `x_norm_clstoken`                       |
| patch.npy      | (n, P, D)   | float16 | `x_norm_patchtokens`, row-major grid    |
| image_ids.npy  | (n,)        | int64   | manifest order restricted to this split |
| meta.json      |             |         | backbone, res, patch_size=14, grid, dim, n, resize="squash", mean, std |

Always open `patch.npy` with `np.load(path, mmap_mode="r")`; it is several GB.
Helper: `lip.features.store.FeatureStore(backbone, res, split)` exposes
`.cls`, `.patch`, `.image_ids`, `.meta`, `.grid`, and `.rows_for(image_ids)`.

Mapping a normalized box to grid cells: a patch (r, c) has center ((c+0.5)/G, (r+0.5)/G).
A patch is *inside* a box if its center lies in the box. If a box contains no patch center
(sub-patch object), the single patch containing the box center counts as inside.
Helper: `lip.eval.bins.box_to_patch_mask(box_xyxy, grid) -> (G*G,) bool`.

## Size bins

Six log-spaced bins on `max_box_frac`, edges:
`[0, 0.001, 0.003, 0.01, 0.03, 0.10, 1.0]`, names
`<0.1%`, `0.1-0.3%`, `0.3-1%`, `1-3%`, `3-10%`, `>10%`.
At R=224 one patch covers 1/256 = 0.39% of the image, so the first two bins are sub-patch.
Also report COCO S/M/L (from `max_seg_frac` and original pixel area).

## Metric

For class c and bin b: AUROC where positives are images with `y[:,c]=1` and
`max_box_frac[:,c]` in bin b, negatives are all images with `y[:,c]=0` (shared across bins).
Macro-average over classes with at least `min_pos=10` positives in that bin.
Bootstrap (resample images, 500 draws) gives a 95% CI on the macro average.
Also report overall (all sizes pooled) macro AUROC and mAP.

## Results JSON: `results/{readout}__{backbone}__res{R}.json`

```json
{
  "readout": "cls", "backbone": "dinov2_vitb14_reg", "res": 224, "split": "test",
  "n_images": 5000, "train_split": "train", "notes": "...",
  "bins": {"edges": [...], "names": [...]},
  "per_bin": {"<0.1%": {"auroc_macro": 0.71, "ci_lo": 0.69, "ci_hi": 0.73,
                         "n_classes": 42, "n_pos": 812}, ...},
  "overall": {"auroc_macro": 0.93, "map": 0.71},
  "per_class_bin": {"person": {"<0.1%": {"auroc": 0.6, "n_pos": 30}, ...}, ...}
}
```

Scores for the test split are also saved as `results/scores/{readout}__{backbone}__res{R}.npy`
with shape (n_test, 80), float32, test-manifest order, so curves can be recomputed without retraining.

## Read-out names

`cls`, `mean`, `cls_mean` (concat), `max` (per-dimension max over patches), `maxmil`
(per-patch linear scorer, max over patches), `oracle` (mean of patches inside GT boxes of
the class; negatives use a random box with area drawn from that class's positive box areas),
`attn` (gated attention MIL head), `attn_ms` (multi-scale variant), with suffixes like
`attn_ent0.01` for regularizer settings.

## Rules for contributors (humans and agents)

* Never load `patch.npy` fully into RAM (machine has 16 GB). Index memmaps in batches.
* Everything deterministic: seed 0 everywhere, record seeds in JSON.
* Scripts take `--backbone`, `--res`, `--out` flags with the defaults above.
* Do not commit; the repo owner commits. Do not add dependencies beyond `requirements.txt`
  without saying so in your report.
* Tests live in `tests/` and must run on synthetic data in seconds without the real dataset.
