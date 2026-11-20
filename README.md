# Lost in Pooling

Measuring and recovering small-signal evidence discarded by frozen foundation models.

CSCI 566, Deep Learning and its Applications, University of Southern California, Fall 2026.

## Idea

Frozen vision transformers keep evidence about small objects in their patch tokens,
but the usual recipe (take the global CLS/mean embedding, fit a linear probe) collapses
those tokens into one vector and the evidence is lost. We measure detection quality as a
function of relative target size (a "blindness curve"), compare several read-outs over
the *same* cached tokens, and train a small attention head that learns where to look
from image-level labels only.

## Layout

```
src/lip/        library code (data, features, readouts, models, eval)
scripts/        numbered pipeline steps (run in order), run_*.sh arm runners, demo.py
results/        committed JSON results, per-image test scores and figures (results_cxr/, results_det10/, results_luna/ likewise)
report/         midterm and final reports (LaTeX + PDF); slides/ clinic deck; poster/ poster
docs/           plan, data sources and layout (DATA.md, NIH_CXR_SOURCES.md), file contracts, exact image lists (docs/data/)
checkpoints*/   trained heads and probe weights (local; the demo loads these)
data/ features*/ raw data and fp16 token caches, local only, gitignored, rebuilt by the scripts
```

## Headline results (COCO val2017, 5,000 images, macro AUROC on sub-patch targets, <0.1% of the image)

| Backbone | CLS probe | Mean probe | Attention head | Oracle region |
|---|---|---|---|---|
| DINOv2 ViT-B/14-reg | 0.832 | 0.862 | 0.899 (multi-scale 0.908) | 0.973 |
| CLIP ViT-B/16 | 0.832 | 0.838 | 0.910 (multi-scale 0.913) | 0.935 |
| SigLIP ViT-B/16 | 0.824 | 0.847 | 0.872 (multi-scale 0.878) | 0.934 |
| DINO ViT-B/16 | 0.804 | 0.798 | 0.824 (multi-scale 0.822) | 0.900 |

Every read-out in a row is trained and evaluated on the same cached tokens (heads: mean of three seeds). The oracle pools only the
patches under the ground-truth boxes (negatives get a matched random box). Large targets score ~0.99 for
every read-out. Doubling the input to 448 px lifts the oracle to 0.994 and leaves the global read-outs
unchanged: global pooling tracks the fraction of tokens an object occupies, not the patches it covers.

Chest radiographs (NIH ChestX-ray14, 880 boxed images): with DINOv2, global probes score 0.58 (5-fold CV) to
0.70 (trained on 20k radiographs) overall AUROC while the oracle on the same tokens reaches 0.88 to 0.94 per
lesion-size bin and the attention head 0.73 +- 0.02; RAD-DINO lifts every read-out (oracle 0.95 to 0.97, probes 0.77)
without closing the gap. At 448 px the chest oracle rises to 0.96 to 0.98 and the cross-validated global probes do not
move, while the attention head trained on the full 448 px cache reaches 0.77 +- 0.01 (DINOv2) and 0.80 +- 0.01
(RAD-DINO) over three seeds. A context-only control (the oracle protocol on the tokens *outside* the box) scores the
same as the global probes on sub-patch targets, so the global vector's floor is the scene. ChestX-Det10 (3,543 NIH
images re-annotated with 8,339 boxes) shows the same cliff on 5.4x the boxes, and the NIH-trained head transfers to
the new annotators better than the probes do (0.77 vs 0.70 for DINOv2 and 0.83 vs 0.80 for RAD-DINO at 448 px).
Tiled-teacher distillation does not move a 224 px head off its baseline on radiographs (students 0.72 to 0.74 vs
teacher 0.77) and gains at most 0.01 on COCO, while tiling the unchanged 224 px head over 448 px tokens at test time
recovers three quarters of the 448 px head's gain on COCO (0.91 vs 0.92, baseline 0.87): the evidence a 448 px head
uses is not in the 224 px tokens. On LUNA16 CT slices (subsets 0 and 1, 2,507 slices), where every nodule is a
sub-patch target, the cross-validated global probes score 0.61 AUROC at every nodule diameter against 0.95 for the
oracle on the same tokens. Full tables, intervals and per-class
breakdowns: `report/final.pdf`, `results/`, `results_cxr/`, `results_det10/`, `results_luna/`.

## Reproduce

Tested on an Apple M4 laptop (16 GB, MPS, one GPU job at a time). The core COCO arm (first block below) runs in
under 40 minutes; the whole plan took about two days of wall-clock GPU time, dominated by token extraction at 448 px.
Disk: the raw data and fp16 token caches of the full plan occupy about 255 GB (`features*/` 190 GB, `data/` 57 GB,
attention maps 6 GB). All of it is gitignored and rebuilt by the commands below; the committed repository is 240 MB.
Data sources, licences, the expected layout under `data/` and the per-folder disk budget: `docs/DATA.md`.

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
# core COCO arm (DINOv2 ViT-B/14-reg, 224 px)
.venv/bin/python scripts/00_download.py          # COCO annotations + val2017 + seeded 10k train subset (~2.4 GB)
.venv/bin/python scripts/01_build_manifest.py    # manifest.parquet, labels.npz, boxes.parquet
.venv/bin/python scripts/01b_dataset_stats.py    # positives per size bin
.venv/bin/python scripts/02_extract.py           # DINOv2 ViT-B/14-reg, 224 px, fp16 CLS + patch tokens (~7 min)
.venv/bin/python scripts/03_probes.py            # cls / mean / cls_mean / max / maxmil read-outs
.venv/bin/python scripts/04_oracle.py            # oracle region read-out (matched random negatives)
.venv/bin/python scripts/05_train_head.py --lam 0        # gated attention head + localisation audit
.venv/bin/python scripts/07_plots.py --fig-readouts cls,mean,max,maxmil,attn,oracle
.venv/bin/python scripts/08_attention_figure.py
# resolution ablation at 336 and 448 px (the higher-resolution train caches hold the first 5,000 training images)
.venv/bin/python scripts/02_extract.py --res 336 --limit 5000
.venv/bin/python scripts/03_probes.py --res 336 --kinds cls,mean,max && .venv/bin/python scripts/04_oracle.py --res 336
.venv/bin/python scripts/02_extract.py --res 448 --limit 5000 --batch-size 32
.venv/bin/python scripts/03_probes.py --res 448 --kinds cls,mean,max && .venv/bin/python scripts/04_oracle.py --res 448
# more backbones on COCO
.venv/bin/python scripts/02_extract.py --backbone clip_vitb16 && scripts/run_backbone.sh clip_vitb16
.venv/bin/python scripts/02_extract.py --backbone siglip_vitb16 && scripts/run_backbone.sh siglip_vitb16
.venv/bin/python scripts/12_cross_backbone.py --backbones dinov2_vitb14_reg,clip_vitb16,siglip_vitb16
# chest X-rays (NIH ChestX-ray14) at 224 px: labels + 40,880 images from the Hugging Face mirror (~15 min), then both backbones
scripts/00c_download_nih_cxr.sh && .venv/bin/python scripts/13_build_cxr_manifest.py
.venv/bin/python scripts/02_extract.py --manifest data/nih_cxr/manifest.parquet --features-root features_cxr
scripts/run_cxr_backbone.sh dinov2_vitb14_reg
.venv/bin/python scripts/02_extract.py --backbone rad_dino --manifest data/nih_cxr/manifest.parquet --features-root features_cxr
scripts/run_cxr_backbone.sh rad_dino
scripts/run_cxr_seeds.sh dinov2_vitb14_reg rad_dino          # head seeds 1 and 2
.venv/bin/python scripts/09_perclass.py && .venv/bin/python scripts/10_compare.py --pairs attn:mean,oracle:mean
.venv/bin/python scripts/11_seeds.py --readouts attn --suffixes ",_s1,_s2"
.venv/bin/python scripts/15_patch_units.py
# chest X-rays at 448 px: CV read-outs and 5k-subset probes/head, then heads (three seeds) and like-for-like probes on
# the full 20,721-image training cache (30 GB per backbone, ~31 min to extract)
.venv/bin/python scripts/02_extract.py --manifest data/nih_cxr/manifest.parquet --features-root features_cxr --res 448 --limit 5000 --batch-size 32
scripts/run_cxr_backbone_448.sh dinov2_vitb14_reg
scripts/run_cxr_448_full.sh && scripts/run_cxr_448_full_probes.sh && scripts/run_cxr_448_full_seeds.sh 1 && scripts/run_cxr_448_full_seeds.sh 2
.venv/bin/python scripts/02_extract.py --backbone rad_dino --manifest data/nih_cxr/manifest.parquet --features-root features_cxr --res 448 --limit 5000 --batch-size 32
scripts/run_cxr_backbone_448.sh rad_dino && scripts/run_cxr_448_full_rad.sh
# context-only control (the oracle protocol with the tokens outside the box)
scripts/run_context.sh
# fourth backbone (DINO ViT-B/16), then seed repeats for every read-out that had one seed
scripts/run_dino16.sh
.venv/bin/python scripts/12_cross_backbone.py --backbones dinov2_vitb14_reg,clip_vitb16,siglip_vitb16,dino_vitb16
scripts/run_seed_repeats.sh && scripts/run_probe_seeds.sh
# ChestX-Det10 and LUNA16 raw data (Det10 1.4 GB; LUNA16 annotations, lung masks and subsets 0 and 1, ~13 GB of zips)
scripts/00b_download_extras.sh
.venv/bin/python scripts/18_build_det10_manifest.py && scripts/run_det10.sh
# tiled-teacher distillation (COCO and radiographs) and its table
scripts/run_distill.sh coco && scripts/run_distill.sh chest
.venv/bin/python scripts/23_distill_table.py --dataset coco && .venv/bin/python scripts/23_distill_table.py --dataset chest
# LUNA16 slices (the builder unzips the archives), then CV read-outs, context control, heads and the diameter curve at 224 and 448 px
.venv/bin/python scripts/21_build_luna16.py --subsets subset0,subset1 && scripts/run_luna.sh && .venv/bin/python scripts/24_luna_table.py
# pre-deployment check (fits and figure), then the CLI for a token grid and a target size
.venv/bin/python scripts/17_deploy_check.py
.venv/bin/python -m lip.eval.deploy_check --grid 16 --target-frac 0.002
# live demo on one image: curve position, attention map, probe scores (needs only checkpoints/ and results/, no caches)
.venv/bin/python scripts/demo.py --image path/to/image.jpg --out demo.png
# report, slides, poster (tectonic)
report/build.sh && (cd slides && tectonic clinic.tex) && (cd poster && tectonic poster.tex)
```

`scripts/run_laneA.sh` and `scripts/run_laneB.sh` are the two sequential GPU lanes that ran the last phase (RAD-DINO at
448 px, ChestX-Det10 at 448 px, DINO ViT-B/16 at 448 px, distillation, LUNA16); each step is skipped when its output exists.

Tests: `.venv/bin/python -m pytest -q` (59 tests on synthetic data, no download needed).

File formats, bins, metric and result schema: `docs/CONTRACTS.md`. Plan and decisions: `docs/PLAN.md`.
Progress checklist: `TODO.md`. Committed results live in `results/` (JSON + per-image test scores).
