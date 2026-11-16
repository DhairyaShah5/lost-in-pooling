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
scripts/        numbered pipeline steps, run in order
results/        committed JSON results and figures
report/         midterm report (LaTeX + PDF)
docs/           plan, data/feature contracts, todo
data/ features/ local only, gitignored
```

## Headline results (COCO val2017, 5,000 images, macro AUROC on sub-patch targets, <0.1% of the image)

| Backbone | CLS probe | Mean probe | Attention head | Oracle region |
|---|---|---|---|---|
| DINOv2 ViT-B/14-reg | 0.832 | 0.862 | 0.899 (multi-scale 0.908) | 0.973 |
| CLIP ViT-B/16 | 0.832 | 0.838 | 0.910 (multi-scale 0.913) | 0.935 |
| SigLIP ViT-B/16 | 0.824 | 0.847 | 0.872 (multi-scale 0.878) | 0.934 |

Every read-out in a row is trained and evaluated on the same cached tokens. The oracle pools only the
patches under the ground-truth boxes (negatives get a matched random box). Large targets score ~0.99 for
every read-out. Doubling the input to 448 px lifts the oracle to 0.994 and leaves the global read-outs
unchanged: global pooling tracks the fraction of tokens an object occupies, not the patches it covers.

Chest radiographs (NIH ChestX-ray14, 880 boxed images): with DINOv2, global probes score 0.58 (5-fold CV) to
0.70 (trained on 20k radiographs) overall AUROC while the oracle on the same tokens reaches 0.88 to 0.94 per
lesion-size bin and the attention head 0.73; RAD-DINO lifts every read-out (oracle 0.95 to 0.97, probes 0.77)
without closing the gap. At 448 px the chest oracle rises to 0.96 to 0.98 and the global probes do not move, while the attention head trained on the full 448 px cache reaches 0.77 +- 0.01 over three seeds (0.73 +- 0.02 at 224 px): the learned read-out gains from resolution where the global probes do not. Full tables, intervals and per-class breakdowns: `report/midterm.pdf`, `results/`,
`results_cxr/`.

## Reproduce

Tested on an Apple M4 laptop (16 GB, MPS). Full pipeline is under 40 minutes; the probes
themselves take seconds to minutes on cached features.

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python scripts/00_download.py          # COCO annotations + val2017 + seeded 10k train subset (~2.4 GB)
.venv/bin/python scripts/01_build_manifest.py    # manifest.parquet, labels.npz, boxes.parquet
.venv/bin/python scripts/01b_dataset_stats.py    # positives per size bin
.venv/bin/python scripts/02_extract.py           # DINOv2 ViT-B/14-reg, 224 px, fp16 CLS + patch tokens (~7 min)
.venv/bin/python scripts/03_probes.py            # cls / mean / cls_mean / max / maxmil read-outs
.venv/bin/python scripts/04_oracle.py            # oracle region read-out (matched random negatives)
.venv/bin/python scripts/05_train_head.py --lam 0        # gated attention head + localisation audit
.venv/bin/python scripts/07_plots.py --fig-readouts cls,mean,max,maxmil,attn,oracle
.venv/bin/python scripts/08_attention_figure.py
# resolution ablation
.venv/bin/python scripts/02_extract.py --res 448 --limit 5000
.venv/bin/python scripts/03_probes.py --res 448 --kinds cls,mean,max && .venv/bin/python scripts/04_oracle.py --res 448
# more backbones, chest X-rays, analyses
scripts/run_backbone.sh clip_vitb16 && scripts/run_backbone.sh siglip_vitb16
.venv/bin/python scripts/12_cross_backbone.py --backbones dinov2_vitb14_reg,clip_vitb16,siglip_vitb16
.venv/bin/python scripts/13_build_cxr_manifest.py      # after placing NIH ChestX-ray14 under data/nih_cxr/
.venv/bin/python scripts/02_extract.py --manifest data/nih_cxr/manifest.parquet --features-root features_cxr
scripts/run_cxr_backbone.sh dinov2_vitb14_reg
.venv/bin/python scripts/09_perclass.py && .venv/bin/python scripts/10_compare.py --pairs attn:mean,oracle:mean
.venv/bin/python scripts/11_seeds.py --readouts attn --suffixes ",_s1,_s2"
.venv/bin/python scripts/15_patch_units.py
# report
report/build.sh
```

Tests: `.venv/bin/python -m pytest -q` (synthetic data, no download needed).

File formats, bins, metric and result schema: `docs/CONTRACTS.md`. Plan and decisions: `docs/PLAN.md`.
Progress checklist: `TODO.md`. Committed results live in `results/` (JSON + per-image test scores).
