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
# report
report/build.sh
```

Tests: `.venv/bin/python -m pytest -q` (synthetic data, no download needed).

File formats, bins, metric and result schema: `docs/CONTRACTS.md`. Plan and decisions: `docs/PLAN.md`.
Progress checklist: `TODO.md`. Committed results live in `results/` (JSON + per-image test scores).
