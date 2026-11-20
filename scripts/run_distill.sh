#!/bin/sh
# Tiled-teacher distillation. usage: scripts/run_distill.sh coco|chest
set -e
PY=.venv/bin/python
if [ "${1:-coco}" = "coco" ]; then
  # high-resolution teacher for COCO: a head on the 448 px tokens (train cache = the 5,000-image subset)
  [ -f checkpoints/attn__dinov2_vitb14_reg__res448.pt ] || $PY scripts/05_train_head.py --backbone dinov2_vitb14_reg --res 448 --kind attn --lam 0 --lr 1e-3 --epochs 20
  $PY scripts/19_distill.py --backbone dinov2_vitb14_reg --suffix _train5k --students tiled,hi,none --seeds 0,1,2
  echo "DISTILL_COCO_DONE $(date)"
else
  $PY scripts/19_distill.py --backbone dinov2_vitb14_reg --features-root features_cxr --labels data/nih_cxr/labels.npz \
    --manifest data/nih_cxr/manifest.parquet --boxes data/nih_cxr/boxes.parquet --out results_cxr --checkpoints checkpoints_cxr \
    --min-pos 5 --teacher-hi checkpoints_cxr/attn_full__dinov2_vitb14_reg__res448.pt --students tiled,hi --seeds 0,1,2
  echo "DISTILL_CHEST_DONE $(date)"
fi
