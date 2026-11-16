#!/bin/sh
# Bonus: chest X-ray head on the FULL 448 px training cache (removes the 5k-train caveat). DINOv2 only.
set -e
PY=.venv/bin/python
$PY scripts/02_extract.py --backbone dinov2_vitb14_reg --manifest data/nih_cxr/manifest.parquet --features-root features_cxr --res 448 --splits train --batch-size 32 --workers 6
$PY scripts/05_train_head.py --backbone dinov2_vitb14_reg --res 448 --kind attn --lam 0 --lr 1e-3 --epochs 20 --readout-suffix _full \
  --features-root features_cxr --labels data/nih_cxr/labels.npz --manifest data/nih_cxr/manifest.parquet \
  --boxes data/nih_cxr/boxes.parquet --out results_cxr --checkpoints checkpoints_cxr --min-pos 5
