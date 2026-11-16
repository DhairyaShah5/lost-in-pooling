#!/bin/sh
# Seed repeats (1, 2) of the DINOv2 chest head on the FULL 448 px training cache. Run after run_cxr_448_full.sh.
set -e
PY=.venv/bin/python
SD=$1
$PY scripts/05_train_head.py --backbone dinov2_vitb14_reg --res 448 --kind attn --lam 0 --lr 1e-3 --epochs 20 --seed $SD --readout-suffix _full_s$SD \
  --features-root features_cxr --labels data/nih_cxr/labels.npz --manifest data/nih_cxr/manifest.parquet \
  --boxes data/nih_cxr/boxes.parquet --out results_cxr --checkpoints checkpoints_cxr --min-pos 5
