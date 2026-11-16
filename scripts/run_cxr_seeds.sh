#!/bin/sh
# Seed repeats of the chest X-ray attention head (224 px) for the given backbones.
set -e
PY=.venv/bin/python
for BB in "$@"; do
  for SD in 1 2; do
    $PY scripts/05_train_head.py --backbone "$BB" --res 224 --kind attn --lam 0 --lr 1e-3 --epochs 20 --seed $SD --readout-suffix _s$SD \
      --features-root features_cxr --labels data/nih_cxr/labels.npz --manifest data/nih_cxr/manifest.parquet \
      --boxes data/nih_cxr/boxes.parquet --out results_cxr --checkpoints checkpoints_cxr --min-pos 5
  done
done
