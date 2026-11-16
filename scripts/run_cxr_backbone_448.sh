#!/bin/sh
# Chest X-ray arm at 448 px: CV read-outs on boxed images, image-level probes (5k train subset), attention head.
set -e
BB=${1:?backbone}; RES=448
PY=.venv/bin/python
CX="--features-root features_cxr --labels data/nih_cxr/labels.npz --manifest data/nih_cxr/manifest.parquet --out results_cxr"
$PY scripts/14_cxr_cv_readouts.py --backbone "$BB" --res $RES --folds 5 --min-pos 5
$PY scripts/03_probes.py --backbone "$BB" --res $RES --kinds cls,mean,max --min-pos 5 $CX --checkpoints checkpoints_cxr
$PY scripts/05_train_head.py --backbone "$BB" --res $RES --kind attn --lam 0 --lr 1e-3 --epochs 20 --min-pos 5 $CX --boxes data/nih_cxr/boxes.parquet --checkpoints checkpoints_cxr
