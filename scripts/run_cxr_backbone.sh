#!/bin/sh
# Chest X-ray arm on an extracted backbone: CV read-outs on boxed images, image-level probes, attention head.
# usage: scripts/run_cxr_backbone.sh <backbone> [res]
set -e
BB=${1:?backbone}; RES=${2:-224}
PY=.venv/bin/python
CX="--features-root features_cxr --labels data/nih_cxr/labels.npz --manifest data/nih_cxr/manifest.parquet --out results_cxr"
$PY scripts/14_cxr_cv_readouts.py --backbone "$BB" --res "$RES" --folds 5 --min-pos 5
$PY scripts/03_probes.py --backbone "$BB" --res "$RES" --kinds cls,mean,max,maxmil --min-pos 5 $CX --checkpoints checkpoints_cxr
$PY scripts/05_train_head.py --backbone "$BB" --res "$RES" --kind attn --lam 0 --lr 1e-3 --epochs 20 --min-pos 5 $CX --boxes data/nih_cxr/boxes.parquet --checkpoints checkpoints_cxr
$PY scripts/05_train_head.py --backbone "$BB" --res "$RES" --kind attn_ms --lam 0 --lr 1e-3 --epochs 20 --min-pos 5 $CX --boxes data/nih_cxr/boxes.parquet --checkpoints checkpoints_cxr
