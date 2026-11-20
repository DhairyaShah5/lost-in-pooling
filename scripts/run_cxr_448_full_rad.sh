#!/bin/sh
# RAD-DINO chest arm on the FULL 448 px training cache: extraction, three head seeds, like-for-like probes.
set -e
PY=.venv/bin/python
CX="--features-root features_cxr --labels data/nih_cxr/labels.npz --manifest data/nih_cxr/manifest.parquet --out results_cxr --checkpoints checkpoints_cxr --min-pos 5"
$PY scripts/02_extract.py --backbone rad_dino --manifest data/nih_cxr/manifest.parquet --features-root features_cxr --res 448 --splits train --batch-size 32 --workers 6
echo "EXTRACT_DONE $(date)"
head() { $PY scripts/05_train_head.py --backbone rad_dino --res 448 --kind attn --lam 0 --lr 1e-3 --epochs 20 --seed $1 --readout-suffix "$2" $CX --boxes data/nih_cxr/boxes.parquet; }
( head 0 _full ) & ( head 1 _full_s1 ) & wait
( head 2 _full_s2 ) & ( $PY scripts/03_probes.py --backbone rad_dino --res 448 --kinds cls,mean,max --readout-suffix _full $CX ) & wait
echo "RAD448_FULL_DONE $(date)"
