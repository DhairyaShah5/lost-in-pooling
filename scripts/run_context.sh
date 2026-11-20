#!/bin/sh
# Context-only control: the oracle protocol with the tokens OUTSIDE the box (what the region read-out does not see).
set -e
PY=.venv/bin/python
for BB in dinov2_vitb14_reg clip_vitb16 siglip_vitb16; do
  $PY scripts/04_oracle.py --backbone $BB --res 224 --region outside
done
$PY scripts/04_oracle.py --backbone dinov2_vitb14_reg --res 448 --region outside
CX="--features-root features_cxr --labels data/nih_cxr/labels.npz --boxes data/nih_cxr/boxes.parquet --out results_cxr --min-pos 5 --folds 5"
for BB in dinov2_vitb14_reg rad_dino; do
  $PY scripts/14_cxr_cv_readouts.py --backbone $BB --res 224 --region outside $CX
done
echo "CONTEXT_DONE $(date)"
