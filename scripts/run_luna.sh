#!/bin/sh
# LUNA16 slice arm (DINOv2): scan-disjoint CV read-outs + context control on every slice, image-level head on a
# scan split, at 224 and 448 px; then the diameter curve.
set -e
PY=.venv/bin/python
BB=${1:-dinov2_vitb14_reg}
LX="--features-root features_luna --labels data/luna16/labels.npz --boxes data/luna16/boxes.parquet --out results_luna --min-pos 5 --folds 5"
HX="--features-root features_luna_head --labels data/luna16/labels_head.npz --manifest data/luna16/manifest_head.parquet --boxes data/luna16/boxes.parquet --out results_luna --checkpoints checkpoints_luna --min-pos 5"
for RES in 224 448; do
  BS=64; [ "$RES" -eq 448 ] && BS=32
  $PY scripts/02_extract.py --backbone $BB --manifest data/luna16/manifest.parquet --features-root features_luna --res $RES --splits test --batch-size $BS --workers 6
  $PY scripts/14_cxr_cv_readouts.py --backbone $BB --res $RES $LX
  $PY scripts/14_cxr_cv_readouts.py --backbone $BB --res $RES --region outside $LX
  $PY scripts/02_extract.py --backbone $BB --manifest data/luna16/manifest_head.parquet --features-root features_luna_head --res $RES --splits train,val,test --batch-size $BS --workers 6
  for SD in 0 1 2; do
    SFX=""; [ "$SD" -gt 0 ] && SFX="_s$SD"
    $PY scripts/05_train_head.py --backbone $BB --res $RES --kind attn --lam 0 --lr 1e-3 --epochs 30 --patience 8 --seed $SD --readout-suffix "$SFX" $HX
  done
  $PY scripts/22_luna_curve.py --backbone $BB --res $RES
done
echo "LUNA_DONE $(date)"
