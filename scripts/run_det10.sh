#!/bin/sh
# ChestX-Det10 arm: features at 224/448 for DINOv2 and RAD-DINO, patient-disjoint CV read-outs, NIH-trained transfer.
set -e
PY=.venv/bin/python
DX="--features-root features_det10 --labels data/chestx_det10/labels.npz --boxes data/chestx_det10/boxes.parquet --out results_det10 --min-pos 5"
for BB in dinov2_vitb14_reg rad_dino; do
  [ -f features_det10/$BB/res224/test/meta.json ] || $PY scripts/02_extract.py --backbone $BB --manifest data/chestx_det10/manifest.parquet --features-root features_det10 --res 224 --splits test --batch-size 64 --workers 6
  $PY scripts/14_cxr_cv_readouts.py --backbone $BB --res 224 --folds 5 $DX
  $PY scripts/20_det10_transfer.py --backbone $BB --res 224 --heads attn,attn_s1,attn_s2,attn_ms
done
for BB in dinov2_vitb14_reg rad_dino; do
  [ -f features_det10/$BB/res448/test/meta.json ] || $PY scripts/02_extract.py --backbone $BB --manifest data/chestx_det10/manifest.parquet --features-root features_det10 --res 448 --splits test --batch-size 32 --workers 6
  $PY scripts/14_cxr_cv_readouts.py --backbone $BB --res 448 --folds 5 $DX
  $PY scripts/20_det10_transfer.py --backbone $BB --res 448 --heads attn_full,attn_full_s1,attn_full_s2
done
echo "DET10_DONE $(date)"
