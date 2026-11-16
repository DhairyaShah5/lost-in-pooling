#!/bin/sh
# Like-for-like global probes on the FULL 448 px chest cache (DINOv2), read-outs cls_full / mean_full / max_full.
set -e
PY=.venv/bin/python
$PY scripts/03_probes.py --backbone dinov2_vitb14_reg --res 448 --kinds cls,mean,max --min-pos 5 --readout-suffix _full \
  --features-root features_cxr --labels data/nih_cxr/labels.npz --manifest data/nih_cxr/manifest.parquet --out results_cxr --checkpoints checkpoints_cxr
