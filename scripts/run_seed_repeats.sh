#!/bin/sh
# Seed repeats (1, 2) for every read-out that had a single seed at the midterm:
# COCO oracles (DINOv2 at 224/336/448, CLIP, SigLIP), CLIP/SigLIP probes, DINOv2 max/max-MIL, chest probes.
set -e
PY=.venv/bin/python
CX="--features-root features_cxr --labels data/nih_cxr/labels.npz --manifest data/nih_cxr/manifest.parquet --out results_cxr --checkpoints checkpoints_cxr --min-pos 5"
for SD in 1 2; do
  for R in 224 336 448; do
    $PY scripts/04_oracle.py --backbone dinov2_vitb14_reg --res $R --seed $SD --readout-suffix _s$SD --force
  done
  for BB in clip_vitb16 siglip_vitb16; do
    $PY scripts/04_oracle.py --backbone $BB --res 224 --seed $SD --readout-suffix _s$SD --force
    $PY scripts/03_probes.py --backbone $BB --res 224 --kinds cls,mean,max,maxmil --seed $SD --readout-suffix _s$SD
  done
  $PY scripts/03_probes.py --backbone dinov2_vitb14_reg --res 224 --kinds max,maxmil --seed $SD --readout-suffix _s$SD
  for BB in dinov2_vitb14_reg rad_dino; do
    $PY scripts/03_probes.py --backbone $BB --res 224 --kinds cls,mean,max --seed $SD --readout-suffix _s$SD $CX
  done
  $PY scripts/03_probes.py --backbone dinov2_vitb14_reg --res 448 --kinds cls,mean,max --seed $SD --readout-suffix _full_s$SD $CX
done
echo "SEEDS_DONE $(date)"
