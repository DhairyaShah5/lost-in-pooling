#!/bin/sh
# Seed repeats of the DINOv2 linear probes (cls, mean) on COCO at 224 px.
set -e
PY=.venv/bin/python
for SD in 1 2; do
  $PY scripts/03_probes.py --kinds cls,mean --seed $SD --readout-suffix _s$SD
done
