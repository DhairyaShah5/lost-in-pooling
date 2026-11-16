#!/bin/sh
# Run the full read-out suite on an already-extracted backbone: probes, oracle, attention head, plots.
# usage: scripts/run_backbone.sh <backbone> [res]
set -e
BB=${1:?backbone name}; RES=${2:-224}
PY=.venv/bin/python
$PY scripts/03_probes.py --backbone "$BB" --res "$RES" --kinds cls,mean,cls_mean,max,maxmil
$PY scripts/04_oracle.py --backbone "$BB" --res "$RES"
$PY scripts/05_train_head.py --backbone "$BB" --res "$RES" --kind attn --lam 0 --lr 1e-3 --epochs 20
$PY scripts/07_plots.py --backbone "$BB" --res "$RES" --fig-readouts cls,mean,max,maxmil,attn,oracle
