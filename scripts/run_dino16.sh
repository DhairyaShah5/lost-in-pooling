#!/bin/sh
# Fourth backbone, DINO ViT-B/16 (self-supervised, 16 px patch) on COCO: 224 px suite with three head seeds,
# then the 448 px resolution ablation on the 5,000-image training subset, then plots.
set -e
PY=.venv/bin/python
BB=dino_vitb16
$PY scripts/02_extract.py --backbone $BB --res 224 --batch-size 64 --workers 6
echo "EXTRACT224_DONE $(date)"
$PY scripts/03_probes.py --backbone $BB --res 224 --kinds cls,mean,cls_mean,max,maxmil
$PY scripts/04_oracle.py --backbone $BB --res 224
for SD in 0 1 2; do
  SFX=""; [ "$SD" -gt 0 ] && SFX="_s$SD"
  $PY scripts/05_train_head.py --backbone $BB --res 224 --kind attn --lam 0 --lr 1e-3 --epochs 20 --seed $SD --readout-suffix "$SFX"
  $PY scripts/05_train_head.py --backbone $BB --res 224 --kind attn_ms --lam 0 --lr 1e-3 --epochs 20 --seed $SD --readout-suffix "$SFX"
done
$PY scripts/07_plots.py --backbone $BB --res 224 --fig-readouts cls,mean,max,maxmil,attn,attn_ms,oracle
$PY scripts/02_extract.py --backbone $BB --res 448 --limit 5000 --batch-size 32 --workers 6
echo "EXTRACT448_DONE $(date)"
$PY scripts/03_probes.py --backbone $BB --res 448 --kinds cls,mean,max
$PY scripts/04_oracle.py --backbone $BB --res 448
$PY scripts/03_probes.py --backbone $BB --res 224 --kinds cls,mean,max --train-limit 5000 --readout-suffix _train5k
echo "DINO16_DONE $(date)"
