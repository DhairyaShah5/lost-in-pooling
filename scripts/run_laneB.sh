#!/bin/sh
# Lane B (COCO/other GPU work, sequential): Det10 DINOv2 448 transfer, DINO ViT-B/16 remainder, COCO distillation, LUNA16.
PY=.venv/bin/python
step() { echo "== STEP $(date): $*"; "$@" || echo "STEP_FAILED: $*"; }
DX="--features-root features_det10 --labels data/chestx_det10/labels.npz --boxes data/chestx_det10/boxes.parquet --out results_det10 --min-pos 5"
step $PY scripts/20_det10_transfer.py --backbone dinov2_vitb14_reg --res 448 --heads attn_full,attn_full_s1,attn_full_s2
BB=dino_vitb16
[ -f results/attn_ms_s2__${BB}__res224.json ] || step $PY scripts/05_train_head.py --backbone $BB --res 224 --kind attn_ms --lam 0 --lr 1e-3 --epochs 20 --seed 2 --readout-suffix _s2
[ -f features/$BB/res448/train/meta.json ] || step $PY scripts/02_extract.py --backbone $BB --res 448 --limit 5000 --batch-size 32 --workers 6
step $PY scripts/03_probes.py --backbone $BB --res 448 --kinds cls,mean,max
step $PY scripts/04_oracle.py --backbone $BB --res 448
step $PY scripts/03_probes.py --backbone $BB --res 224 --kinds cls,mean,max --train-limit 5000 --readout-suffix _train5k
step $PY scripts/07_plots.py --backbone $BB --res 224 --fig-readouts cls,mean,max,maxmil,attn,attn_ms,oracle
echo "DINO16_DONE $(date)"
step /bin/sh scripts/run_distill.sh coco
step /bin/sh scripts/run_luna.sh
echo "LANE_B_DONE $(date)"
