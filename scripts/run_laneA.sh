#!/bin/sh
# Lane A (chest GPU work, sequential): RAD-DINO full-448 heads and probes, Det10 RAD-DINO 448, chest distillation.
PY=.venv/bin/python
step() { echo "== STEP $(date): $*"; "$@" || echo "STEP_FAILED: $*"; }
CX="--features-root features_cxr --labels data/nih_cxr/labels.npz --manifest data/nih_cxr/manifest.parquet --out results_cxr --checkpoints checkpoints_cxr --min-pos 5"
for SD in 0 1 2; do
  SFX=_full; [ "$SD" -gt 0 ] && SFX="_full_s$SD"
  [ -f results_cxr/attn${SFX}__rad_dino__res448.json ] || step $PY scripts/05_train_head.py --backbone rad_dino --res 448 --kind attn --lam 0 --lr 1e-3 --epochs 20 --seed $SD --readout-suffix "$SFX" $CX --boxes data/nih_cxr/boxes.parquet
done
step $PY scripts/03_probes.py --backbone rad_dino --res 448 --kinds cls,mean,max --readout-suffix _full $CX
DX="--features-root features_det10 --labels data/chestx_det10/labels.npz --boxes data/chestx_det10/boxes.parquet --out results_det10 --min-pos 5"
[ -f features_det10/rad_dino/res448/test/meta.json ] || step $PY scripts/02_extract.py --backbone rad_dino --manifest data/chestx_det10/manifest.parquet --features-root features_det10 --res 448 --splits test --batch-size 32 --workers 6
step $PY scripts/14_cxr_cv_readouts.py --backbone rad_dino --res 448 --folds 5 $DX
step $PY scripts/20_det10_transfer.py --backbone rad_dino --res 448 --heads attn_full,attn_full_s1,attn_full_s2
step /bin/sh scripts/run_distill.sh chest
echo "LANE_A_DONE $(date)"
