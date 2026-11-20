#!/bin/sh
# NIH ChestX-ray14 subset used by the chest arm: the five label/list files and 40,880 PNGs (all 880 boxed images plus a
# seeded random 40,000), pulled from the public Hugging Face mirror alkzar90/NIH-Chest-X-ray-dataset without downloading
# the 46 GB of archives in full (HTTP Range requests into the ZIP members; CRC-checked). About 16.5 GB transferred,
# 15 GB on disk, ~15 min at 20 MB/s. Provenance and counts: docs/NIH_CXR_SOURCES.md.
set -e
PY=.venv/bin/python
D=data/nih_cxr; mkdir -p "$D/_index" "$D/images"
HF="https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data"
for f in BBox_List_2017.csv Data_Entry_2017_v2020.csv train_val_list.txt test_list.txt; do
  [ -f "$D/$f" ] || curl -sL -o "$D/$f" "$HF/$f"
done
[ -f "$D/Data_Entry_2017.csv" ] || cp "$D/Data_Entry_2017_v2020.csv" "$D/Data_Entry_2017.csv"   # name the builder expects
F="$PY scripts/00c_fetch_nih_cxr.py"
[ -f "$D/_index/hf_zip_index.csv" ] || $F index --out "$D/_index/hf_zip_index.csv"
if [ ! -f "$D/subset_plan.csv" ]; then
  if [ -f docs/data/nih_cxr_subset_plan.csv ]; then
    cp docs/data/nih_cxr_subset_plan.csv "$D/subset_plan.csv"        # the exact 40,880 images used in the report
  else
    $F plan --index "$D/_index/hf_zip_index.csv" --data-dir "$D" --n 40000 --seed 42 --out "$D/subset_plan.csv"
  fi
fi
$F fetch --index "$D/_index/hf_zip_index.csv" --plan "$D/subset_plan.csv" --out-dir "$D/images" --workers 24
$F verify --plan "$D/subset_plan.csv" --out-dir "$D/images" --sample 200
$F labels --data-dir "$D"
echo "NIH_CXR_DONE $(date): $(ls "$D/images" | wc -l | tr -d ' ') images under $D/images"
