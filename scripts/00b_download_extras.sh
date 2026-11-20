#!/bin/sh
# Fetch the phase-7 datasets: ChestX-Det10 annotations + images, LUNA16 annotations, lung masks and two subsets.
# Everything lands under data/ (gitignored). Sizes: Det10 1.4 GB, LUNA16 subsets ~6.8 GB each.
set -e
D=data/chestx_det10/raw; mkdir -p "$D"
for f in train.json test.json README.md; do
  [ -f "$D/$f" ] || curl -sL -o "$D/$f" "https://raw.githubusercontent.com/Deepwise-AILab/ChestX-Det10-Dataset/master/$f"
done
for f in test_data train_data; do
  [ -f "$D/$f.zip" ] || curl -L -C - -o "$D/$f.zip" "http://resource.deepwise.com/xraychallenge/$f.zip"
done
mkdir -p data/chestx_det10/images_raw && (cd data/chestx_det10/images_raw && unzip -q -o ../raw/test_data.zip && unzip -q -o ../raw/train_data.zip)
L=data/luna16/raw; mkdir -p "$L"
[ -f "$L/annotations.csv" ] || curl -sL -o "$L/annotations.csv" "https://zenodo.org/api/records/3723295/files/annotations.csv/content"
[ -f "$L/seg-lungs-LUNA16.zip" ] || curl -L -o "$L/seg-lungs-LUNA16.zip" "https://zenodo.org/api/records/3723295/files/seg-lungs-LUNA16.zip/content"
for s in subset0 subset1; do
  curl -L -C - -o "$L/$s.zip" "https://zenodo.org/api/records/3723295/files/$s.zip/content"
done
echo "extras downloaded"
