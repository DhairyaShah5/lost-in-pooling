# TODO: midterm sprint, Nov 15 2026 (deadline 23:59 PT)

Legend: `[ ]` pending, `[~]` in progress, `[x]` done, `[-]` cut for today

## 0. Setup
- [x] 0.1 Python venv + dependencies (torch, torchvision, scikit-learn, ...)
- [x] 0.2 Repo scaffold: .gitignore, README, pyproject, docs/CONTRACTS.md, TODO.md
- [x] 0.3 Public GitHub repo `DhairyaShah5/lost-in-pooling` created, first push
- [~] 0.4 Download COCO 2017 annotations (241 MB)
- [~] 0.5 Download val2017 images (5,000 images, ~780 MB) = `test` split
- [~] 0.6 Download 10,000-image seeded subset of train2017 (~1.6 GB) = `train`/`val` splits

## 1. Data
- [ ] 1.1 Manifest builder: images, splits (train 9k / val 1k / test 5k)
- [ ] 1.2 Labels: presence (N,80), largest-instance box-area fraction (N,80), seg-area fraction
- [ ] 1.3 Boxes table in normalized coordinates
- [ ] 1.4 Dataset stats: positives per class per size bin (sanity table for the report)

## 2. Features
- [ ] 2.1 DINOv2 ViT-B/14-reg loader on MPS; squash-resize 224; fp16 memmap writer
- [ ] 2.2 Benchmark 200 images; decide ViT-B vs ViT-S (threshold 20 img/s)
- [ ] 2.3 Extract `test` (val2017) features
- [ ] 2.4 Extract `train` + `val` subset features
- [ ] 2.5 Go/no-go at ~19:00: extract 448-res features (test + 5k train) for the resolution ablation

## 3. Evaluation harness
- [ ] 3.1 Size bins (6 log-spaced) + COCO S/M/L mapping + box-to-patch mask
- [ ] 3.2 Per-class per-bin AUROC, macro average, bootstrap CI, overall mAP/AUROC
- [ ] 3.3 Results JSON writer following docs/CONTRACTS.md
- [ ] 3.4 Blindness-curve plot (all read-outs on one figure) + per-bin table generator
- [ ] 3.5 Unit tests on synthetic data

## 4. Fixed read-outs (baselines)
- [ ] 4.1 Pooling: CLS, mean, CLS+mean concat, per-dimension max
- [ ] 4.2 Linear probe trainer (joint BCE, early stopping on `val` split)
- [ ] 4.3 Max-MIL (per-patch linear scores, max over patches)
- [ ] 4.4 Oracle region pooling (GT boxes; matched random boxes for negatives), per-class probes
- [ ] 4.5 Run 4.1 to 4.4 on `test`, save results JSON + scores
- [ ] 4.6 Blindness curve v1 figure + table committed

## 5. Learned head
- [ ] 5.1 Gated attention MIL head (class-specific attention) over memmapped tokens
- [ ] 5.2 Entropy regularizer; sweep lambda in {0, 0.01, 0.1}
- [ ] 5.3 Localization metrics: attention mass inside GT boxes vs chance, pointing game, per bin
- [ ] 5.4 Train + evaluate on `test`; results JSON
- [ ] 5.5 Blindness curve v2 (with head) + localization figure
- [ ] 5.6 Optional: multi-scale attention variant
- [ ] 5.7 Optional: resolution ablation probes at 448

## 6. Report (`report/midterm.tex`, 5 to 6 pages, pre-proposal style)
- [ ] 6.1 LaTeX skeleton in pre-proposal style, compiles with tectonic
- [ ] 6.2 Bibliography with verified references
- [ ] 6.3 Literature review draft
- [ ] 6.4 Progress-so-far section (scope change after feedback, decisions, pipeline)
- [ ] 6.5 Experimental setup section
- [ ] 6.6 Initial results: figures, tables, discussion (context-prior caveat)
- [ ] 6.7 Updated plan for the remaining three weeks
- [ ] 6.8 Abstract + introduction refresh
- [ ] 6.9 Final compile, proofread, PDF committed

## 7. Wrap-up
- [ ] 7.1 README with reproduction commands
- [ ] 7.2 Final push; verify GitHub shows all commits under DhairyaShah5
