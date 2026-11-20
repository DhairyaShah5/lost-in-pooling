# TODO: midterm sprint, Nov 15 2026 (deadline 23:59 PT)

Legend: `[ ]` pending, `[~]` in progress, `[x]` done, `[-]` cut for today

## 0. Setup
- [x] 0.1 Python venv + dependencies (torch, torchvision, scikit-learn, ...)
- [x] 0.2 Repo scaffold: .gitignore, README, pyproject, docs/CONTRACTS.md, TODO.md
- [x] 0.3 Public GitHub repo `DhairyaShah5/lost-in-pooling` created, first push
- [x] 0.4 Download COCO 2017 annotations (241 MB)
- [x] 0.5 Download val2017 images (5,000 images, ~780 MB) = `test` split
- [x] 0.6 Download 10,000-image seeded subset of train2017 (~1.6 GB) = `train`/`val` splits

## 1. Data
- [x] 1.1 Manifest builder: images, splits (train 9k / val 1k / test 5k)
- [x] 1.2 Labels: presence (N,80), largest-instance box-area fraction (N,80), seg-area fraction
- [x] 1.3 Boxes table in normalized coordinates
- [x] 1.4 Dataset stats: positives per class per size bin (sanity table for the report)

## 2. Features
- [x] 2.1 DINOv2 ViT-B/14-reg loader on MPS; squash-resize 224; fp16 memmap writer
- [x] 2.2 Benchmark: ViT-B/14-reg runs at 38 img/s on MPS, kept
- [x] 2.3 Extract `test` (val2017) features (5,000 x 256 x 768 fp16, 1.8 GB)
- [x] 2.4 Extract `train` + `val` subset features (9,000 + 1,000, 3.7 GB)
- [x] 2.5 Extract 448-res features (test + val + 5k train) for the resolution ablation (started early, GPU was idle)

## 3. Evaluation harness
- [x] 3.1 Size bins (6 log-spaced) + COCO S/M/L mapping + box-to-patch mask
- [x] 3.2 Per-class per-bin AUROC, macro average, bootstrap CI, overall mAP/AUROC
- [x] 3.3 Results JSON writer following docs/CONTRACTS.md
- [x] 3.4 Blindness-curve plot (all read-outs on one figure) + per-bin table generator
- [x] 3.5 Unit tests on synthetic data (20 tests, bootstrap 3 s at N=5000)

## 4. Fixed read-outs (baselines)
- [x] 4.1 Pooling: CLS, mean, CLS+mean concat, per-dimension max
- [x] 4.2 Linear probe trainer (joint BCE, early stopping on `val` split)
- [x] 4.3 Max-MIL (per-patch linear scores, max over patches)
- [x] 4.4 Oracle region pooling (GT boxes; matched random boxes for negatives), per-class probes
- [x] 4.5 Run 4.1 to 4.4 on `test`, save results JSON + scores
- [x] 4.6 Blindness curve v1 figure + table committed (oracle 0.973 vs CLS 0.832 AUROC on sub-patch objects)

## 5. Learned head
- [x] 5.1 Gated attention MIL head (class-specific attention) over memmapped tokens
- [x] 5.2 Entropy regularizer; sweep lambda in {-0.1, 0, 0.01, 0.1}
- [x] 5.3 Localization metrics: attention mass inside GT boxes vs chance, pointing game, per bin
- [x] 5.4 Train + evaluate on `test`; results JSON (attn: sub-patch 0.899 vs mean 0.862, mAP 0.800 vs 0.734)
- [x] 5.5 Blindness curve v2 (with head) + localization figure
- [-] 5.6 Multi-scale attention variant (implemented, not run today)
- [x] 5.7 Resolution ablation probes at 448 (oracle +0.02 on sub-patch, global read-outs unchanged)

## 6. Report (`report/midterm.tex`, 5 to 6 pages, pre-proposal style)
- [x] 6.1 LaTeX skeleton in pre-proposal style, compiles with tectonic
- [x] 6.2 Bibliography with verified references
- [x] 6.3 Literature review draft
- [x] 6.4 Progress-so-far section (scope change after feedback, decisions, pipeline)
- [x] 6.5 Experimental setup section
- [x] 6.6 Initial results: figures, tables, discussion (context-prior caveat)
- [x] 6.7 Updated plan for the remaining three weeks
- [x] 6.8 Abstract + introduction refresh
- [x] 6.9 Final compile, proofread, PDF committed (8 pages incl. references, 0 placeholders)

## 7. Wrap-up
- [x] 7.1 README with reproduction commands
- [x] 7.2 Final push; verify GitHub shows all commits under DhairyaShah5

# Phase 2: expand the study (Nov 15, 17:30 to 23:00)

## 8. More backbones (same harness, same images)
- [x] 8.1 Backbone abstraction: torch.hub DINOv2 + Hugging Face CLIP/SigLIP, per-model normalisation, patch size in meta
- [x] 8.2 Extract CLIP ViT-B/16 (224, 14x14 grid) for all splits
- [x] 8.3 Extract SigLIP ViT-B/16 (224, 14x14 grid) for all splits
- [x] 8.4 Probes + oracle + attention head on CLIP (head 0.910 sub-patch, oracle 0.935)
- [x] 8.5 Probes + oracle + attention head on SigLIP (head 0.872, oracle 0.934)
- [x] 8.6 Cross-backbone figure (one panel per backbone) + table

## 9. Robustness
- [x] 9.1 Attention head seeds 1 and 2 on DINOv2 (0.899, 0.891, 0.889 sub-patch)
- [x] 9.2 Multi-scale attention head on DINOv2 (best read-out: 0.908 sub-patch, mAP 0.816)
- [x] 9.3 Max-MIL top-8 variant on DINOv2
- [x] 9.4 Paired bootstrap CI of per-bin AUROC differences (head vs mean, oracle vs mean)

## 10. Analysis
- [x] 10.1 Per-class sub-patch AUROC and oracle gap; context-rich vs context-poor classes
- [x] 10.2 Figure + table for the per-class analysis

## 11. Chest X-ray replication (stretch, strict time budget)
- [x] 11.1 Obtain NIH ChestX-ray14 labels, the 984 boxed images, and a training subset
- [x] 11.2 Manifest/labels/boxes in the same contract (findings as classes, box area fraction as size)
- [x] 11.3 Features (DINOv2 and RAD-DINO) + CLS/mean/oracle/head read-outs
- [x] 11.4 Chest X-ray blindness curve in the report

## 12. Report v2
- [x] 12.1 Cross-backbone results section
- [x] 12.2 Robustness and per-class subsections
- [x] 12.3 Abstract, introduction, progress, plan updated
- [x] 12.4 Final compile, proofread, push

# Phase 3: depth (Nov 15, 18:45 to 22:00)
- [x] 13.1 Chest X-rays at 448 px (DINOv2, RAD-DINO): oracle rises to 0.96-0.98, global probes unchanged (heads still running)
- [x] 13.2 Chest X-ray attention examples, per-finding table, paired bootstrap (oracle gap +0.27 to +0.31)
- [x] 13.3 Multi-scale head on CLIP and SigLIP (best read-out on every backbone)
- [x] 13.4 Mechanism test: AUROC vs patches covered at 224, 336, 448 px (global pooling tracks token fraction)
- [x] 13.5 Report v2: abstract, intro, feedback response, status, plan rewritten; literature trimmed
- [x] 13.6 Final pass: trim, full visual review, README results summary, push

# Phase 4 (Nov 15, 19:30 to 21:30)
- [x] 14.1 COCO at 336 px: third resolution point on the mechanism figure
- [x] 14.2 Seed repeats of the CLIP and SigLIP heads (CLIP 0.892 +- 0.016, SigLIP 0.875 +- 0.007)
- [x] 14.3 Chest X-ray heads at 448 px (DINOv2 done; 5k-train caveat noted)
- [x] 14.4 Final pass: full visual review, trims, README, push

# Phase 5 (Nov 15, 20:20 to 20:50)
- [x] 15.1 Multi-scale head seeds on DINOv2 (0.904 +- 0.004 sub-patch; lead over single-scale holds)
- [x] 15.2 Chest head seeds: DINOv2 gain holds (0.732 +- 0.016 vs 0.699); RAD-DINO gain does not (0.764 +- 0.015 vs 0.773), report corrected
- [x] 15.3 Limitations paragraph; final rebuild and push

# Phase 6 (Nov 15, 20:47 to 23:30): chest X-rays on the full 448 px cache
- [x] 16.1 Extract the full 20,721-image 448 px training cache for DINOv2 (31 min, 11 img/s)
- [x] 16.2 Attention head on the full 448 px cache, seed 0: 0.766 overall AUROC (0.727 at 224 px on the same images), mAP 0.375, attention mass 3.4x chance, pointing 0.50
- [x] 16.3 Seeds 1 and 2 of the full-cache 448 px head: 0.771 +- 0.005 overall (0.732 +- 0.016 at 224 px), mAP 0.381 +- 0.008
- [x] 16.4 Like-for-like CLS / mean / max probes on the full 448 px cache (0.711 / 0.710 / 0.663 vs head 0.766)
- [x] 16.5 Report (chest resolution paragraph, Table tab:cxr block, paired test, abstract, limitations, plan), README, push

# Phase 7 (Nov 19 onward): finish the plan from the midterm report
- [x] 17.1 RAD-DINO chest head on the full 448 px cache, three seeds, like-for-like probes (attn 0.800+-0.006 vs mean 0.808 overall; head leads in every size bin, +0.12 at 0.1-0.3%)
- [x] 17.2 Fourth backbone: DINO ViT-B/16 (self-supervised, 16 px patch) on COCO at 224 and 448 px: probes, oracle, heads x3 seeds, multi-scale head (oracle 0.900 -> 0.941 at 448 px, global unchanged)
- [x] 17.3 Seed repeats for every one-seed read-out (COCO oracles at 224/336/448, CLIP/SigLIP probes and oracles, chest probes, max-MIL); seeds__*.json summaries
- [x] 17.4 ChestX-Det10 boxes: fetch annotations, map to our NIH images, patient-disjoint manifest, CV read-outs and NIH-trained heads at 224/448 px (both backbones; transfer head 0.77 vs 0.70 DINOv2, 0.83 vs 0.80 RAD-DINO at 448)
- [x] 17.5 Token-fraction pre-deployment check: fits within 0.02 across resolutions, leave-one-backbone-out MAE <= 0.012, radiograph offset 0.17-0.28, 200 boxed images detect the gap; CLI `python -m lip.eval.deploy_check`
- [ ] 17.6 Tiled-teacher distillation: 448 px head as teacher, 224 px student on the same images (COCO and chest)
- [ ] 17.7 LUNA16 arm (scoped): one or two subsets, 2D nodule slices, blindness curve of CLS / mean / oracle / head by nodule size
- [x] 17.9 Context-only control (tokens outside the box): equals the mean probe on sub-patch targets (0.862 vs 0.862, DINOv2), on COCO x3 backbones + 448 px and on chest
- [ ] 17.8 Update the memory notes and README as results land

# Phase 8: deliverables
- [ ] 18.1 Final report (same format as the midterm), every number seeded, new sections for 17.2-17.7
- [ ] 18.2 Clinic slides (Nov 23): cross-backbone curves, oracle gap, mechanism figure, radiograph replication
- [ ] 18.3 Poster (Nov 30) and live demo script (curve + attention maps on a user-chosen image)
- [ ] 18.4 Final push: README reproduce commands for every arm, release tag
