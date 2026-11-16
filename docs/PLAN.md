# Plan (midterm sprint, Nov 15 2026)

## Decisions and why

* **One dataset, MS-COCO 2017.** Every image has boxes, so every image contributes to the
  size curve. Test = all 5,000 val2017 images. Train = seeded 10,000-image subset of train2017
  (9,000 train / 1,000 val for model selection). Chest X-ray sets with boxes are either tiny or
  42 GB; LUNA16 is 3D CT and is dropped.
* **One backbone, DINOv2 ViT-B/14 with registers, frozen, input 224 squash-resized.**
  Registers give cleaner patch tokens. Squash resize keeps every box inside the 16x16 grid.
* **Size variable.** Relative bbox area of the largest instance of the class in the image,
  six log-spaced bins from <0.1% to >10%. One patch at 224 covers 0.39%, so the first two
  bins are sub-patch.
* **Metric.** Per class and bin AUROC (positives in the bin vs all negatives), macro over
  classes with >= 10 positives, bootstrap CI over images. Matches the chest-X-ray paper.
* **Read-outs on identical cached tokens.** CLS, mean, CLS+mean, max, max-MIL, oracle
  region pooling (matched random boxes for negatives), gated attention MIL head with
  entropy sweep. Localisation audit of the head: attention mass in GT boxes vs chance,
  pointing game.
* **Causal framing.** The controlled same-token swap of the aggregation operator is the
  evidence; cross-modal similarity would only show generality. Negative findings are
  stated specifically (dataset, backbone, resolution, bins).
* **Context caveat.** COCO co-occurrence lets a global embedding score tiny objects above
  chance without seeing them. The oracle gap and the localisation audit separate the two.
* **Deferred** (per pre-proposal feedback): audio, distillation, additional backbones,
  chest X-ray replication. Resolution ablation at 448 is a go/no-go around 19:00.

## Pipeline

```
scripts/00_download.py        annotations + val2017 + 10k train subset
scripts/01_build_manifest.py  manifest.parquet, labels.npz, boxes.parquet
scripts/02_extract.py         cached fp16 CLS + patch tokens per split
scripts/03_probes.py          cls / mean / cls_mean / max / maxmil read-outs
scripts/04_oracle.py          oracle region read-out (upper bound)
scripts/05_train_head.py      gated attention head (+ entropy sweep, localisation)
scripts/07_plots.py           blindness curves + LaTeX table
scripts/08_attention_figure.py attention examples figure
report/build.sh               midterm.pdf
```

## Schedule (PT)

| window        | critical path                                   | parallel agents                         |
|---------------|-------------------------------------------------|-----------------------------------------|
| 16:20-16:50   | env, repo, scaffold, downloads, extractor       | eval, read-outs, head, report launched  |
| 16:50-17:45   | extraction (test, val, train)                   | agents deliver code + tests             |
| 17:45-19:00   | baselines + oracle, curve v1                    | report skeleton compiles                |
| 19:00-20:15   | attention head, localisation, curve v2          | results-to-LaTeX                        |
| 20:15-22:15   | write report, compile                           | lit-review polish                       |
| 22:15-23:00   | proofread, final PDF, push                      |                                         |
