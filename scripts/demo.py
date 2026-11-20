"""Live demo (poster session): run a frozen backbone on ONE image and show what global pooling and the
attention head see, with the blindness-curve numbers for the predicted classes.

    .venv/bin/python scripts/demo.py --image path/to/photo.jpg --out demo.png
    .venv/bin/python scripts/demo.py --image photo.jpg --classes "sports ball,person"

Left: the image. Middle: the top classes under the mean-pooling probe and the attention head. Right: the head's
attention map for the chosen class. The caption gives each class's sub-patch AUROC under mean pooling, the
head and the oracle from the cached COCO results, so the audience sees where the image sits on the curve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_or_load_probe(kind, backbone, res, ckpt_dir, features_root, labels_path, device):
    import torch
    from lip.data.labels import load_labels
    from lip.features.store import FeatureStore
    from lip.readouts.pooling import align_labels, cache_pooled
    from lip.readouts.probe import LinearProbe
    path = Path(ckpt_dir) / f"probe_{kind}__{backbone}__res{res}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    tr, va = FeatureStore(backbone, res, "train", root=features_root), FeatureStore(backbone, res, "val", root=features_root)
    ytr = align_labels(tr, load_labels(labels_path, split="train"))["y"]; yva = align_labels(va, load_labels(labels_path, split="val"))["y"]
    pr = LinearProbe().fit(cache_pooled(tr, kind), ytr, cache_pooled(va, kind), yva, epochs=40, lr=1e-3, weight_decay=1e-4, batch_size=512, device="cpu", seed=0)
    w = {"mean": pr.mean_, "std": pr.std_, "weight": pr.model.weight.detach().cpu().numpy(), "bias": pr.model.bias.detach().cpu().numpy()}
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(w, path)
    return w


def probe_scores(w, x):
    z = (x - w["mean"]) / w["std"]
    return 1 / (1 + np.exp(-(z @ w["weight"].T + w["bias"])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--backbone", default="dinov2_vitb14_reg")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--head", default=None, help="attention head checkpoint (default checkpoints/attn__BB__resR.pt)")
    p.add_argument("--classes", default="", help="comma-separated class names to show (default: the head's top class)")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--out", default="demo.png")
    p.add_argument("--features-root", default="features")
    p.add_argument("--labels", default="data/coco/labels.npz")
    p.add_argument("--results", default="results")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    import torch
    from PIL import Image
    from lip.features.backbones import get_device, load_backbone
    from lip.models.train_head import load_head, resolve_device

    device = get_device() if a.device == "auto" else torch.device(a.device)
    bb = load_backbone(a.backbone, device)
    im = Image.open(a.image).convert("RGB")
    x = np.asarray(im.resize((a.res, a.res), Image.BICUBIC), dtype=np.float32) / 255.0
    x = (x - np.array(bb.mean, dtype=np.float32)) / np.array(bb.std, dtype=np.float32)
    xt = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        cls, patches = bb.tokens(xt)
    cls, patches = cls.float().cpu().numpy(), patches.float().cpu().numpy()
    grid = int(np.sqrt(patches.shape[1]))

    if Path(a.labels).exists():
        names = [str(c) for c in np.load(a.labels, allow_pickle=False)["cat_names"]]
    else:  # raw data cleared: the 80 class names are committed in the same order
        names = json.loads((Path(__file__).resolve().parents[1] / "docs/data/coco_cat_names.json").read_text())["names"]
    head = load_head(a.head or Path(a.checkpoints) / f"attn__{a.backbone}__res{a.res}.pt", device=resolve_device(a.device))
    with torch.no_grad():
        logits, alpha = head(torch.from_numpy(patches).to(resolve_device(a.device)))
    p_head = torch.sigmoid(logits)[0].cpu().numpy(); alpha = alpha[0].cpu().numpy()      # (C,), (C, P)
    w_mean = fit_or_load_probe("mean", a.backbone, a.res, a.checkpoints, a.features_root, a.labels, device)
    p_mean = probe_scores(w_mean, patches[0].mean(0, keepdims=True))[0]
    w_cls = fit_or_load_probe("cls", a.backbone, a.res, a.checkpoints, a.features_root, a.labels, device)
    p_cls = probe_scores(w_cls, cls)[0]

    def small_bin(readout, cls_name):
        f = Path(a.results) / f"{readout}__{a.backbone}__res{a.res}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())["per_class_bin"].get(cls_name, {})
        v = d.get("<0.1%") or d.get("0.1-0.3%") or {}
        return v.get("auroc")

    chosen = [c.strip() for c in a.classes.split(",") if c.strip()] or [names[int(np.argmax(p_head))]]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2 + len(chosen), figsize=(4.2 * (2 + len(chosen)), 4.2))
    axes[0].imshow(im); axes[0].set_axis_off(); axes[0].set_title(Path(a.image).name, fontsize=9)
    top = np.argsort(-np.maximum(p_head, p_mean))[: a.top]
    yy = np.arange(len(top))
    axes[1].barh(yy - 0.2, p_cls[top], 0.2, label="CLS probe"); axes[1].barh(yy, p_mean[top], 0.2, label="mean probe"); axes[1].barh(yy + 0.2, p_head[top], 0.2, label="attention head")
    axes[1].set_yticks(yy); axes[1].set_yticklabels([names[i] for i in top], fontsize=8); axes[1].invert_yaxis(); axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("probability"); axes[1].legend(fontsize=7, loc="lower right"); axes[1].set_title("global read-outs vs head", fontsize=9)
    for k, cname in enumerate(chosen):
        ax = axes[2 + k]
        if cname not in names:
            ax.set_title(f"unknown class {cname!r}"); ax.set_axis_off(); continue
        c = names.index(cname)
        amap = alpha[c].reshape(grid, grid)
        ax.imshow(im.resize((a.res, a.res))); ax.imshow(np.kron(amap, np.ones((a.res // grid, a.res // grid))), cmap="inferno", alpha=0.55)
        ax.set_axis_off()
        sm, sh, so = small_bin("mean", cname), small_bin("attn", cname), small_bin("oracle", cname)
        sub = f"\nsub-patch AUROC on COCO: mean {sm:.2f}, head {sh:.2f}, oracle {so:.2f}" if None not in (sm, sh, so) else ""
        ax.set_title(f"{cname}: head {p_head[c]:.2f}, mean {p_mean[c]:.2f}, CLS {p_cls[c]:.2f}{sub}", fontsize=8)
    fig.tight_layout(); fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"[demo] {a.image}: top under head = {names[int(np.argmax(p_head))]} ({p_head.max():.2f}); under mean probe = {names[int(np.argmax(p_mean))]} ({p_mean.max():.2f}); figure -> {a.out}")


if __name__ == "__main__":
    main()
