"""Frozen backbones behind one interface: `tokens(x) -> (cls (B,D), patches (B,P,D))`.

Names
-----
dinov2_vitb14_reg / dinov2_vits14_reg : torch.hub DINOv2 with registers (patch 14, ImageNet norm).
    cls = x_norm_clstoken, patches = x_norm_patchtokens.
clip_vitb16   : Hugging Face `openai/clip-vit-base-patch16` vision tower (patch 16, CLIP norm).
    cls = pooler_output (post-LN CLS), patches = post_layernorm(last_hidden_state[:, 1:]).
siglip_vitb16 : Hugging Face `google/siglip-base-patch16-224` vision tower (patch 16, mean/std 0.5).
    SigLIP has no CLS token; its global vector is a learned multihead attention-pooling (MAP) head,
    so cls := pooler_output (MAP output) and patches = last_hidden_state (already post-LN).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

HF_IDS = {"clip_vitb16": "openai/clip-vit-base-patch16", "siglip_vitb16": "google/siglip-base-patch16-224"}


@dataclass
class Backbone:
    name: str
    model: torch.nn.Module
    embed_dim: int
    patch_size: int
    mean: tuple
    std: tuple
    global_token: str  # what `cls` means for this model
    _tokens: Callable = field(repr=False)

    @torch.no_grad()
    def tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._tokens(x)


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_backbone(name: str, device: torch.device | None = None) -> Backbone:
    device = device or get_device()
    if name.startswith("dinov2"):
        m = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True).eval().requires_grad_(False).to(device)

        def tok(x):
            f = m.forward_features(x)
            return f["x_norm_clstoken"], f["x_norm_patchtokens"]

        return Backbone(name, m, int(m.embed_dim), int(m.patch_size), IMAGENET_MEAN, IMAGENET_STD,
                        "x_norm_clstoken", tok)

    if name == "clip_vitb16":
        from transformers import CLIPVisionModel
        m = CLIPVisionModel.from_pretrained(HF_IDS[name]).eval().requires_grad_(False).to(device)
        ln = m.vision_model.post_layernorm

        def tok(x):
            out = m(pixel_values=x)
            return out.pooler_output, ln(out.last_hidden_state[:, 1:])

        return Backbone(name, m, int(m.config.hidden_size), int(m.config.patch_size), CLIP_MEAN, CLIP_STD,
                        "post_layernorm(CLS)", tok)

    if name == "siglip_vitb16":
        from transformers import SiglipVisionModel
        m = SiglipVisionModel.from_pretrained(HF_IDS[name]).eval().requires_grad_(False).to(device)

        def tok(x):
            out = m(pixel_values=x)
            return out.pooler_output, out.last_hidden_state

        return Backbone(name, m, int(m.config.hidden_size), int(m.config.patch_size), SIGLIP_MEAN, SIGLIP_STD,
                        "MAP attention-pooling head output (no CLS token)", tok)

    raise ValueError(f"unknown backbone {name!r}; known: dinov2_*, {list(HF_IDS)}")
