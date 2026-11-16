"""Gated attention MIL head with class-specific attention (the learned read-out).

Ilse, Tomczak & Welling (2018) "Attention-based deep multiple instance learning",
extended to multi-label classification with one attention distribution per class.

Given frozen patch tokens z in R^{B x P x D} (P = G*G tokens, row-major grid):

    z_std   = (z - mu) / sigma                      fixed per-dim buffers (no LayerNorm)
    h       = tanh(U z_std) * sigmoid(V z_std)      shared gated projection, D -> H
    a       = W h                                    per-class attention logits, H -> C
    alpha_c = softmax_p(a[:, c])                     (B, C, P), sums to 1 over patches
    pooled_c = sum_p alpha[c, p] z_std[p]            (B, C, D)
    logit_c  = w_c . pooled_c + b_c                  per-class linear classifier, (C, D) weight

Parameter count ~ 2*D*H + H*C + C*D  (D=768, H=256, C=80  ->  ~0.48 M).

Entropy regulariser
-------------------
`attention_entropy(alpha)` is the Shannon entropy of alpha_c over patches divided by
log(P), so it lies in [0, 1]: 1 = uniform attention, 0 = one-hot attention.
`mil_loss` adds `lam * mean(entropy over *present* classes only)` to the BCE loss; only
classes with y = 1 are regularised, absent classes are left unconstrained.

* lam > 0 pushes attention to be peaky and discourages the uniform solution.
* lam = 0 is the plain head.
* lam -> -inf *rewards* entropy and forces alpha to be uniform; with uniform alpha
  pooled_c = mean_p z_p for every c, so the head collapses to a linear probe on the
  mean-pooled token, i.e. **mean pooling is nested inside this head as the lam -> -inf
  limit**. That makes "mean" a nested baseline of "attn".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionHead(nn.Module):
    """Class-specific gated attention pooling + per-class linear classifier.

    Inputs: tokens z of shape (B, P, D).  Outputs: logits (B, C) and alpha (B, C, P).
    """

    kind = "attn"

    def __init__(self, dim: int = 768, n_classes: int = 80, hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.dim, self.n_classes, self.hidden, self.dropout_p = dim, n_classes, hidden, dropout
        self.U = nn.Linear(dim, hidden)                 # tanh branch
        self.V = nn.Linear(dim, hidden)                 # sigmoid gate
        self.drop = nn.Dropout(dropout)
        self.W = nn.Linear(hidden, n_classes, bias=False)  # per-class attention logits
        self.cls_w = nn.Parameter(torch.empty(n_classes, dim))  # per-class linear classifier
        self.cls_b = nn.Parameter(torch.zeros(n_classes))
        nn.init.normal_(self.cls_w, std=1.0 / math.sqrt(dim))
        # Fixed input standardisation estimated from a token subsample at fit time.
        self.register_buffer("in_mean", torch.zeros(dim))
        self.register_buffer("in_std", torch.ones(dim))

    # ----------------------------------------------------------------- config
    @property
    def config(self) -> dict:
        return {"kind": self.kind, "dim": self.dim, "n_classes": self.n_classes,
                "hidden": self.hidden, "dropout": self.dropout_p}

    @torch.no_grad()
    def set_input_stats(self, mean, std) -> None:
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(self.dim)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(self.dim)
        self.in_mean.copy_(mean.to(self.in_mean.device))
        self.in_std.copy_(std.clamp_min(1e-6).to(self.in_std.device))

    # ---------------------------------------------------------------- forward
    def standardise(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.in_mean) / self.in_std

    def tokens(self, z: torch.Tensor) -> torch.Tensor:
        """Token set the attention runs over (hook for the multi-scale variant)."""
        return self.standardise(z)

    def attend(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, P', D) standardised tokens -> alpha (B, C, P')."""
        h = torch.tanh(self.U(t)) * torch.sigmoid(self.V(t))   # (B, P', H)
        h = self.drop(h)
        a = self.W(h)                                           # (B, P', C)
        return torch.softmax(a.transpose(1, 2), dim=-1)         # (B, C, P')

    def classify(self, alpha: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pooled = torch.bmm(alpha, t)                            # (B, C, D)
        return torch.einsum("bcd,cd->bc", pooled, self.cls_w) + self.cls_b

    def forward(self, z: torch.Tensor):
        t = self.tokens(z.float())
        alpha = self.attend(t)
        logits = self.classify(alpha, t)
        return logits, alpha

    def fine_attention(self, alpha: torch.Tensor) -> torch.Tensor:
        """Attention projected onto the fine G x G grid (identity for the plain head)."""
        return alpha


class MultiScaleGatedAttentionHead(GatedAttentionHead):
    """Gated attention over a multi-scale token set.

    Besides the G x G patch tokens, the grid is average-pooled with kernel=stride=2 and
    kernel=stride=4 (G/2 x G/2 and G/4 x G/4 tokens).  A learned scale embedding (3 x D)
    is added and attention runs jointly over all G^2 + G^2/4 + G^2/16 tokens
    (16x16 + 8x8 + 4x4 = 336 for the 224 setting).  Outputs are as for the plain head;
    alpha is over the concatenated token set and `fine_attention` spreads coarse mass
    uniformly over the fine patches it covers so (B, C, G*G) maps are always available.
    """

    kind = "attn_ms"
    factors = (1, 2, 4)

    def __init__(self, dim: int = 768, n_classes: int = 80, hidden: int = 256,
                 dropout: float = 0.1, grid: int = 16):
        super().__init__(dim, n_classes, hidden, dropout)
        if grid % 4 != 0:
            raise ValueError(f"grid must be divisible by 4 for the multi-scale head, got {grid}")
        self.grid = grid
        self.scale_emb = nn.Parameter(torch.zeros(len(self.factors), dim))
        nn.init.normal_(self.scale_emb, std=0.02)

    @property
    def config(self) -> dict:
        return {**super().config, "grid": self.grid}

    @property
    def n_tokens_per_scale(self) -> list[int]:
        return [(self.grid // f) ** 2 for f in self.factors]

    def tokens(self, z: torch.Tensor) -> torch.Tensor:
        B, P, D = z.shape
        G = self.grid
        if P != G * G:
            raise ValueError(f"expected {G * G} tokens for grid {G}, got {P}")
        t = self.standardise(z)
        img = t.transpose(1, 2).reshape(B, D, G, G)             # (B, D, G, G)
        parts = []
        for k, f in enumerate(self.factors):
            if f == 1:
                s = t
            else:
                s = F.avg_pool2d(img, kernel_size=f, stride=f).flatten(2).transpose(1, 2)
            parts.append(s + self.scale_emb[k])
        return torch.cat(parts, dim=1)                           # (B, sum_s (G/f)^2, D)

    def fine_attention(self, alpha: torch.Tensor) -> torch.Tensor:
        """(B, C, P_total) -> (B, C, G*G): coarse cells spread their mass over children."""
        B, C, _ = alpha.shape
        G = self.grid
        out = torch.zeros(B, C, G, G, dtype=alpha.dtype, device=alpha.device)
        start = 0
        for f, n in zip(self.factors, self.n_tokens_per_scale):
            g = G // f
            a = alpha[:, :, start:start + n].reshape(B, C, g, g)
            if f > 1:
                a = a.repeat_interleave(f, dim=2).repeat_interleave(f, dim=3) / float(f * f)
            out = out + a
            start += n
        return out.reshape(B, C, G * G)


# ------------------------------------------------------------------ losses etc.
def attention_entropy(alpha: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalised Shannon entropy of alpha over its last axis: (B, C, P) -> (B, C) in [0, 1]."""
    P = alpha.shape[-1]
    ent = -(alpha * torch.log(alpha.clamp_min(eps))).sum(-1)
    return ent / math.log(P) if P > 1 else torch.zeros_like(ent)


def mil_loss(logits: torch.Tensor, alpha: torch.Tensor, y: torch.Tensor, lam: float = 0.0):
    """BCE-with-logits + lam * mean normalised attention entropy over present classes.

    Returns (loss, bce, mean_present_entropy).  The entropy term is averaged over the
    (image, class) pairs with y = 1 only; with no positives in the batch it is zero.
    """
    y = y.float()
    bce = F.binary_cross_entropy_with_logits(logits, y)
    ent = attention_entropy(alpha)                               # (B, C)
    n_pos = y.sum()
    ent_present = (ent * y).sum() / n_pos if n_pos > 0 else ent.sum() * 0.0
    loss = bce + lam * ent_present if lam != 0.0 else bce
    return loss, bce, ent_present


def build_head(kind: str = "attn", dim: int = 768, n_classes: int = 80, hidden: int = 256,
               dropout: float = 0.1, grid: int = 16) -> GatedAttentionHead:
    if kind == "attn":
        return GatedAttentionHead(dim, n_classes, hidden, dropout)
    if kind == "attn_ms":
        return MultiScaleGatedAttentionHead(dim, n_classes, hidden, dropout, grid)
    raise ValueError(f"unknown head kind {kind!r} (expected 'attn' or 'attn_ms')")


def n_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
