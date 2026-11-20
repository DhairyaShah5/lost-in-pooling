import numpy as np
import torch

from lip.eval.deploy_check import check, curve, fit_curve, predict
from lip.models.attention_mil import build_head
from lip.models.distill import tiled_scores


class _Store:
    def __init__(self, n, grid, dim, seed=0):
        rng = np.random.default_rng(seed)
        self.grid, self.dim = grid, dim
        self.patch = rng.standard_normal((n, grid * grid, dim)).astype(np.float16)
        self.image_ids = np.arange(n)

    def __len__(self):
        return len(self.image_ids)

    def iter_patch_batches(self, bs):
        for s in range(0, len(self), bs):
            yield slice(s, s + bs), self.patch[s:s + bs]


def test_tiled_scores_matches_direct_quadrant_application():
    torch.manual_seed(0)
    head = build_head("attn", dim=6, n_classes=3, hidden=4, dropout=0.0, grid=4).eval()
    hi = _Store(3, 8, 6)
    T = tiled_scores(head, hi, tiles=2, batch_size=2, device="cpu")
    z = torch.tensor(hi.patch).float().view(3, 8, 8, 6)
    quads = [z[:, :4, :4], z[:, :4, 4:], z[:, 4:, :4], z[:, 4:, 4:]]
    with torch.no_grad():
        direct = torch.stack([torch.sigmoid(head(q.reshape(3, 16, 6))[0]) for q in quads], 1).max(1).values.numpy()
    assert T.shape == (3, 3)
    assert np.allclose(T, direct, atol=1e-5)


def test_fit_curve_recovers_synthetic_logistic():
    rng = np.random.default_rng(0)
    x = np.linspace(-4, -0.5, 60)
    true = (0.7, 1.5, -2.5)
    y = curve(true, x) + rng.normal(0, 0.005, size=x.shape)
    f = fit_curve(x, y)
    assert abs(f["a"] - true[0]) < 0.05 and abs(f["m"] - true[2]) < 0.2
    assert abs(predict(f, 10 ** -2.5) - curve(true, -2.5)) < 0.02
    fits = {"g_coco": f, "o_patches": {"a": 0.0, "k": 1.4, "m": -3.0}}
    r = check(16, 0.002, fits)
    assert 0 < r["patches_covered"] < 1 and r["region_auroc"] > r["global_auroc_with_scene_prior"]
