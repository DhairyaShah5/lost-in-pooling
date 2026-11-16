"""Synthetic tests for lip.eval (bins, metrics, results, plots). No real data needed."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from lip.eval import bins as B
from lip.eval import metrics as M
from lip.eval import plots as P
from lip.eval import results as R

REPO = Path(__file__).resolve().parents[1]
N_IMAGES, N_CLASSES = 600, 80


# --------------------------------------------------------------------------------------
# Synthetic data: signal strength grows with object size, so the curve is monotone.
# --------------------------------------------------------------------------------------
def make_synthetic(n: int = N_IMAGES, n_classes: int = N_CLASSES, seed: int = 0,
                   prevalence: float = 0.2, strength: float = 3.0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, n_classes)) < prevalence).astype(np.uint8)
    frac = 10.0 ** rng.uniform(-4, 0, size=(n, n_classes))  # log-uniform in [1e-4, 1]
    max_box_frac = np.where(y == 1, frac, 0.0).astype(np.float32)
    shift = strength * (np.log10(frac) + 4.0) / 4.0  # 0 for 1e-4 ... strength for 1.0
    scores = rng.normal(size=(n, n_classes)) + np.where(y == 1, shift, 0.0)
    labels = {
        "image_ids": np.arange(n, dtype=np.int64) + 1000,
        "split": np.array(["test"] * n),
        "y": y,
        "max_box_frac": max_box_frac,
        "max_seg_frac": (max_box_frac * 0.7).astype(np.float32),
        "cat_ids": np.arange(n_classes, dtype=np.int64) + 1,
        "cat_names": np.array([f"cat{i:02d}" for i in range(n_classes)]),
    }
    width = rng.integers(400, 640, size=n)
    height = rng.integers(300, 480, size=n)
    return scores.astype(np.float32), labels, width, height


@pytest.fixture(scope="module")
def synth():
    return make_synthetic()


@pytest.fixture(scope="module")
def pcb(synth):
    scores, labels, _, _ = synth
    return M.per_class_bin_auroc(scores, labels["y"], labels["max_box_frac"], min_pos=10)


# --------------------------------------------------------------------------------------
# bins
# --------------------------------------------------------------------------------------
def test_bin_constants():
    assert B.BIN_EDGES == [0, 0.001, 0.003, 0.01, 0.03, 0.10, 1.0]
    assert B.BIN_NAMES == ["<0.1%", "0.1-0.3%", "0.3-1%", "1-3%", "3-10%", ">10%"]
    assert B.N_BINS == 6


def test_assign_bins_edges():
    frac = np.array([0.0, 1e-5, 0.0005, 0.001, 0.0029, 0.003, 0.0099, 0.01, 0.03, 0.0999, 0.1, 0.5, 1.0])
    want = np.array([-1, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 5], dtype=np.int8)
    got = B.assign_bins(frac)
    assert got.dtype == np.int8
    np.testing.assert_array_equal(got, want)
    # float32 inputs that equal an edge land in the bin starting at that edge
    got32 = B.assign_bins(frac.astype(np.float32))
    np.testing.assert_array_equal(got32, want)
    # shape is preserved for 2-D inputs; zeros are -1
    two_d = B.assign_bins(np.array([[0.0, 0.02], [1.0, 0.004]], dtype=np.float32))
    np.testing.assert_array_equal(two_d, np.array([[-1, 3], [5, 2]], dtype=np.int8))


def test_coco_sml():
    frac = np.array([[0.0, 0.05, 0.5], [0.001, 0.005, 0.5], [0.1024, 0.9216, 1.0]])
    width = np.array([100, 1000, 100])
    height = np.array([100, 1000, 100])
    got = B.coco_sml(frac, width, height)
    # areas: row0 = 0, 500, 5000; row1 = 1000, 5000, 5e5; row2 = 1024 (medium), 9216 (large), 1e4
    want = np.array([[-1, 0, 1], [0, 1, 2], [1, 2, 2]], dtype=np.int8)
    assert got.dtype == np.int8
    np.testing.assert_array_equal(got, want)


def test_patch_area_frac():
    assert B.patch_area_frac(224) == pytest.approx(1 / 256)
    assert B.patch_area_frac(448) == pytest.approx(1 / 1024)
    assert B.patch_area_frac(224, patch=16) == pytest.approx(1 / 196)


def test_box_to_patch_mask_sub_patch():
    g = 16
    m = B.box_to_patch_mask(np.array([0.10, 0.10, 0.12, 0.12]), g)  # no centre inside
    assert m.shape == (g * g,) and m.dtype == bool
    assert m.sum() == 1
    assert m[1 * g + 1]  # box centre (0.11, 0.11) is in patch (row 1, col 1)
    # degenerate (zero-area) box still marks the patch containing its centre
    m0 = B.box_to_patch_mask(np.array([0.5, 0.5, 0.5, 0.5]), 4)
    assert m0.sum() == 1 and m0[2 * 4 + 2]
    # centre on the far edge is clipped into the grid
    m1 = B.box_to_patch_mask(np.array([0.999, 0.999, 1.0, 1.0]), 4)
    assert m1.sum() == 1 and m1[3 * 4 + 3]


def test_box_to_patch_mask_whole_grid():
    g = 8
    m = B.box_to_patch_mask(np.array([0.0, 0.0, 1.0, 1.0]), g)
    assert m.all()


def test_box_to_patch_mask_straddling():
    g = 4  # centres at 0.125, 0.375, 0.625, 0.875
    m = B.box_to_patch_mask(np.array([0.3, 0.3, 0.7, 0.7]), g)
    want = np.zeros(16, dtype=bool)
    want[[5, 6, 9, 10]] = True
    np.testing.assert_array_equal(m, want)
    # right-open: a box ending exactly on a centre excludes it
    m2 = B.box_to_patch_mask(np.array([0.0, 0.0, 0.375, 0.375]), g)
    want2 = np.zeros(16, dtype=bool)
    want2[0] = True
    np.testing.assert_array_equal(m2, want2)
    # top-left quadrant
    m3 = B.box_to_patch_mask(np.array([0.0, 0.0, 0.5, 0.5]), g)
    assert set(np.flatnonzero(m3)) == {0, 1, 4, 5}


def test_boxes_to_patch_mask_union_and_empty():
    g = 4
    boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])
    m = B.boxes_to_patch_mask(boxes, g)
    assert set(np.flatnonzero(m)) == {0, 1, 4, 5, 10, 11, 14, 15}
    assert not B.boxes_to_patch_mask(np.zeros((0, 4)), g).any()


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def test_auroc_rank_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    labels = rng.random(300) < 0.3
    scores = np.round(rng.normal(size=300) + labels, 1)  # many ties
    assert M.auroc_rank(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
    assert np.isnan(M.auroc_rank(scores, np.zeros(300, bool)))
    assert M.auroc_rank([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert M.auroc_rank([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5


def test_per_class_bin_auroc_structure_and_sklearn(synth, pcb):
    scores, labels, _, _ = synth
    y, mbf = labels["y"], labels["max_box_frac"]
    bins = B.assign_bins(mbf)
    assert set(pcb) == set(range(N_CLASSES))
    assert set(pcb[0]["bins"]) == set(range(6))
    c = 3
    neg = y[:, c] == 0
    for b in range(6):
        sel = (y[:, c] == 1) & (bins[:, c] == b)
        cell = pcb[c]["bins"][b]
        assert cell["n_pos"] == int(sel.sum())
        if cell["n_pos"] >= 10:
            yy = np.r_[np.ones(sel.sum()), np.zeros(neg.sum())]
            ss = np.r_[scores[sel, c], scores[neg, c]]
            assert cell["auroc"] == pytest.approx(roc_auc_score(yy, ss), abs=1e-9)
        else:
            assert np.isnan(cell["auroc"])
    assert pcb[c]["n_neg"] == int(neg.sum())
    assert pcb[c]["n_pos_all"] == int((y[:, c] == 1).sum())
    assert pcb[c]["auroc_all"] == pytest.approx(roc_auc_score(y[:, c], scores[:, c]), abs=1e-9)
    # at least one cell is starved of positives and reports nan
    assert any(np.isnan(pcb[k]["bins"][b]["auroc"]) for k in pcb for b in range(6))


def test_weighted_auroc_with_unit_weights_equals_point_estimate(synth):
    scores, labels, _, _ = synth
    y, mbf = labels["y"], labels["max_box_frac"]
    bins = B.assign_bins(mbf)
    ref = M.per_class_auroc_from_bins(scores, y, bins, 6, min_pos=1)
    w = np.ones((1, N_IMAGES))
    for c in (0, 17, 42):
        pos_bins = np.where(y[:, c] == 1, bins[:, c], -1).astype(np.int64)
        au, wp = M._weighted_auroc_by_bin(scores[:, c].astype(np.float64), w, y[:, c] == 0, pos_bins, 6)
        for b in range(6):
            assert wp[0, b] == ref[c]["bins"][b]["n_pos"]
            if ref[c]["bins"][b]["n_pos"] > 0:
                assert au[0, b] == pytest.approx(ref[c]["bins"][b]["auroc"], abs=1e-12)
            else:
                assert np.isnan(au[0, b])
    # ties: identical scores give 0.5 under both implementations
    s_tie = np.zeros(N_IMAGES)
    pos_bins = np.where(y[:, 0] == 1, bins[:, 0], -1).astype(np.int64)
    au, _ = M._weighted_auroc_by_bin(s_tie, w, y[:, 0] == 0, pos_bins, 6)
    assert np.allclose(au[0][np.isfinite(au[0])], 0.5)


def test_macro_curve_monotone(pcb):
    curve = M.macro_curve(pcb, min_pos=10)
    assert set(curve) == set(range(6))
    vals = np.array([curve[b]["auroc_macro"] for b in range(6)])
    n_cls = np.array([curve[b]["n_classes"] for b in range(6)])
    assert (n_cls > 0).all()
    assert np.isfinite(vals).all()
    assert (np.diff(vals) > 0).all(), vals  # planted signal grows with size
    assert vals[0] < 0.7 and vals[-1] > 0.9
    assert all(curve[b]["n_pos"] >= 10 * curve[b]["n_classes"] for b in range(6))


def test_bootstrap_ci_contains_point_and_is_deterministic(synth, pcb):
    scores, labels, _, _ = synth
    curve = M.macro_curve(pcb, min_pos=10)
    ci = M.bootstrap_macro_curve(scores, labels["y"], labels["max_box_frac"], n_boot=100, seed=0)
    ci2 = M.bootstrap_macro_curve(scores, labels["y"], labels["max_box_frac"], n_boot=100, seed=0)
    assert ci == ci2
    for b in range(6):
        lo, hi = ci[b]["ci_lo"], ci[b]["ci_hi"]
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= curve[b]["auroc_macro"] <= hi
        assert 0.0 < hi - lo < 0.2


def test_overall_metrics(synth):
    scores, labels, _, _ = synth
    o = M.overall_metrics(scores, labels["y"])
    assert 0.7 < o["auroc_macro"] <= 1.0
    assert 0.2 < o["map"] <= 1.0
    assert o["n_classes"] == N_CLASSES
    assert M.expected_chance == 0.5


# --------------------------------------------------------------------------------------
# results JSON
# --------------------------------------------------------------------------------------
def test_results_paths():
    assert R.results_path("cls", "dinov2_vitb14_reg", 224) == Path("results/cls__dinov2_vitb14_reg__res224.json")
    assert R.scores_path("attn", "dinov2_vitb14_reg", 448) == Path("results/scores/attn__dinov2_vitb14_reg__res448.npy")


def test_evaluate_readout_json_roundtrip(synth, tmp_path):
    scores, labels, width, height = synth
    d = R.evaluate_readout(scores, labels, readout="cls", backbone="bb", res=224, n_boot=50,
                           width=width, height=height, notes="synthetic")
    for k in ("readout", "backbone", "res", "split", "n_images", "train_split", "notes",
              "bins", "per_bin", "overall", "per_class_bin", "seed"):
        assert k in d
    assert d["n_images"] == N_IMAGES and d["split"] == "test" and d["train_split"] == "train"
    assert d["bins"] == {"edges": B.BIN_EDGES, "names": B.BIN_NAMES}
    assert list(d["per_bin"]) == B.BIN_NAMES
    for name in B.BIN_NAMES:
        cell = d["per_bin"][name]
        assert set(cell) == {"auroc_macro", "ci_lo", "ci_hi", "n_classes", "n_pos"}
        assert cell["ci_lo"] <= cell["auroc_macro"] <= cell["ci_hi"]
    assert list(d["per_class_bin"]) == [str(n) for n in labels["cat_names"]]
    assert list(d["per_class_bin"]["cat00"]) == B.BIN_NAMES
    assert set(d["per_class_bin"]["cat00"]["<0.1%"]) == {"auroc", "n_pos"}
    assert set(d["coco_sml"]) == {"small", "medium", "large"}
    assert 0.5 < d["overall"]["auroc_macro"] <= 1.0

    path = R.save_results(d, tmp_path / "cls__bb__res224.json")
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    back = json.loads(text)
    assert back["per_bin"] == d["per_bin"]
    assert back["overall"] == d["overall"]
    # nan cells became null and n_pos survived
    nulls = [cell["auroc"] for cls in back["per_class_bin"].values() for cell in cls.values()
             if cell["auroc"] is None]
    assert nulls, "expected some starved (class, bin) cells to be null"
    assert isinstance(back["per_class_bin"]["cat00"]["<0.1%"]["n_pos"], int)

    loaded = R.load_all_results(tmp_path)
    assert len(loaded) == 1 and loaded[0]["readout"] == "cls"
    sp = R.save_scores(scores, tmp_path / "scores" / "cls.npy")
    arr = np.load(sp)
    assert arr.dtype == np.float32 and arr.shape == (N_IMAGES, N_CLASSES)

    # without width/height no S/M/L block; n_boot=0 gives null CIs
    d2 = R.evaluate_readout(scores, labels, readout="mean", backbone="bb", res=224, n_boot=0)
    assert d2["coco_sml"] is None
    assert np.isnan(d2["per_bin"]["<0.1%"]["ci_lo"])


def test_evaluate_readout_rejects_split_mismatch(synth):
    scores, labels, _, _ = synth
    with pytest.raises(ValueError):
        R.evaluate_readout(scores, labels, readout="cls", backbone="bb", res=224, split="val", n_boot=0)


# --------------------------------------------------------------------------------------
# plots and tables
# --------------------------------------------------------------------------------------
def _three_results(tmp_dir: Path, backbone: str = "synthetic_bb", res: int = 224) -> list[dict]:
    out = []
    for readout, strength in (("oracle", 5.0), ("cls", 2.0), ("attn_ent0.01", 3.0), ("mean", 1.5)):
        scores, labels, w, h = make_synthetic(strength=strength)
        d = R.evaluate_readout(scores, labels, readout=readout, backbone=backbone, res=res,
                               n_boot=30, width=w, height=h)
        R.save_results(d, R.results_path(readout, backbone, res, tmp_dir))
        out.append(d)
    return out


def test_readout_order_and_styles():
    names = ["oracle", "attn_ent0.1", "zzz", "cls", "attn", "max", "mean", "maxmil", "cls_mean"]
    got = [r["readout"] for r in P.sort_results([{"readout": n} for n in names])]
    assert got == ["cls", "mean", "cls_mean", "max", "maxmil", "attn", "attn_ent0.1", "zzz", "oracle"]
    st = P.series_styles(got)
    assert st["oracle"]["color"] == "#000000"
    assert len({v["color"] for v in st.values()}) == len(got)
    assert len({v["marker"] for v in st.values()}) == len(got)
    assert P.one_patch_boundary(224) == 2  # 1/256 is nearest to the 0.3% edge
    assert P.one_patch_boundary(448) == 1  # 1/1024 is nearest to the 0.1% edge


def test_plot_and_table(tmp_path):
    results = _three_results(tmp_path)
    files = P.plot_blindness_curves(results, tmp_path / "fig" / "curve", res=224, title="synthetic")
    assert [f.suffix for f in files] == [".png", ".pdf"]
    assert all(f.exists() and f.stat().st_size > 1000 for f in files)

    tex = P.results_table_latex(results, caption="Synthetic.", label="tab:synthetic")
    for token in (r"\toprule", r"\midrule", r"\bottomrule", r"\caption{Synthetic.}", r"\label{tab:synthetic}", r"\textbf{"):
        assert token in tex
    rows = [ln for ln in tex.splitlines() if ln.endswith(r"\\") and not ln.startswith("read-out")]
    assert [r.split(" & ")[0] for r in rows] == ["cls", "mean", r"attn\_ent0.01", "oracle"]
    oracle_row = rows[-1]
    assert r"\textbf" not in oracle_row  # best is chosen among non-oracle rows only
    assert all(r"\textbf" in r for r in rows[:-1]) or r"\textbf" in rows[2]  # attn is best in some column
    txt = P.results_table_text(results)
    assert txt.splitlines()[0].startswith("read-out") and "all" in txt.splitlines()[0]
    assert "*" in txt


def test_script_07_end_to_end(tmp_path):
    _three_results(tmp_path / "results")
    cmd = [sys.executable, str(REPO / "scripts" / "07_plots.py"),
           "--results-dir", str(tmp_path / "results"),
           "--fig-dir", str(tmp_path / "results" / "figures"),
           "--report-fig-dir", str(tmp_path / "report" / "figures"),
           "--res", "224", "--backbone", "synthetic_bb"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    assert "read-out" in proc.stdout
    stem = "blindness_curve__synthetic_bb__res224"
    for d in (tmp_path / "results" / "figures", tmp_path / "report" / "figures"):
        assert (d / f"{stem}.png").exists() and (d / f"{stem}.pdf").exists()
    assert (tmp_path / "results" / "blindness_table__synthetic_bb__res224.tex").exists()
    # no matching results -> non-zero exit
    proc2 = subprocess.run(cmd[:-1] + ["other_bb"], capture_output=True, text=True, cwd=REPO)
    assert proc2.returncode == 1
