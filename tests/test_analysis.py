"""Synthetic tests for the analysis tooling: paired bootstrap, per-class, seeds, cross-backbone.

No real data needed; everything runs in seconds on a few hundred synthetic images.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from lip.eval import compare as CMP
from lip.eval import perclass as PC
from lip.eval import plots as P
from lip.eval import results as R
from lip.eval import seeds as S
from lip.eval.bins import BIN_EDGES, BIN_NAMES, N_BINS, assign_bins

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


# --------------------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------------------
def make_labels(n: int, n_classes: int, seed: int = 0, prevalence: float = 0.25) -> dict:
    """Presence labels with log-uniform relative sizes in [1e-4, 1]."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n, n_classes)) < prevalence).astype(np.uint8)
    frac = 10.0 ** rng.uniform(-4, 0, size=(n, n_classes))
    max_box_frac = np.where(y == 1, frac, 0.0).astype(np.float32)
    return {
        "image_ids": np.arange(n, dtype=np.int64) + 1000,
        "split": np.array(["test"] * n),
        "y": y,
        "max_box_frac": max_box_frac,
        "max_seg_frac": (max_box_frac * 0.7).astype(np.float32),
        "cat_ids": np.arange(n_classes, dtype=np.int64) + 1,
        "cat_names": np.array([f"cat{i:02d}" for i in range(n_classes)]),
    }


def make_scores(labels: dict, strength: float, seed: int, base: np.ndarray | None = None,
                noise: float = 1.0) -> np.ndarray:
    """Gaussian scores whose positive shift grows with object size (``strength`` at full size).

    With ``base`` given, the new scores share its noise (``base + noise * fresh``
    plus the extra shift), which makes two read-outs correlated like real ones.
    """
    rng = np.random.default_rng(seed)
    y = labels["y"]
    frac = np.where(y == 1, labels["max_box_frac"], 1e-4).astype(np.float64)
    shift = strength * (np.log10(frac) + 4.0) / 4.0
    fresh = rng.normal(size=y.shape)
    if base is None:
        return (fresh + np.where(y == 1, shift, 0.0)).astype(np.float32)
    return (base + noise * fresh + np.where(y == 1, shift, 0.0)).astype(np.float32)


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@pytest.fixture(scope="module")
def labels():
    return make_labels(600, 80)


@pytest.fixture(scope="module")
def four_readouts(labels):
    """cls < mean < attn < oracle in signal strength; attn/oracle share mean's noise."""
    mean = make_scores(labels, strength=2.0, seed=1)
    return {
        "cls": make_scores(labels, strength=1.5, seed=2),
        "mean": mean,
        "attn": make_scores(labels, strength=0.8, seed=3, base=mean, noise=0.3),
        "oracle": make_scores(labels, strength=3.0, seed=4, base=mean, noise=0.3),
    }


def results_for(labels: dict, scores: dict[str, np.ndarray], backbone: str, n_boot: int = 20) -> list[dict]:
    return [R.evaluate_readout(s, labels, readout=name, backbone=backbone, res=224, n_boot=n_boot,
                               notes=f"synthetic {name}")
            for name, s in scores.items()]


# --------------------------------------------------------------------------------------
# Weighted AP and bootstrap draws
# --------------------------------------------------------------------------------------
def test_weighted_ap_matches_sklearn():
    rng = np.random.default_rng(0)
    n = 300
    s = np.round(rng.normal(size=n), 1)  # plenty of ties
    pos = rng.random(n) < 0.3
    s[pos] += 0.5
    w = rng.integers(0, 4, size=(5, n)).astype(np.float64)
    w[0] = 1.0
    ap = CMP.weighted_average_precision(s, w, pos)
    for b in range(5):
        idx = np.repeat(np.arange(n), w[b].astype(int))
        assert ap[b] == pytest.approx(average_precision_score(pos[idx], s[idx]), abs=1e-12)
    # no positive weight -> nan
    w0 = np.ones((1, n))
    w0[0, pos] = 0.0
    assert np.isnan(CMP.weighted_average_precision(s, w0, pos)[0])


def test_macro_draws_unit_weights_reproduce_point_estimates(labels):
    scores = make_scores(labels, strength=2.0, seed=5)
    y, mbf = labels["y"], labels["max_box_frac"]
    bins = assign_bins(mbf)
    draws = CMP.macro_draws(scores.astype(np.float64), y, bins, np.ones((1, y.shape[0])), min_pos=10)
    from lip.eval.metrics import macro_curve, overall_metrics, per_class_bin_auroc
    curve = macro_curve(per_class_bin_auroc(scores, y, mbf, min_pos=10), min_pos=10)
    for b in range(N_BINS):
        assert draws["per_bin"][0, b] == pytest.approx(curve[b]["auroc_macro"], abs=1e-9)
    ov = overall_metrics(scores, y)
    assert draws["auroc_macro"][0] == pytest.approx(ov["auroc_macro"], abs=1e-9)
    assert draws["map"][0] == pytest.approx(ov["map"], abs=1e-9)


def test_summarise_difference_p_value():
    draws = np.concatenate([np.full(95, 0.1), np.full(5, -0.1)])
    out = CMP.summarise_difference(0.1, draws)
    assert out["p_two_sided"] == pytest.approx(0.10)
    assert out["ci_lo"] <= 0.1 <= out["ci_hi"]
    assert CMP.summarise_difference(-0.1, -draws)["p_two_sided"] == pytest.approx(0.10)
    assert CMP.summarise_difference(0.0, draws)["p_two_sided"] == 1.0
    assert CMP.summarise_difference(0.3, draws)["p_two_sided"] == pytest.approx(0.10)  # same sign counting
    assert CMP.summarise_difference(0.1, np.full(10, 0.2))["p_two_sided"] == 0.0
    assert math.isnan(CMP.summarise_difference(float("nan"), draws)["ci_lo"])


# --------------------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------------------
def test_paired_bootstrap_ci_contains_true_difference():
    """Per-bin constant shifts give closed-form AUROCs: Phi(shift / sqrt 2)."""
    rng = np.random.default_rng(11)
    n, n_classes = 3000, 10
    lab = make_labels(n, n_classes, seed=11, prevalence=0.3)
    y, mbf = lab["y"], lab["max_box_frac"]
    bins = assign_bins(mbf)
    shift_a = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
    extra = 0.5
    noise = rng.normal(size=y.shape)
    pos = y == 1
    sa = noise + np.where(pos, shift_a[np.clip(bins, 0, None)], 0.0)
    sb = sa + np.where(pos, extra, 0.0)
    true_delta = np.array([phi(s / math.sqrt(2)) - phi((s + extra) / math.sqrt(2)) for s in shift_a])
    out = CMP.paired_bootstrap_difference(sa, sb, y, mbf, n_boot=200, seed=0, min_pos=10)
    assert set(out["per_bin"]) == set(BIN_NAMES)
    covered = 0
    for b, name in enumerate(BIN_NAMES):
        cell = out["per_bin"][name]
        assert set(cell) >= {"delta", "ci_lo", "ci_hi", "p_two_sided", "auroc_a", "auroc_b", "n_classes", "n_pos"}
        assert cell["ci_lo"] <= cell["delta"] <= cell["ci_hi"]
        assert abs(cell["delta"] - true_delta[b]) < 0.03
        covered += cell["ci_lo"] <= true_delta[b] <= cell["ci_hi"]
        assert cell["delta"] < 0 and cell["p_two_sided"] == 0.0  # B is strictly better
    assert covered >= 5  # 95% intervals, 6 bins
    # pooled truth: bins are hit with probability proportional to their log width
    widths = np.diff(np.log10(np.array(BIN_EDGES[1:], dtype=np.float64)), prepend=-4.0)
    p_bin = widths / widths.sum()
    true_all = float((p_bin * true_delta).sum())
    ov = out["overall"]["auroc_macro"]
    assert ov["ci_lo"] <= true_all <= ov["ci_hi"]
    assert out["overall"]["map"]["delta"] < 0 and out["overall"]["map"]["ci_hi"] < 0
    assert out["paired"] is True and out["n_boot"] == 200


def test_paired_ci_narrower_than_unpaired_when_correlated(labels):
    y, mbf = labels["y"], labels["max_box_frac"]
    a = make_scores(labels, strength=2.0, seed=21)
    b = make_scores(labels, strength=0.3, seed=22, base=a, noise=0.2)  # strongly correlated with a
    paired = CMP.paired_bootstrap_difference(a, b, y, mbf, n_boot=150, seed=0)
    unpaired = CMP.paired_bootstrap_difference(a, b, y, mbf, n_boot=150, seed=0, paired=False)
    assert unpaired["paired"] is False
    for name in BIN_NAMES:
        wp = paired["per_bin"][name]["ci_hi"] - paired["per_bin"][name]["ci_lo"]
        wu = unpaired["per_bin"][name]["ci_hi"] - unpaired["per_bin"][name]["ci_lo"]
        assert wp < wu
        # the point estimate does not depend on the pairing
        assert paired["per_bin"][name]["delta"] == pytest.approx(unpaired["per_bin"][name]["delta"])
    for key in ("auroc_macro", "map"):
        wp = paired["overall"][key]["ci_hi"] - paired["overall"][key]["ci_lo"]
        wu = unpaired["overall"][key]["ci_hi"] - unpaired["overall"][key]["ci_lo"]
        assert wp < wu


def test_paired_bootstrap_identical_scores_and_determinism(labels):
    y, mbf = labels["y"], labels["max_box_frac"]
    a = make_scores(labels, strength=2.0, seed=31)
    same = CMP.paired_bootstrap_difference(a, a.copy(), y, mbf, n_boot=50, seed=3)
    for name in BIN_NAMES:
        cell = same["per_bin"][name]
        assert cell["delta"] == 0.0 and cell["ci_lo"] == 0.0 and cell["ci_hi"] == 0.0
        assert cell["p_two_sided"] == 1.0
    b = make_scores(labels, strength=1.0, seed=32)
    r1 = CMP.paired_bootstrap_difference(a, b, y, mbf, n_boot=50, seed=3)
    r2 = CMP.paired_bootstrap_difference(a, b, y, mbf, n_boot=50, seed=3)
    assert r1 == r2
    lines = CMP.format_compare_lines(r1, "a", "b")
    assert len(lines) == N_BINS + 2 and lines[0].startswith(f"{BIN_NAMES[0]:>10}")
    with pytest.raises(ValueError):
        CMP.paired_bootstrap_difference(a, b[:-1], y, mbf, n_boot=5)


# --------------------------------------------------------------------------------------
# Per-class analysis
# --------------------------------------------------------------------------------------
def test_small_target_auroc_matches_json_pooling(labels, four_readouts):
    y, mbf = labels["y"], labels["max_box_frac"]
    cat_names = [str(c) for c in labels["cat_names"]]
    scores = four_readouts["mean"]
    auroc, n_small = PC.small_target_auroc(scores, y, mbf, min_pos=10)
    bins = assign_bins(mbf)
    np.testing.assert_array_equal(n_small, ((y == 1) & (bins <= 1)).sum(axis=0))
    assert np.isnan(auroc[n_small < 10]).all()
    res = R.evaluate_readout(scores, labels, readout="mean", backbone="bb", res=224, n_boot=0)
    auroc_j, n_small_j = PC.small_target_auroc_from_results(res, cat_names, min_pos=10)
    np.testing.assert_array_equal(n_small_j, n_small)
    both = np.isfinite(auroc_j)
    assert both.sum() >= 20  # most classes have >= 10 positives in each small bin
    np.testing.assert_allclose(auroc_j[both], auroc[both], atol=1e-9)
    # the JSON route is nan only where a non-empty small bin had < min_pos positives
    for c in np.flatnonzero(~both & (n_small >= 10)):
        cells = res["per_class_bin"][cat_names[c]]
        assert any(0 < cells[b]["n_pos"] < 10 for b in BIN_NAMES[:2])


def test_perclass_rows_summary_and_extremes(labels, four_readouts):
    y, mbf = labels["y"], labels["max_box_frac"]
    cat_names = [str(c) for c in labels["cat_names"]]
    small = {}
    n_small = None
    for name, s in four_readouts.items():
        small[name], n_small = PC.small_target_auroc(s, y, mbf, min_pos=10)
    rows = PC.perclass_rows(small, n_small, cat_names, n_pos_all=(y == 1).sum(axis=0), min_pos=10)
    assert len(rows) == 80
    want = {"class", "n_small", "n_pos_all", "cls", "mean", "attn", "oracle",
            "oracle_gap", "head_gain", "context_score", "eligible"}
    assert all(set(r) == want for r in rows)
    elig = [r for r in rows if r["eligible"]]
    assert 10 <= len(elig) <= 80
    for r in elig:
        assert r["n_small"] >= 10
        assert r["oracle_gap"] == pytest.approx(r["oracle"] - r["mean"])
        assert r["head_gain"] == pytest.approx(r["attn"] - r["mean"])
        assert r["context_score"] == r["mean"]
    for r in rows:
        if r["n_small"] < 10:
            assert not r["eligible"] and math.isnan(r["oracle_gap"])
    # oracle is the strongest synthetic read-out, so gaps are mostly positive
    assert np.mean([r["oracle_gap"] > 0 for r in elig]) > 0.8

    largest, smallest = PC.extreme_classes(rows, k=8)
    assert len(largest) == 8 and len(smallest) == 8
    assert not {r["class"] for r in largest} & {r["class"] for r in smallest}
    gl = [r["oracle_gap"] for r in largest]
    gs = [r["oracle_gap"] for r in smallest]
    assert gl == sorted(gl, reverse=True) and gs == sorted(gs)
    assert min(gl) >= max(gs)
    assert min(gl) == max(r["oracle_gap"] for r in elig if r["class"] not in {x["class"] for x in largest[:7]})

    summary = PC.perclass_summary(rows, k=8)
    assert summary["n_classes_eligible"] == len(elig) and summary["n_classes_total"] == 80
    assert summary["largest_gap"] == [r["class"] for r in largest]
    assert summary["smallest_gap"] == [r["class"] for r in smallest]
    assert -1.0 <= summary["spearman_context_vs_gap"] <= 1.0

    tex = PC.perclass_table_latex(largest, smallest)
    assert tex.startswith(r"\begin{tabular}") and tex.rstrip().endswith(r"\end{tabular}")
    assert r"\begin{table}" not in tex
    assert sum(line.endswith(r"\\") and line[0] != "\\" for line in tex.splitlines()) == 16 + 1  # rows + header
    assert largest[0]["class"] in tex and smallest[0]["class"] in tex
    txt = PC.perclass_table_text(largest, smallest)
    assert "largest oracle gap" in txt and "smallest oracle gap" in txt

    df = PC.perclass_dataframe(rows)
    assert df.shape == (80, 11) and "small_auroc_oracle" in df.columns

    # fewer than 2k eligible rows: the second list shrinks and never overlaps
    few = [dict(r, eligible=(i < 10)) for i, r in enumerate(rows)]
    for r in few:
        if r["eligible"] and math.isnan(r["oracle_gap"]):
            r["oracle_gap"] = 0.1
    lg, sm = PC.extreme_classes(few, k=8)
    assert len(lg) == 8 and len(sm) == 2 and not {r["class"] for r in lg} & {r["class"] for r in sm}


def test_perclass_scatter_writes_files(tmp_path, labels, four_readouts):
    y, mbf = labels["y"], labels["max_box_frac"]
    cat_names = [str(c) for c in labels["cat_names"]]
    small = {name: PC.small_target_auroc(s, y, mbf)[0] for name, s in four_readouts.items()}
    n_small = PC.small_target_auroc(four_readouts["mean"], y, mbf)[1]
    rows = PC.perclass_rows(small, n_small, cat_names)
    files = PC.plot_perclass_scatter(rows, tmp_path / "scatter.txt")
    assert [f.suffix for f in files] == [".png", ".pdf"]
    assert all(f.exists() and f.stat().st_size > 1000 for f in files)
    with pytest.raises(ValueError):
        PC.plot_perclass_scatter([dict(r, eligible=False) for r in rows], tmp_path / "none")


def test_perclass_script(tmp_path, labels, four_readouts):
    np.savez(tmp_path / "labels.npz", **labels)
    rdir = tmp_path / "results"
    for name, s in four_readouts.items():
        R.save_scores(s, R.scores_path(name, "synthetic_bb", 224, rdir))
    cmd = [PY, str(REPO / "scripts" / "09_perclass.py"), "--backbone", "synthetic_bb", "--res", "224",
           "--labels", str(tmp_path / "labels.npz"), "--results-dir", str(rdir),
           "--fig-dir", str(rdir / "figures"), "--report-fig-dir", str(tmp_path / "report" / "figures")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    assert "largest oracle gap" in proc.stdout
    assert (rdir / "perclass__synthetic_bb__res224.json").exists()
    assert (rdir / "perclass__synthetic_bb__res224.csv").exists()
    assert (rdir / "perclass_table__synthetic_bb__res224.tex").exists()
    for d in (rdir / "figures", tmp_path / "report" / "figures"):
        assert (d / "perclass__synthetic_bb__res224.png").exists()
        assert (d / "perclass__synthetic_bb__res224.pdf").exists()
    d = json.loads((rdir / "perclass__synthetic_bb__res224.json").read_text())
    assert len(d["classes"]) == 80 and d["summary"]["n_classes_eligible"] > 0
    assert d["source"]["mean"].endswith(".npy")


def test_compare_script(tmp_path, labels, four_readouts):
    np.savez(tmp_path / "labels.npz", **labels)
    rdir = tmp_path / "results"
    for name in ("attn", "mean", "cls"):
        R.save_scores(four_readouts[name], R.scores_path(name, "synthetic_bb", 224, rdir))
    cmd = [PY, str(REPO / "scripts" / "10_compare.py"), "--pairs", "attn:mean,attn:cls,oracle:mean",
           "--backbone", "synthetic_bb", "--res", "224", "--n-boot", "30",
           "--labels", str(tmp_path / "labels.npz"), "--results-dir", str(rdir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 1, proc.stderr  # oracle scores are missing -> reported, non-zero exit
    assert "skipping oracle vs mean" in proc.stderr
    for a, b in (("attn", "mean"), ("attn", "cls")):
        path = rdir / f"compare__{a}_vs_{b}__synthetic_bb__res224.json"
        assert path.exists()
        d = json.loads(path.read_text())
        assert d["a"] == a and d["b"] == b and d["n_boot"] == 30 and set(d["per_bin"]) == set(BIN_NAMES)
    assert proc.stdout.count("all mAP") == 2
    # the single-pair form
    proc2 = subprocess.run(cmd[:2] + ["--a", "mean", "--b", "cls"] + cmd[4:], capture_output=True, text=True, cwd=REPO)
    assert proc2.returncode == 0, proc2.stderr
    assert (rdir / "compare__mean_vs_cls__synthetic_bb__res224.json").exists()


# --------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------
def seed_json(readout: str, per_bin: list[float], auroc_all: float, map_: float, seed: int) -> dict:
    return {
        "readout": readout, "backbone": "bb", "res": 224, "seed": seed,
        "per_bin": {name: {"auroc_macro": v, "ci_lo": None if v is None else v - 0.01,
                           "ci_hi": None if v is None else v + 0.01, "n_classes": 5, "n_pos": 50}
                    for name, v in zip(BIN_NAMES, per_bin)},
        "overall": {"auroc_macro": auroc_all, "map": map_},
    }


def test_aggregate_seeds_two_and_one():
    r0 = seed_json("attn", [0.80, 0.85, 0.90, 0.92, 0.95, 0.97], 0.93, 0.70, seed=0)
    r1 = seed_json("attn_s1", [0.90, 0.87, 0.90, 0.94, 0.95, 0.99], 0.95, 0.74, seed=1)
    agg = S.aggregate_seeds([r0, r1], readout="attn", files=["a.json", "b.json"])
    assert agg["n_seeds"] == 2 and agg["readout"] == "attn" and agg["seeds"] == [0, 1]
    assert agg["readouts"] == ["attn", "attn_s1"] and agg["std_ddof"] == 1
    b0 = agg["per_bin"][BIN_NAMES[0]]
    assert b0["auroc_macro_mean"] == pytest.approx(0.85)
    assert b0["auroc_macro_std"] == pytest.approx(0.1 / math.sqrt(2))  # sample std of {0.8, 0.9}
    assert b0["values"] == [0.80, 0.90] and b0["n"] == 2 and b0["n_classes"] == 5
    assert agg["per_bin"][BIN_NAMES[2]]["auroc_macro_std"] == 0.0
    assert agg["overall"]["auroc_macro"]["mean"] == pytest.approx(0.94)
    assert agg["overall"]["map"]["std"] == pytest.approx(0.04 / math.sqrt(2))
    assert "2 seed" in agg["note"]
    tex = S.seeds_table_latex(agg)
    assert tex.startswith(r"\begin{tabular}") and r"\pm" in tex and "0.850 $\\pm$ 0.071" in tex and "$k=2$" in tex
    assert "±" in S.seeds_table_text(agg)

    one = S.aggregate_seeds([r0])
    assert one["n_seeds"] == 1 and one["readout"] == "attn"
    assert all(one["per_bin"][n]["auroc_macro_std"] == 0.0 for n in BIN_NAMES)
    assert one["per_bin"][BIN_NAMES[0]]["auroc_macro_mean"] == 0.80
    assert "single seed" in one["note"]
    assert "0.800 $\\pm$ 0.000" in S.seeds_table_latex(one)

    # a missing bin value is ignored in the mean and counted in n
    r2 = seed_json("attn_s2", [None, 0.85, 0.90, 0.92, 0.95, 0.97], 0.93, 0.70, seed=2)
    agg3 = S.aggregate_seeds([r0, r1, r2])
    assert agg3["per_bin"][BIN_NAMES[0]]["n"] == 2 and agg3["per_bin"][BIN_NAMES[0]]["auroc_macro_mean"] == pytest.approx(0.85)
    assert agg3["per_bin"][BIN_NAMES[0]]["values"][2] is None
    with pytest.raises(ValueError):
        S.aggregate_seeds([r0, dict(r1, backbone="other")])
    with pytest.raises(ValueError):
        S.aggregate_seeds([])


def test_seeds_script(tmp_path):
    rdir = tmp_path / "results"
    R.save_results(seed_json("attn", [0.80, 0.85, 0.90, 0.92, 0.95, 0.97], 0.93, 0.70, 0), rdir / "attn__bb__res224.json")
    R.save_results(seed_json("attn_s1", [0.90, 0.87, 0.90, 0.94, 0.95, 0.99], 0.95, 0.74, 1), rdir / "attn_s1__bb__res224.json")
    cmd = [PY, str(REPO / "scripts" / "11_seeds.py"), "--readouts", "attn", "--suffixes", ",_s1,_s2",
           "--backbone", "bb", "--res", "224", "--results-dir", str(rdir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    assert "attn_s2__bb__res224.json" in proc.stderr  # the missing seed is named
    d = json.loads((rdir / "seeds__attn__bb__res224.json").read_text())
    assert d["n_seeds"] == 2 and d["per_bin"][BIN_NAMES[0]]["auroc_macro_mean"] == pytest.approx(0.85)
    assert len(d["files"]) == 2
    tex = (rdir / "seeds_table__attn__bb__res224.tex").read_text()
    assert r"\pm" in tex and "0.850" in tex
    # unknown read-out -> non-zero exit
    proc2 = subprocess.run(cmd[:3] + ["nothere"] + cmd[4:], capture_output=True, text=True, cwd=REPO)
    assert proc2.returncode == 1


# --------------------------------------------------------------------------------------
# Cross-backbone panels and table
# --------------------------------------------------------------------------------------
def test_resolve_patch_size():
    assert P.resolve_patch_size("dinov2_vitb14_reg") == 14
    assert P.resolve_patch_size("clip_vitb16") == 16
    assert P.resolve_patch_size("siglip_vitb16") == 16
    assert P.resolve_patch_size("dinov2_something") == 14
    assert P.resolve_patch_size("unknown_backbone") == 16
    assert P.resolve_patch_size("unknown_backbone", [{"notes": "grid=28 patch_size=8"}]) == 8
    assert P.resolve_patch_size("clip_vitb16", [{"notes": "patch=32"}]) == 32
    assert P.resolve_patch_size("clip_vitb16", [{"notes": "patch=32"}], {"clip_vitb16": 7}) == 7


def test_plot_backbone_panels_and_table(tmp_path, labels, four_readouts):
    rb1 = results_for(labels, four_readouts, "dinov2_vitb14_reg")
    rb2 = results_for(labels, {k: four_readouts[k] for k in ("cls", "mean")}, "clip_vitb16")
    by_bb = {"dinov2_vitb14_reg": rb1, "clip_vitb16": rb2, "empty_bb": []}
    files = P.plot_backbone_panels(by_bb, tmp_path / "figs" / "cross.png", res=224,
                                   titles={"clip_vitb16": "CLIP ViT-B/16"})
    assert [f.name for f in files] == ["cross.png", "cross.pdf"]
    assert all(f.exists() and f.stat().st_size > 1000 for f in files)
    with pytest.raises(ValueError):
        P.plot_backbone_panels({"x": []}, tmp_path / "none.png")

    tex = P.cross_backbone_table_latex(by_bb, res=224)
    assert tex.startswith(r"\begin{tabular}") and r"\begin{table}" not in tex
    assert "dinov2 vitb14 reg" in tex and "clip vitb16" in tex and r"\textbf{" in tex
    rows = [ln for ln in tex.splitlines() if ln.endswith(r"\\") and not ln.startswith("backbone")]
    assert len(rows) == 4 + 2
    # bold: best non-oracle per backbone per column; the oracle row is never bold
    oracle_row = [ln for ln in rows if "& oracle &" in ln][0]
    assert r"\textbf" not in oracle_row
    for ln in rows[:3]:  # dinov2 non-oracle rows: each column has exactly one bold among them
        assert ln.count("&") == 8
    dino = rows[:3]
    for j in range(7):
        cells = [ln.split("&")[2 + j].strip() for ln in dino]
        assert sum(c.startswith(r"\textbf") for c in cells) >= 1
    txt = P.cross_backbone_table_text(by_bb, res=224)
    assert "dinov2_vitb14_reg" in txt and "*" in txt


def test_cross_backbone_script(tmp_path, labels, four_readouts):
    rdir = tmp_path / "results"
    for r in results_for(labels, four_readouts, "bb_one") + results_for(labels, {"cls": four_readouts["cls"]}, "bb_two"):
        R.save_results(r, R.results_path(r["readout"], r["backbone"], 224, rdir))
    cmd = [PY, str(REPO / "scripts" / "12_cross_backbone.py"), "--backbones", "bb_one,missing_bb,bb_two",
           "--res", "224", "--results-dir", str(rdir), "--fig-dir", str(rdir / "figures"),
           "--report-fig-dir", str(tmp_path / "report" / "figures")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    assert "skipping missing_bb" in proc.stdout
    for d in (rdir / "figures", tmp_path / "report" / "figures"):
        assert (d / "cross_backbone__res224.png").exists() and (d / "cross_backbone__res224.pdf").exists()
    tex = (rdir / "cross_backbone_table__res224.tex").read_text()
    assert "bb one" in tex and "bb two" in tex
    proc2 = subprocess.run(cmd[:3] + ["nothing"] + cmd[4:], capture_output=True, text=True, cwd=REPO)
    assert proc2.returncode == 1
