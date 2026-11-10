"""Tests for the hold-out protocol, grouped splits and the dependence-aware statistics."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from qmletd import config, prep
from qmletd.stats import (_weighted_ap, bootstrap_diff, bootstrap_diff_fast,
                          corrected_resampled_ttest, corrected_tost)


@pytest.fixture
def y():
    return (np.random.default_rng(0).random(4000) < 0.085).astype(int)


def test_splits_protocol_unchanged(y, monkeypatch):
    monkeypatch.setattr(config, "PROTOCOL", "splits")
    monkeypatch.setattr(config, "GROUPS_FILE", config.ROOT / "nonexistent_groups.npy")
    for seed in (0, 7):
        s = prep.make_split(y, seed)
        idx = np.arange(len(y))
        tr, tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed + 10_000)
        for a, b in ((tr, s.train), (va, s.val), (te, s.test)):
            assert np.array_equal(np.sort(a), b)


def test_holdout_test_fixed_and_untouched(y, monkeypatch):
    monkeypatch.setattr(config, "PROTOCOL", "holdout")
    monkeypatch.setattr(config, "GROUPS_FILE", config.ROOT / "nonexistent_groups.npy")
    S = [prep.make_split(y, k) for k in range(config.HOLDOUT_FOLDS)]
    te = S[0].test
    assert all(np.array_equal(s.test, te) for s in S)
    val = np.concatenate([s.val for s in S])
    assert len(np.unique(val)) == len(val) == len(y) - len(te)      # folds partition the rest
    for s in S:
        assert not (set(s.train) | set(s.val)) & set(te)
        assert not set(s.train) & set(s.val)
    assert abs(y[te].mean() - y.mean()) < 0.01
    with pytest.raises(ValueError):
        prep.make_split(y, config.HOLDOUT_FOLDS)


@pytest.mark.parametrize("protocol", ["splits", "holdout"])
def test_groups_never_split(y, monkeypatch, protocol):
    monkeypatch.setattr(config, "PROTOCOL", protocol)
    g = np.repeat(np.arange(40), len(y) // 40)
    for k in range(3):
        s = prep.make_split(y, k, groups=g)
        parts = [set(g[s.train]), set(g[s.val]), set(g[s.test])]
        assert not parts[0] & parts[1] and not parts[0] & parts[2] and not parts[1] & parts[2]


def test_weighted_ap_matches_sklearn():
    rng = np.random.default_rng(3)
    yy = (rng.random(700) < 0.2).astype(int)
    s = np.round(yy + rng.normal(size=700), 1)                      # with ties
    W = rng.integers(0, 3, size=(5, 700))
    ref = [average_precision_score(yy, s, sample_weight=w) for w in W]
    assert np.allclose(_weighted_ap(yy, s, W), ref)


def test_fast_bootstrap_agrees_with_reference():
    rng = np.random.default_rng(4)
    yy = (rng.random(1500) < 0.1).astype(int)
    s1, s2 = yy * 0.8 + rng.normal(size=1500), yy * 0.6 + rng.normal(size=1500)
    a = bootstrap_diff_fast(yy, s1, s2, n_boot=3000, seed=1)
    b = bootstrap_diff(yy, s1, s2, n_boot=600, seed=2)
    assert a["diff"] == pytest.approx(b["diff"])
    assert a["se"] == pytest.approx(b["se"], rel=0.15)


def test_corrected_ttest_is_wider_than_naive():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.30, 0.01, 10), rng.normal(0.29, 0.01, 10)
    m, p, (lo, hi) = corrected_resampled_ttest(a, b, 1 / 3)
    d = a - b
    naive = 2.262 * d.std(ddof=1) / np.sqrt(10)
    assert m == pytest.approx(d.mean())
    assert (hi - lo) / 2 == pytest.approx(2.262 * d.std(ddof=1) * np.sqrt(0.1 + 1 / 3), rel=1e-3)
    assert (hi - lo) / 2 > naive
    p_eq, ci90 = corrected_tost(a, b, 1.0, 1 / 3)
    assert p_eq < 0.001 and ci90[0] < m < ci90[1]
