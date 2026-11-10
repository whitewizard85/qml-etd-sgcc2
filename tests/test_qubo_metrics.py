import itertools

import numpy as np
import pytest
from sklearn.metrics import fbeta_score

from qmletd.metrics import best_fbeta_threshold
from qmletd.qubo_svm import build_qubo, decode_alphas, qubo_energy
from qmletd.stats import delong, holm


def test_qubo_matches_dual_objective():
    rng = np.random.default_rng(0)
    n, B, K, xi = 4, 2, 2, 1.0
    X = rng.normal(size=(n, 2))
    Km = np.exp(-((X[:, None] - X[None]) ** 2).sum(-1))
    t = np.array([1, -1, 1, -1.0])
    Q = build_qubo(Km, t, B, K, xi)
    for bits in itertools.product([0, 1], repeat=n * K):
        a = np.array(bits, float)
        al = decode_alphas(a[None], B, K)[0]
        dual = 0.5 * (al * t) @ Km @ (al * t) - al.sum() + 0.5 * xi * (al @ t) ** 2
        assert qubo_energy(Q, a) == pytest.approx(dual, abs=1e-9)


def test_threshold_exhaustive():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.1).astype(int)
    s = np.round(rng.normal(size=300) + y, 1)          # many ties on purpose
    thr, f = best_fbeta_threshold(y, s, beta=2)
    brute = max(fbeta_score(y, (s >= u).astype(int), beta=2, zero_division=0) for u in np.unique(s))
    assert f == pytest.approx(brute, abs=1e-12)
    assert fbeta_score(y, (s >= thr).astype(int), beta=2) == pytest.approx(f, abs=1e-12)


def test_delong_identical_and_holm():
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.2).astype(int)
    s = rng.normal(size=500) + y
    a1, a2, z, p = delong(y, s, s)
    assert a1 == a2 and p == 1.0
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
