"""QUBO formulation of the kernel SVM (Willsch, Willsch, De Raedt, Michielsen, CPC 2020)
with bagging over small subsets, and the matched classical control.

alpha_n = sum_{k=0}^{K-1} B^k a_{Kn+k},   C = sum_k B^k
E(a) = 1/2 sum a_i a_j B^{k+j} t_n t_m (k(x_n,x_m) + xi) - sum B^k a_{Kn+k}
Bias (Willsch eq. for b): weighted average with weights alpha_n (C - alpha_n).

Solvers: simulated annealing (dwave-samplers, seeded and reproducible) by default;
the same QUBOs can be sent to a D-Wave QPU with scripts/07_dwave_qmsvm.py.

Classical control ("svm_bagged"): the EXACT SVM (libsvm, same kernel, same C, same subsets,
same aggregation). Comparing qmsvm vs svm_bagged isolates the effect of the QUBO
encoding + low-energy ensemble from the effect of subsampling.
"""
from __future__ import annotations

import time

import dimod
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC


def build_qubo(Kmat: np.ndarray, t: np.ndarray, B: int, K: int, xi: float) -> np.ndarray:
    """Upper-triangular QUBO matrix Q such that E(a) = sum_{i<=j} Q_ij a_i a_j."""
    Bk = float(B) ** np.arange(K)
    T = np.outer(t, t) * (Kmat + xi)
    Qf = 0.5 * np.kron(T, np.outer(Bk, Bk))
    Qf[np.diag_indices_from(Qf)] -= np.tile(Bk, len(t))
    return np.triu(Qf + Qf.T, 1) + np.diag(np.diag(Qf))


def qubo_energy(Q: np.ndarray, a: np.ndarray) -> float:
    return float(a @ Q @ a)


def decode_alphas(samples: np.ndarray, B: int, K: int) -> np.ndarray:
    Bk = float(B) ** np.arange(K)
    n = samples.shape[1] // K
    return samples.reshape(samples.shape[0], n, K) @ Bk


def willsch_bias(alpha, t, Kmat, C):
    w = alpha * (C - alpha)
    resid = t - Kmat @ (alpha * t)
    if w.sum() > 0:
        return float((w * resid).sum() / w.sum())
    sv = alpha > 0                      # fallback: mean residual over support vectors
    return float(resid[sv].mean()) if sv.any() else 0.0


class _Member:
    def __init__(self, X, t, alphas, biases):
        self.X, self.t, self.alphas, self.biases = X, t, alphas, biases

    def decision(self, Xq, gamma):
        Kq = rbf_kernel(self.X, Xq, gamma=gamma)              # (n_sub, n_query)
        f = (self.alphas * self.t) @ Kq + self.biases[:, None]   # (n_solutions, n_query)
        return f.mean(axis=0)


def _subsets(y, n_subsets, subset_size, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    h = subset_size // 2
    out = []
    for _ in range(n_subsets):
        out.append(np.sort(np.r_[rng.choice(pos, h, replace=False),
                                 rng.choice(neg, subset_size - h, replace=False)]))
    return out


class QMSVM:
    def __init__(self, B=2, K=2, xi=1.0, gamma=1.0, n_subsets=20, subset_size=80,
                 num_reads=200, num_sweeps=1000, n_solutions=20, seed=0, sampler=None):
        self.B, self.K, self.xi, self.gamma = B, K, xi, gamma
        self.n_subsets, self.subset_size = n_subsets, subset_size
        self.num_reads, self.num_sweeps, self.n_solutions = num_reads, num_sweeps, n_solutions
        self.seed, self.sampler = seed, sampler
        self.C = float(sum(B ** k for k in range(K)))

    def fit(self, X, y):
        from dwave.samplers import SimulatedAnnealingSampler

        sampler = self.sampler or SimulatedAnnealingSampler()
        t_all = np.where(y == 1, 1.0, -1.0)
        self.members_, t0 = [], time.time()
        self.subsets_ = _subsets(y, self.n_subsets, self.subset_size, self.seed)
        for i, idx in enumerate(self.subsets_):
            Xs, ts = X[idx], t_all[idx]
            Ks = rbf_kernel(Xs, Xs, gamma=self.gamma)
            Q = build_qubo(Ks, ts, self.B, self.K, self.xi)
            bqm = dimod.BinaryQuadraticModel.from_qubo(
                {(a, b): Q[a, b] for a, b in zip(*np.nonzero(Q))})
            kw = dict(num_reads=self.num_reads)
            if self.sampler is None:
                kw.update(num_sweeps=self.num_sweeps, seed=self.seed * 1000 + i)
            ss = sampler.sample(bqm, **kw).aggregate()
            order = np.argsort(ss.record.energy)[: self.n_solutions]
            var_order = list(bqm.variables)
            S = np.zeros((len(order), Q.shape[0]))
            for r, rec in enumerate(ss.record.sample[order]):
                S[r, var_order] = rec
            alphas = decode_alphas(S, self.B, self.K)
            biases = np.array([willsch_bias(a, ts, Ks, self.C) for a in alphas])
            self.members_.append(_Member(Xs, ts, alphas, biases))
        self.fit_seconds_ = time.time() - t0
        return self

    def decision_function(self, Xq):
        return np.mean([m.decision(Xq, self.gamma) for m in self.members_], axis=0)


class BaggedSVM:
    """Exact SVM on the same subsets, same kernel, same C, same averaging of decisions."""

    def __init__(self, C=3.0, gamma=1.0, subsets=None):
        self.C, self.gamma, self.subsets = C, gamma, subsets

    def fit(self, X, y):
        t0 = time.time()
        self.models_ = []
        for idx in self.subsets:
            m = SVC(C=self.C, kernel="rbf", gamma=self.gamma).fit(X[idx], y[idx])
            self.models_.append(m)
        self.fit_seconds_ = time.time() - t0
        return self

    def decision_function(self, Xq):
        return np.mean([m.decision_function(Xq) for m in self.models_], axis=0)
