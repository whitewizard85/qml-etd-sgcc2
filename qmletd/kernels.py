"""Quantum (fidelity / projected) and classical kernels plus kernel diagnostics."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel

from .sim import pauli_expectations, zz_feature_states


def quantum_states(X, bandwidth=1.0, reps=2, entanglement="full", device=None,
                   complex_dtype=torch.complex64):
    """Bandwidth scales the input before encoding (Shaydulin & Wild style)."""
    return zz_feature_states(np.asarray(X) * bandwidth, reps=reps, entanglement=entanglement,
                             device=device, complex_dtype=complex_dtype)


def fidelity_gram(A: torch.Tensor, B: torch.Tensor | None = None, chunk: int = 2048) -> np.ndarray:
    """K[i, j] = |<a_i|b_j>|^2 computed in chunks; returns float64 numpy array."""
    Bm = A if B is None else B
    out = np.empty((A.shape[0], Bm.shape[0]), dtype=np.float64)
    BH = Bm.conj().T
    for s in range(0, A.shape[0], chunk):
        G = A[s:s + chunk] @ BH
        out[s:s + chunk] = (G.real ** 2 + G.imag ** 2).double().cpu().numpy()
    if B is None:
        np.fill_diagonal(out, 1.0)
    return out


def projected_features(psi: torch.Tensor, d: int) -> np.ndarray:
    return pauli_expectations(psi, d).double().cpu().numpy()


def projected_gram(FA: np.ndarray, FB: np.ndarray | None = None, gamma: float = 1.0) -> np.ndarray:
    """Projected quantum kernel (Huang et al. 2021): exp(-gamma * sum_k ||rho_k - rho'_k||_F^2).

    With single-qubit RDMs, ||rho - rho'||_F^2 = 1/2 * sum_P (<P> - <P>')^2.
    """
    return rbf_kernel(FA, FA if FB is None else FB, gamma=0.5 * gamma)


# ------------------------------------------------------------------ diagnostics

def offdiag_variance(K: np.ndarray) -> float:
    iu = np.triu_indices_from(K, k=1)
    return float(np.var(K[iu]))


def offdiag_mean(K: np.ndarray) -> float:
    iu = np.triu_indices_from(K, k=1)
    return float(np.mean(K[iu]))


def centered_kta(K: np.ndarray, y: np.ndarray) -> float:
    """Centered kernel-target alignment (Cortes et al. 2012), labels mapped to +-1."""
    t = np.where(np.asarray(y) > 0, 1.0, -1.0)
    n = len(t)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Y = np.outer(t, t)
    Yc = H @ Y @ H
    den = np.linalg.norm(Kc) * np.linalg.norm(Yc)
    return float((Kc * Yc).sum() / den) if den > 0 else 0.0


def _psd_sqrt(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.T


def geometric_difference(K_C: np.ndarray, K_Q: np.ndarray, reg: float = 1e-3) -> float:
    """g(K_C || K_Q) = sqrt(|| sqrt(K_Q) (K_C + reg I)^{-1} sqrt(K_Q) ||_inf), Huang et al. (2021).

    Both kernels are normalised to trace N first. Small g (close to 1) means a classical
    model with K_C can match the quantum kernel on any labelling; large g (order sqrt(N))
    is a necessary (not sufficient) condition for a potential quantum advantage.
    """
    n = K_C.shape[0]
    KC = K_C * n / np.trace(K_C)
    KQ = K_Q * n / np.trace(K_Q)
    sQ = _psd_sqrt(KQ)
    M = sQ @ np.linalg.solve(KC + reg * np.eye(n), sQ)
    lam = np.linalg.eigvalsh((M + M.T) / 2).max()
    return float(np.sqrt(max(lam, 0.0)))
