"""Batched exact statevector simulation in PyTorch (CPU or GPU).

Conventions follow Qiskit: qubit q is bit q of the basis-state index (little-endian),
so results can be checked against qiskit.quantum_info.Statevector (see tests/).

Why a custom simulator: the circuits used here (ZZ feature map, angle encoding + RZ/RY/CZ
ansatz) have a simple structure that can be simulated for thousands of samples at once,
which makes kernels on the FULL training set and exact-gradient training feasible.
Correctness is verified gate-by-gate against Qiskit in tests/test_sim_vs_qiskit.py.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import torch

# --------------------------------------------------------------------------- basics

def bits_table(d: int, device=None) -> torch.Tensor:
    idx = torch.arange(2 ** d, device=device)
    return ((idx[:, None] >> torch.arange(d, device=device)) & 1).to(torch.int64)


def entangling_pairs(d: int, entanglement: str):
    if d < 2:
        return []
    if entanglement == "linear":
        return [(i, i + 1) for i in range(d - 1)]
    if entanglement == "ring":
        return [(i, i + 1) for i in range(d - 1)] + ([(d - 1, 0)] if d > 2 else [])
    if entanglement == "full":
        return list(combinations(range(d), 2))
    raise ValueError(entanglement)


def _view_q(psi: torch.Tensor, q: int, d: int):
    B = psi.shape[0]
    return psi.reshape(B, 2 ** (d - q - 1), 2, 2 ** q)


def apply_ry(psi, theta, q, d):
    """RY(theta) on qubit q. theta: scalar tensor or shape (B,)."""
    v = _view_q(psi, q, d)
    a, b = v[:, :, 0, :], v[:, :, 1, :]
    th = theta.reshape(-1, 1, 1) if theta.dim() else theta
    c, s = torch.cos(th / 2), torch.sin(th / 2)
    return torch.stack((c * a - s * b, s * a + c * b), dim=2).reshape(psi.shape)


def apply_rz(psi, theta, q, d):
    """RZ(theta) = diag(e^{-i theta/2}, e^{i theta/2}) on qubit q."""
    v = _view_q(psi, q, d)
    a, b = v[:, :, 0, :], v[:, :, 1, :]
    th = theta.reshape(-1, 1, 1) if theta.dim() else theta
    ph = torch.polar(torch.ones_like(th), th / 2).to(psi.dtype)
    return torch.stack((a * ph.conj(), b * ph), dim=2).reshape(psi.shape)


def apply_h_all(psi, d):
    """Hadamard on every qubit (fast Walsh-Hadamard transform)."""
    inv = 1.0 / math.sqrt(2.0)
    for q in range(d):
        v = _view_q(psi, q, d)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        psi = (torch.stack((a + b, a - b), dim=2) * inv).reshape(psi.shape)
    return psi


def cnot_perm(control: int, target: int, d: int, device=None) -> torch.Tensor:
    """Index permutation p such that (CNOT psi)[i] = psi[p[i]]."""
    idx = torch.arange(2 ** d, device=device)
    flip = ((idx >> control) & 1) << target
    return idx ^ flip


def chain_perm(perms):
    """Compose permutations applied left-to-right: psi -> psi[p1] -> ...[p2]."""
    out = perms[0]
    for p in perms[1:]:
        out = out[p]
    return out


def cz_sign(pairs, d, device=None, dtype=torch.float32) -> torch.Tensor:
    bits = bits_table(d, device)
    par = torch.zeros(2 ** d, dtype=torch.int64, device=device)
    for i, j in pairs:
        par = par + bits[:, i] * bits[:, j]
    return (1 - 2 * (par % 2)).to(dtype)


def z_signs(d, device=None, dtype=torch.float32):
    """Return (Z on each qubit [2^d, d], parity Z^{x d} [2^d]) as +-1 tensors."""
    bits = bits_table(d, device)
    zq = (1 - 2 * bits).to(dtype)
    parity = (1 - 2 * (bits.sum(dim=1) % 2)).to(dtype)
    return zq, parity


# ------------------------------------------------------------- ZZ feature map states

def zz_basis_matrix(d: int, entanglement: str = "full", device=None, dtype=torch.float64):
    bits = bits_table(d, device)
    pairs = entangling_pairs(d, entanglement)
    rows = [bits[:, i] for i in range(d)] + [bits[:, i] ^ bits[:, j] for i, j in pairs]
    return torch.stack(rows, dim=0).to(dtype), pairs


def zz_feature_states(X: np.ndarray, reps: int = 2, entanglement: str = "full",
                      alpha: float = 2.0, device=None, complex_dtype=torch.complex64,
                      chunk: int = 4096) -> torch.Tensor:
    """States of Qiskit's ZZFeatureMap(feature_dimension=d, reps, entanglement, alpha).

    Each repetition is H^{(x)d} followed by the diagonal U_Phi(x) with
    phase(z) = alpha * [ sum_i x_i z_i + sum_{(i,j)} (pi - x_i)(pi - x_j) (z_i XOR z_j) ].
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    real = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    Bm, pairs = zz_basis_matrix(d, entanglement, device=device, dtype=torch.float64)
    out = torch.empty((N, 2 ** d), dtype=complex_dtype, device=device)
    for s in range(0, N, chunk):
        x = torch.as_tensor(X[s:s + chunk], dtype=torch.float64, device=device)
        feats = [x] + ([torch.stack([(math.pi - x[:, i]) * (math.pi - x[:, j]) for i, j in pairs],
                                    dim=1)] if pairs else [])
        phi = torch.cat(feats, dim=1)
        theta = (alpha * phi @ Bm).to(real)
        diag = torch.polar(torch.ones_like(theta), theta).to(complex_dtype)
        psi = diag / math.sqrt(2 ** d)
        for _ in range(reps - 1):
            psi = apply_h_all(psi, d) * diag
        out[s:s + chunk] = psi
    return out


def pauli_expectations(psi: torch.Tensor, d: int) -> torch.Tensor:
    """Single-qubit <X>, <Y>, <Z> for every qubit -> tensor [N, 3d] (real)."""
    feats = []
    for q in range(d):
        v = _view_q(psi, q, d)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        w = (a.conj() * b).sum(dim=(1, 2))
        z = (a.abs() ** 2).sum(dim=(1, 2)) - (b.abs() ** 2).sum(dim=(1, 2))
        feats += [2 * w.real, 2 * w.imag, z]
    return torch.stack(feats, dim=1)
