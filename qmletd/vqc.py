"""Variational quantum classifier with exact (autograd) gradients, plus gradient-variance
diagnostics for barren-plateau analysis.

Circuit (same family as the thesis, Listing B.1):
    RY(x_q) on every qubit, CNOT chain q -> q+1
    for l in 1..L: [optional re-upload RY(x_q)], RZ(theta), RY(theta) on every qubit, CZ layer
    final RZ(theta), RY(theta) on every qubit
Readout:
    local : <Z_0>                (cost depends on one qubit -> milder gradient decay expected;
                                  NOTE: its backward light cone covers only qubits 0..L+1)
    local_mean: (1/d) sum_q <Z_q> (still a sum of local terms, but every qubit is read out)
    global: <Z^{(x) d}> parity   (global observable -> exponential decay expected)
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from .config import seed_everything
from .sim import (apply_ry, apply_rz, chain_perm, cnot_perm, cz_sign, entangling_pairs,
                  z_signs)


class VQC(nn.Module):
    def __init__(self, d, depth=2, entanglement="linear", readout="local", reupload=False,
                 affine=True, init="uniform", device=None, complex_dtype=torch.complex64):
        super().__init__()
        self.d, self.depth, self.reupload, self.affine = d, depth, reupload, affine
        self.cdtype = complex_dtype
        rdtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
        if init == "uniform":
            th = torch.rand(depth + 1, d, 2, dtype=rdtype) * 2 * math.pi
        elif init == "small":
            th = torch.randn(depth + 1, d, 2, dtype=rdtype) * 0.1
        else:
            raise ValueError(init)
        self.theta = nn.Parameter(th.to(device))
        if affine:
            self.scale = nn.Parameter(torch.tensor(1.0, dtype=rdtype, device=device))
            self.bias = nn.Parameter(torch.tensor(0.0, dtype=rdtype, device=device))
        perms = [cnot_perm(q, q + 1, d, device) for q in range(d - 1)]
        self.register_buffer("enc_perm", chain_perm(perms) if perms else torch.arange(2 ** d, device=device))
        self.register_buffer("cz", cz_sign(entangling_pairs(d, entanglement), d, device, rdtype))
        zq, parity = z_signs(d, device, rdtype)
        obs = {"local": zq[:, 0], "local_mean": zq.mean(dim=1), "global": parity}[readout]
        self.register_buffer("obs", obs)

    def expval(self, x: torch.Tensor) -> torch.Tensor:
        B, d = x.shape[0], self.d
        psi = torch.zeros(B, 2 ** d, dtype=self.cdtype, device=x.device)
        psi[:, 0] = 1.0
        for q in range(d):
            psi = apply_ry(psi, x[:, q], q, d)
        psi = psi[:, self.enc_perm]
        for l in range(self.depth):
            if self.reupload and l > 0:
                for q in range(d):
                    psi = apply_ry(psi, x[:, q], q, d)
            for q in range(d):
                psi = apply_rz(psi, self.theta[l, q, 0], q, d)
                psi = apply_ry(psi, self.theta[l, q, 1], q, d)
            psi = psi * self.cz
        for q in range(d):
            psi = apply_rz(psi, self.theta[self.depth, q, 0], q, d)
            psi = apply_ry(psi, self.theta[self.depth, q, 1], q, d)
        probs = psi.real ** 2 + psi.imag ** 2
        return probs @ self.obs

    def logits(self, x):
        e = self.expval(x)
        return self.scale * e + self.bias if self.affine else e

    def prob(self, x):
        if self.affine:
            return torch.sigmoid(self.logits(x))
        return (1 - self.expval(x)) / 2          # P(qubit/parity = 1), thesis convention


@torch.no_grad()
def predict(model: VQC, X: np.ndarray, device, batch=4096) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = torch.as_tensor(X[s:s + batch], dtype=model.theta.dtype, device=device)
        out.append(model.prob(xb).double().cpu().numpy())
    return np.concatenate(out)


def train_vqc(X_tr, y_tr, X_va, y_va, *, d, depth=2, entanglement="linear", readout="local",
              reupload=False, loss="bce", lr=0.05, epochs=30, batch=256, patience=6,
              init="uniform", seed=0, device=None, input_scale=1.0, verbose=False):
    """Adam with exact gradients; model selection by validation PR-AUC (early stopping)."""
    seed_everything(seed)
    affine = loss == "bce"
    model = VQC(d, depth, entanglement, readout, reupload, affine, init, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(X_tr * input_scale, dtype=model.theta.dtype, device=device)
    yt = torch.as_tensor(y_tr, dtype=model.theta.dtype, device=device)
    Xv = X_va * input_scale
    g = torch.Generator(device="cpu").manual_seed(seed)
    best = (-1.0, None, 0)
    hist, bad, t0 = [], 0, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g).to(device)
        tot = 0.0
        for s in range(0, len(Xt), batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            if loss == "bce":
                l = nn.functional.binary_cross_entropy_with_logits(model.logits(Xt[idx]), yt[idx])
            else:
                l = ((model.prob(Xt[idx]) - yt[idx]) ** 2).mean()
            l.backward()
            opt.step()
            tot += float(l) * len(idx)
        ap = float(average_precision_score(y_va, predict(model, Xv, device)))
        hist.append({"epoch": ep, "train_loss": tot / len(Xt), "val_pr_auc": ap})
        if verbose:
            print(f"  ep {ep:3d} loss {tot / len(Xt):.4f} val AP {ap:.4f}")
        if ap > best[0] + 1e-5:
            best, bad = (ap, copy.deepcopy(model.state_dict()), ep), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best[1])
    return model, {"best_val_pr_auc": best[0], "best_epoch": best[2], "history": hist,
                   "train_seconds": time.time() - t0, "n_params": int(model.theta.numel())}


def gradient_variance(X, y, *, d, depth, entanglement="linear", readout="local", n_init=200,
                      n_samples=128, seed=0, device=None, input_scale=1.0):
    """Variance over random initialisations (theta ~ U[0, 2pi)) of the partial derivatives.

    Two costs: (i) C_E = mean_x <O>(x), the standard barren-plateau quantity;
    (ii) C_MSE = mean_x (p(x) - y)^2, the training loss of the thesis.
    Returns per-cost: mean over parameters of Var[dC/dtheta_j], plus Var of the first parameter.
    """
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    xb = torch.as_tensor(X[sel] * input_scale, dtype=torch.float32, device=device)
    yb = torch.as_tensor(y[sel], dtype=torch.float32, device=device)
    gE, gM = [], []
    for k in range(n_init):
        torch.manual_seed(seed * 100_003 + k)
        m = VQC(d, depth, entanglement, readout, affine=False, init="uniform", device=device)
        e = m.expval(xb)
        cE = e.mean()
        cM = (((1 - e) / 2 - yb) ** 2).mean()
        (dE,) = torch.autograd.grad(cE, m.theta, retain_graph=True)
        (dM,) = torch.autograd.grad(cM, m.theta)
        gE.append(dE.flatten().double().cpu().numpy())
        gM.append(dM.flatten().double().cpu().numpy())
    gE, gM = np.array(gE), np.array(gM)
    return {
        "var_expval_mean": float(gE.var(axis=0).mean()),
        "var_expval_first": float(gE[:, 0].var()),
        "var_mse_mean": float(gM.var(axis=0).mean()),
        "var_mse_first": float(gM[:, 0].var()),
        "grad_norm_mse_mean": float(np.linalg.norm(gM, axis=1).mean()),
        "n_params": int(gE.shape[1]),
    }
