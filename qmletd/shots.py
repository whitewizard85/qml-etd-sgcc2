"""Training a VQC with FINITE-SHOT parameter-shift gradients (what real hardware would give),
next to the exact gradient computed on the same circuits for diagnostics.

For every optimisation step, mini-batch B and P circuit parameters:
  * 2P + 1 circuits per sample are evaluated exactly on the statevector (batched);
  * shot noise is then added analytically:
      - +-1 observables (local Z_0, global parity): estimate = 2*Binomial(S, (1+E)/2)/S - 1  (exact law)
      - local_mean = (1/d) sum_q Z_q: Gaussian with the exact single-shot variance E[O^2]-E^2,
        divided by S (CLT; accurate for S >= ~100)
  * parameter-shift rule (exact for RZ/RY):  dE/dtheta_j = [E(theta_j + pi/2) - E(theta_j - pi/2)] / 2
  * BCE loss on logit = a*E + b (a, b classical parameters), chain rule with the ESTIMATED E.
Because the exact E of every shifted circuit is computed anyway, the exact gradient comes for
free: we log cosine(g_hat, g_exact) and ||g_hat - g_exact|| / ||g_exact|| at each step.
Prediction/evaluation uses exact expectation values, so differences in test performance are
due to TRAINING with noisy gradients only.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .config import seed_everything
from .sim import apply_ry, apply_rz
from .vqc import VQC


def moments_batched(m: VQC, x: torch.Tensor, theta: torch.Tensor, chunk: int = 4096):
    """Exact <O> and <O^2> for per-row parameters theta [N, L+1, d, 2] and inputs x [N, d]."""
    d, L = m.d, m.depth
    E_all, E2_all = [], []
    for s in range(0, x.shape[0], chunk):
        xb, th = x[s:s + chunk], theta[s:s + chunk]
        B = xb.shape[0]
        psi = torch.zeros(B, 2 ** d, dtype=m.cdtype, device=xb.device)
        psi[:, 0] = 1.0
        for q in range(d):
            psi = apply_ry(psi, xb[:, q], q, d)
        psi = psi[:, m.enc_perm]
        for l in range(L):
            if m.reupload and l > 0:
                for q in range(d):
                    psi = apply_ry(psi, xb[:, q], q, d)
            for q in range(d):
                psi = apply_rz(psi, th[:, l, q, 0], q, d)
                psi = apply_ry(psi, th[:, l, q, 1], q, d)
            psi = psi * m.cz
        for q in range(d):
            psi = apply_rz(psi, th[:, L, q, 0], q, d)
            psi = apply_ry(psi, th[:, L, q, 1], q, d)
        probs = psi.real ** 2 + psi.imag ** 2
        E_all.append(probs @ m.obs)
        E2_all.append(probs @ (m.obs ** 2))
    return torch.cat(E_all), torch.cat(E2_all)


def add_shot_noise(E, E2, shots, pm1: bool, gen: torch.Generator):
    if shots is None:
        return E
    if pm1:
        p = ((1 + E) / 2).clamp(0, 1).double().cpu()
        k = torch.binomial(torch.full_like(p, float(shots)), p, generator=gen)
        return (2 * k / shots - 1).to(E.dtype).to(E.device)
    var = (E2 - E ** 2).clamp(min=0)
    z = torch.randn(E.shape, generator=gen, dtype=torch.float64).to(E.device).to(E.dtype)
    return (E + torch.sqrt(var / shots) * z).clamp(-1, 1)


def train_vqc_shots(X_tr, y_tr, X_va, y_va, *, d, depth=2, readout="global", entanglement="linear",
                    shots=None, steps=300, batch=64, lr=0.05, optimizer="adam", eval_every=25,
                    seed=0, device=None, chunk=4096, verbose=False):
    seed_everything(seed)
    gen = torch.Generator().manual_seed(seed + 99)
    m = VQC(d, depth, entanglement, readout, reupload=False, affine=True, init="uniform", device=device)
    pm1 = readout in ("local", "global")
    params = [m.theta, m.scale, m.bias]
    opt = (torch.optim.Adam(params, lr=lr) if optimizer == "adam" else torch.optim.SGD(params, lr=lr))
    Xt = torch.as_tensor(X_tr, dtype=m.theta.dtype, device=device)
    yt = torch.as_tensor(y_tr, dtype=m.theta.dtype, device=device)
    P = m.theta.numel()
    shifts = torch.zeros(2 * P + 1, P, dtype=m.theta.dtype, device=device)
    idx = torch.arange(P, device=device)
    shifts[1 + idx, idx] = math.pi / 2
    shifts[1 + P + idx, idx] = -math.pi / 2
    rng = np.random.default_rng(seed)
    best, hist, diag, t0 = (-1.0, None, -1), [], [], time.time()
    for step in range(1, steps + 1):
        sel = torch.as_tensor(rng.choice(len(Xt), size=batch, replace=False), device=device)
        xb, yb = Xt[sel], yt[sel]
        with torch.no_grad():
            th_rows = (m.theta.flatten()[None, :] + shifts)                  # [2P+1, P]
            th_rows = th_rows.repeat_interleave(batch, dim=0).reshape(-1, *m.theta.shape)
            x_rows = xb.repeat(2 * P + 1, 1)
            E, E2 = moments_batched(m, x_rows, th_rows, chunk)
            Eh = add_shot_noise(E, E2, shots, pm1, gen)
            E, Eh = E.reshape(2 * P + 1, batch), Eh.reshape(2 * P + 1, batch)
            a, b = m.scale, m.bias

            def grads(Em):
                e0 = Em[0]
                dE = (Em[1:P + 1] - Em[P + 1:]) / 2                          # [P, B]
                r = torch.sigmoid(a * e0 + b) - yb                            # dBCE/dlogit
                return (dE * r).mean(dim=1) * a, (r * e0).mean(), r.mean()

            g_th, g_a, g_b = grads(Eh)
            ge_th, _, _ = grads(E)
            cos = float(torch.nn.functional.cosine_similarity(g_th, ge_th, dim=0))
            rel = float((g_th - ge_th).norm() / (ge_th.norm() + 1e-30))
            diag.append((cos, rel, float(ge_th.norm())))
        opt.zero_grad()
        m.theta.grad = g_th.reshape(m.theta.shape).clone()
        m.scale.grad, m.bias.grad = g_a.clone(), g_b.clone()
        opt.step()
        if step % eval_every == 0 or step == steps:
            from .vqc import predict
            ap = float(average_precision_score(y_va, predict(m, X_va, device)))
            c = np.array(diag[-eval_every:])
            hist.append({"step": step, "val_pr_auc": ap, "cos_mean": float(np.nanmean(c[:, 0])),
                         "rel_err_median": float(np.nanmedian(c[:, 1])),
                         "exact_grad_norm_mean": float(c[:, 2].mean())})
            if verbose:
                print(f"  step {step:4d} val AP {ap:.4f} cos {hist[-1]['cos_mean']:.3f} "
                      f"rel.err {hist[-1]['rel_err_median']:.2f}")
            if ap > best[0]:
                best = (ap, copy.deepcopy(m.state_dict()), step)
    m.load_state_dict(best[1])
    c = np.array(diag)
    return m, {"best_val_pr_auc": best[0], "best_step": best[2], "history": hist,
               "cos_mean_all": float(np.nanmean(c[:, 0])),
               "cos_first25": float(np.nanmean(c[:25, 0])),          "rel_err_median_all": float(np.nanmedian(c[:, 1])),
               "train_seconds": time.time() - t0, "n_params": int(P),
               "circuits_per_step": int((2 * P + 1) * batch),
               "shots_total": None if shots is None else int((2 * P + 1) * batch * shots * steps)}
