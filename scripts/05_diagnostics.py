"""E3 - Diagnostics.

--part gradvar : variance of the gradient over >=200 random initialisations, versus number of
                 qubits, depth and readout (local vs global). Barren-plateau evidence = clear
                 exponential decay of the variance with d (fit of log-variance vs d is done in
                 09_aggregate.py).
--part kernels : for fidelity / projected / classical RBF kernels, versus d and bandwidth:
                 off-diagonal mean and variance (exponential concentration), centered
                 kernel-target alignment, geometric difference g(K_C || K_Q) (Huang et al. 2021).

Examples:
  python scripts/05_diagnostics.py --part gradvar --d 2 4 6 8 10 12 14 --depths 1 2 4 8 --n-init 200
  python scripts/05_diagnostics.py --part kernels --d 2 4 6 8 10 13 --splits 0 1 2 --n 800
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import common, config, prep, store  # noqa: E402
from qmletd.config import get_device  # noqa: E402
from qmletd.kernels import (centered_kta, fidelity_gram, geometric_difference,  # noqa: E402
                            offdiag_mean, offdiag_variance, projected_features, projected_gram,
                            quantum_states)
from qmletd.vqc import gradient_variance  # noqa: E402


def part_gradvar(a, cache, device):
    exp = "E3_gradvar"
    for split, d, depth, readout in itertools.product(a.splits, a.d, a.depths, a.readouts):
        name = f"gradvar__d{d}__L{depth}__{readout}__s{split}"
        if store.exists(exp, name) and not a.overwrite:
            continue
        # features beyond 13 are not available: for d > 13 the 13 features are tiled
        P = cache.get(split, min(d, 13))
        X = P.X_tr if d <= 13 else np.tile(P.X_tr, int(np.ceil(d / 13)))[:, :d]
        t0 = time.time()
        r = gradient_variance(X, P.y_tr, d=d, depth=depth, entanglement=a.entanglement,
                              readout=readout, n_init=a.n_init, n_samples=a.n_samples,
                              seed=split, device=device)
        store.save(exp, name, {"split": split, "d": d, "depth": depth, "readout": readout,
                               "entanglement": a.entanglement, "n_init": a.n_init,
                               "seconds": time.time() - t0, "tiled_features": d > 13, **r})
        print(f"[E3 grad] d={d:2d} L={depth} {readout:6s} Var[dE]={r['var_expval_mean']:.3e} "
              f"Var[dMSE]={r['var_mse_mean']:.3e} ({time.time() - t0:.1f}s)")


def part_kernels(a, cache, device):
    exp = "E3_kernels"
    for split, d in itertools.product(a.splits, a.d):
        P = cache.get(split, d)
        X, y = prep.stratified_subsample(P.X_tr, P.y_tr, a.n, seed=split + 5)
        g0 = float(1.0 / (d * X.var()))
        KC = rbf_kernel(X, X, gamma=g0)
        for bw in a.bandwidths:
            name = f"kernels__d{d}__bw{bw}__s{split}"
            if store.exists(exp, name) and not a.overwrite:
                continue
            t0 = time.time()
            S = quantum_states(X, bw, a.reps, a.entanglement, device)
            KQ = fidelity_gram(S)
            KP = projected_gram(projected_features(S, d), gamma=1.0)
            out = {"split": split, "d": d, "bandwidth": bw, "n": len(y), "reps": a.reps,
                   "entanglement": a.entanglement, "rbf_gamma": g0}
            for nm, K in (("fidelity", KQ), ("projected", KP), ("rbf", KC)):
                out[f"{nm}_offdiag_mean"] = offdiag_mean(K)
                out[f"{nm}_offdiag_var"] = offdiag_variance(K)
                out[f"{nm}_kta"] = centered_kta(K, y)
            for reg in a.regs:
                out[f"g_rbf_fidelity_reg{reg}"] = geometric_difference(KC, KQ, reg)
                out[f"g_rbf_projected_reg{reg}"] = geometric_difference(KC, KP, reg)
            out["seconds"] = time.time() - t0
            store.save(exp, name, out)
            print(f"[E3 kern] d={d:2d} bw={bw:<5} var(KQ)={out['fidelity_offdiag_var']:.2e} "
                  f"KTA Q/P/C={out['fidelity_kta']:.3f}/{out['projected_kta']:.3f}/{out['rbf_kta']:.3f} "
                  f"g(C||Q)={out[f'g_rbf_fidelity_reg{a.regs[0]}']:.2f}")


if __name__ == "__main__":
    ap = common.base_parser(__doc__)
    ap.add_argument("--part", required=True, choices=["gradvar", "kernels"])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--readouts", nargs="+", default=["local", "global"])
    ap.add_argument("--entanglement", default=None)
    ap.add_argument("--n-init", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--n", type=int, default=800, help="kernel diagnostics sample size")
    ap.add_argument("--bandwidths", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--regs", type=float, nargs="+", default=[1e-3, 1e-1])
    a = ap.parse_args()
    if a.splits is config.DEFAULT_SPLITS:   # not given on the command line
        a.splits = [0]
    device = get_device(a.device)
    cache = common.DataCache()
    if a.part == "gradvar":
        a.entanglement = a.entanglement or "linear"
        part_gradvar(a, cache, device)
    else:
        a.entanglement = a.entanglement or "full"
        part_kernels(a, cache, device)
