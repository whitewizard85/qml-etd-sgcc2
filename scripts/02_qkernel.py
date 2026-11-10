"""E2 - Quantum kernels (fidelity ZZ and projected) vs a matched classical RBF kernel,
all through the SAME precomputed-kernel SVM code path, on the SAME training subsets.

Protocol per (split, d, M):
  1. hyper-parameters (bandwidth, C, gamma) selected by validation PR-AUC using a tuning
     subsample of size <= --tune-max of the training set;
  2. refit of the selected configuration on the whole (matched) training set of size M;
  3. F2 threshold on validation, single evaluation on test.

Example (full scale, GPU recommended for d >= 10):
  python scripts/02_qkernel.py --d 2 4 6 8 10 13 --M 150 1040 full --models qk_fidelity qk_projected rbf_matched
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import common, metrics, store  # noqa: E402
from qmletd.config import get_device, seed_everything  # noqa: E402
from qmletd.kernels import (fidelity_gram, offdiag_mean, offdiag_variance,  # noqa: E402
                            projected_features, projected_gram, quantum_states)

EXP = "E2_qkernel"


def kernel_fn(model, d, device, reps, ent, cdtype):
    """Returns f(params, XA, XB=None) -> Gram matrix, and the parameter grid (without C)."""
    if model == "qk_fidelity":
        def f(p, XA, XB=None):
            SA = quantum_states(XA, p["bandwidth"], reps, ent, device, cdtype)
            SB = None if XB is None else quantum_states(XB, p["bandwidth"], reps, ent, device, cdtype)
            return fidelity_gram(SA, SB)
        return f, [{"bandwidth": b} for b in A.bandwidths]
    if model == "qk_projected":
        def f(p, XA, XB=None):
            FA = projected_features(quantum_states(XA, p["bandwidth"], reps, ent, device, cdtype), d)
            FB = None if XB is None else projected_features(
                quantum_states(XB, p["bandwidth"], reps, ent, device, cdtype), d)
            return projected_gram(FA, FB, p["gamma"])
        return f, [{"bandwidth": b, "gamma": g} for b in A.bandwidths for g in A.proj_gammas]
    if model == "rbf_matched":
        def f(p, XA, XB=None):
            return rbf_kernel(XA, XA if XB is None else XB, gamma=p["gamma"])
        return f, None  # grid depends on data scale, built below
    raise ValueError(model)


def evaluate_grid(f, grid, Xt, yt, Xv, yv):
    best = (-1, None)
    for p in grid:
        Kt, Kv = f(p, Xt), f(p, Xv, Xt)
        for C in A.C:
            s = SVC(C=C, kernel="precomputed").fit(Kt, yt).decision_function(Kv)
            ap = average_precision_score(yv, s)
            if ap > best[0]:
                best = (ap, {**p, "C": C})
    return best


if __name__ == "__main__":
    ap = common.base_parser(__doc__)
    ap.add_argument("--models", nargs="+", default=["qk_fidelity", "qk_projected", "rbf_matched"])
    ap.add_argument("--bandwidths", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--proj-gammas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--C", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--entanglement", default="full", choices=["full", "linear"])
    ap.add_argument("--tune-max", type=int, default=2000)
    ap.add_argument("--rbf-mults", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0, 10.0],
                    help="rbf_matched grid: gamma = mult / (d * var(X))")
    ap.add_argument("--precision", default="single", choices=["single", "double"],
                    help="double recommended when bandwidth < 0.01 (kernel values very close to 1)")
    A = ap.parse_args()
    device = get_device(A.device)
    import torch
    cdtype = torch.complex128 if A.precision == "double" else torch.complex64
    cache = common.DataCache()
    for split in A.splits:
        for d in A.d:
            P = cache.get(split, d)
            for M in common.parse_M(A.M):
                Xtr, ytr = cache.train_subset(P, M)
                Xtu, ytu = cache.tune_subset(Xtr, ytr, A.tune_max, split)
                if Xtu is None:
                    Xtu, ytu = Xtr, ytr
                for model in A.models:
                    name = store.run_name(model, d, M, split, A.tag)
                    if store.exists(EXP, name) and not A.overwrite:
                        continue
                    seed_everything(split)
                    f, grid = kernel_fn(model, d, device, A.reps, A.entanglement, cdtype)
                    if grid is None:
                        g0 = float(1.0 / (d * Xtr.var()))
                        grid = [{"gamma": g0 * k} for k in A.rbf_mults]
                    t0 = time.time()
                    best_ap, best_p = evaluate_grid(f, grid, Xtu, ytu, P.X_va, P.y_va)
                    t_tune = time.time() - t0
                    t1 = time.time()
                    Ktr = f(best_p, Xtr)
                    svc = SVC(C=best_p["C"], kernel="precomputed").fit(Ktr, ytr)
                    s_va = svc.decision_function(f(best_p, P.X_va, Xtr))
                    s_te = svc.decision_function(f(best_p, P.X_te, Xtr))
                    t_fit = time.time() - t1
                    rng = np.random.default_rng(split)
                    sub = rng.choice(len(ytr), size=min(500, len(ytr)), replace=False)
                    Ksub = Ktr[np.ix_(sub, sub)]
                    res = metrics.finalize(P.y_va, s_va, P.y_te, s_te)
                    store.save(EXP, name, common.meta(
                        P, model, M, len(ytr), params=best_p, val_pr_auc_selection=best_ap,
                        tune_seconds=t_tune, fit_eval_seconds=t_fit, n_support=int(svc.n_support_.sum()),
                        kernel_offdiag_var=offdiag_variance(Ksub), kernel_offdiag_mean=offdiag_mean(Ksub),
                        reps=A.reps, entanglement=A.entanglement, device=str(device),
                        precision=A.precision, grid_size=len(grid) * len(A.C)),
                        res, **common.score_arrays(P, s_va, s_te))
                    print(f"[E2] {name}: {best_p} test PR-AUC {res['test']['pr_auc']:.4f} "
                          f"ROC {res['test']['roc_auc']:.4f} (tune {t_tune:.0f}s, fit {t_fit:.0f}s)")
