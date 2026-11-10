"""E2c - QUBO-SVM (Willsch et al.) with bagging, solved by seeded simulated annealing,
versus the EXACT SVM on the same subsets with the same kernel, C and aggregation.

The pair (qmsvm, svm_bagged) shares: subsets, gamma, C = sum_k B^k, decision averaging.
Any difference is due to the QUBO encoding + ensemble of low-energy solutions.

Example:
  python scripts/04_qmsvm.py --d 2 4 8 13 --n-subsets 20 --subset-size 80
"""
import itertools
import sys
import time
from pathlib import Path

from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import common, metrics, store  # noqa: E402
from qmletd.qubo_svm import QMSVM, BaggedSVM  # noqa: E402

EXP = "E2_qmsvm"

if __name__ == "__main__":
    ap = common.base_parser(__doc__)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--K", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--xi", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--gamma-mult", type=float, nargs="+", default=[0.3, 1.0, 3.0],
                    help="gamma = mult / (d * var(X))")
    ap.add_argument("--n-subsets", type=int, default=20)
    ap.add_argument("--subset-size", type=int, default=80)
    ap.add_argument("--num-reads", type=int, default=200)
    ap.add_argument("--num-sweeps", type=int, default=1000)
    ap.add_argument("--n-solutions", type=int, default=20)
    ap.add_argument("--tune-subsets", type=int, default=5, help="subsets used during tuning")
    a = ap.parse_args()
    a.M = ["full"]  # subsets are drawn from the full balanced training set
    cache = common.DataCache()
    for split in a.splits:
        for d in a.d:
            P = cache.get(split, d)
            Xtr, ytr = P.X_tr, P.y_tr
            g0 = float(1.0 / (d * Xtr.var()))
            tag = f"S{a.n_subsets}_m{a.subset_size}" + (f"_{a.tag}" if a.tag else "")
            M_eff = a.n_subsets * a.subset_size
            n_q, n_c = (store.run_name(m, d, M_eff, split, tag) for m in ("qmsvm", "svm_bagged"))
            if store.exists(EXP, n_q) and store.exists(EXP, n_c) and not a.overwrite:
                continue
            # ---- tuning (cheap: few subsets)
            t0 = time.time()
            best = (-1, None)
            for K, xi, gm in itertools.product(a.K, a.xi, a.gamma_mult):
                q = QMSVM(B=a.B, K=K, xi=xi, gamma=gm * g0, n_subsets=a.tune_subsets,
                          subset_size=a.subset_size, num_reads=a.num_reads, num_sweeps=a.num_sweeps,
                          n_solutions=a.n_solutions, seed=split).fit(Xtr, ytr)
                ap_ = average_precision_score(P.y_va, q.decision_function(P.X_va))
                if ap_ > best[0]:
                    best = (ap_, {"K": K, "xi": xi, "gamma": gm * g0, "gamma_mult": gm})
            t_tune = time.time() - t0
            p = best[1]
            q = QMSVM(B=a.B, K=p["K"], xi=p["xi"], gamma=p["gamma"], n_subsets=a.n_subsets,
                      subset_size=a.subset_size, num_reads=a.num_reads, num_sweeps=a.num_sweeps,
                      n_solutions=a.n_solutions, seed=split).fit(Xtr, ytr)
            c = BaggedSVM(C=q.C, gamma=p["gamma"], subsets=q.subsets_).fit(Xtr, ytr)
            for model, name, m in (("qmsvm", n_q, q), ("svm_bagged", n_c, c)):
                s_va, s_te = m.decision_function(P.X_va), m.decision_function(P.X_te)
                res = metrics.finalize(P.y_va, s_va, P.y_te, s_te)
                store.save(EXP, name, common.meta(
                    P, model, M_eff, M_eff, params={**p, "B": a.B, "C": q.C},
                    n_subsets=a.n_subsets, subset_size=a.subset_size, num_reads=a.num_reads,
                    num_sweeps=a.num_sweeps, n_solutions=a.n_solutions, qubo_vars=a.subset_size * p["K"],
                    val_pr_auc_selection=best[0], tune_seconds=t_tune, fit_seconds=m.fit_seconds_),
                    res, **common.score_arrays(P, s_va, s_te))
                print(f"[E2c] {name}: {p} test PR-AUC {res['test']['pr_auc']:.4f} "
                      f"ROC {res['test']['roc_auc']:.4f} (fit {m.fit_seconds_:.1f}s)")
