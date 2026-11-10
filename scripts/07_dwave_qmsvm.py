"""E5b (optional) - QUBO-SVM on a real D-Wave annealer, small subsets, vs SA and exact SVM.

Requirements: `pip install dwave-system` and a Leap token (`dwave config create`).
QPU time is limited: start with --n-subsets 2 --subset-size 20.
NOTE: untested by the author of this script (no QPU access); verify on a tiny run first.

  python scripts/07_dwave_qmsvm.py --d 4 --n-subsets 5 --subset-size 30 --K 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import common, metrics, store  # noqa: E402
from qmletd.qubo_svm import QMSVM, BaggedSVM  # noqa: E402

EXP = "E5_dwave"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--xi", type=float, default=1.0)
    ap.add_argument("--gamma-mult", type=float, default=1.0)
    ap.add_argument("--n-subsets", type=int, default=5)
    ap.add_argument("--subset-size", type=int, default=30)
    ap.add_argument("--num-reads", type=int, default=1000)
    ap.add_argument("--n-solutions", type=int, default=20)
    a = ap.parse_args()
    from dwave.system import DWaveSampler, EmbeddingComposite

    cache = common.DataCache()
    P = cache.get(a.split, a.d)
    gamma = a.gamma_mult / (a.d * P.X_tr.var())
    kw = dict(B=a.B, K=a.K, xi=a.xi, gamma=gamma, n_subsets=a.n_subsets, subset_size=a.subset_size,
              num_reads=a.num_reads, n_solutions=a.n_solutions, seed=a.split)
    q_hw = QMSVM(**kw, sampler=EmbeddingComposite(DWaveSampler())).fit(P.X_tr, P.y_tr)
    q_sa = QMSVM(**kw).fit(P.X_tr, P.y_tr)                     # same subsets (same seed)
    c = BaggedSVM(C=q_sa.C, gamma=gamma, subsets=q_sa.subsets_).fit(P.X_tr, P.y_tr)
    for model, m in (("qmsvm_dwave", q_hw), ("qmsvm_sa", q_sa), ("svm_bagged", c)):
        s_va, s_te = m.decision_function(P.X_va), m.decision_function(P.X_te)
        res = metrics.finalize(P.y_va, s_va, P.y_te, s_te)
        name = store.run_name(model, a.d, a.n_subsets * a.subset_size, a.split, f"K{a.K}")
        store.save(EXP, name, common.meta(P, model, a.n_subsets * a.subset_size,
                                          a.n_subsets * a.subset_size, params=kw | {"gamma": gamma}),
                   res, **common.score_arrays(P, s_va, s_te))
        print(f"[E5b] {model}: test PR-AUC {res['test']['pr_auc']:.4f} ROC {res['test']['roc_auc']:.4f}")
