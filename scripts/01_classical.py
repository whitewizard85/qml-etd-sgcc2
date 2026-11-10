"""E1 - Classical baselines on exactly the same features / training subsets as the quantum models.

Example:
  python scripts/01_classical.py --models logreg svm_rbf rf lgbm mlp --d 2 4 8 13 --M 150 1040 full
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import baselines, common, metrics, store  # noqa: E402
from qmletd.config import seed_everything  # noqa: E402

EXP = "E1_classical"

if __name__ == "__main__":
    ap = common.base_parser(__doc__)
    ap.add_argument("--models", nargs="+", default=["logreg", "svm_rbf", "rf", "lgbm", "mlp"])
    ap.add_argument("--tune-max", type=int, default=4000,
                    help="hyper-parameter search on a subsample of this size, then refit")
    a = ap.parse_args()
    cache = common.DataCache()
    for split in a.splits:
        for d in a.d:
            P = cache.get(split, d)
            for M in common.parse_M(a.M):
                Xtr, ytr = cache.train_subset(P, M)
                Xtu, ytu = cache.tune_subset(Xtr, ytr, a.tune_max, split)
                for model in a.models:
                    name = store.run_name(model, d, M, split, a.tag)
                    if store.exists(EXP, name) and not a.overwrite:
                        continue
                    seed_everything(split)
                    t0 = time.time()
                    m, info = baselines.fit_select(model, Xtr, ytr, P.X_va, P.y_va, split, Xtu, ytu)
                    fit_s = time.time() - t0
                    s_va, s_te = baselines.score(m, P.X_va), baselines.score(m, P.X_te)
                    res = metrics.finalize(P.y_va, s_va, P.y_te, s_te)
                    store.save(EXP, name, common.meta(P, model, M, len(ytr), total_seconds=fit_s, **info),
                               res, **common.score_arrays(P, s_va, s_te))
                    print(f"[E1] {name}: test PR-AUC {res['test']['pr_auc']:.4f} "
                          f"ROC {res['test']['roc_auc']:.4f} F2 {res['test']['f2']:.4f} ({fit_s:.1f}s)")
