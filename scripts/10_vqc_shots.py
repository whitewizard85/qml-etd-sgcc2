"""E4 - VQC trained with finite-shot parameter-shift gradients vs exact gradients.

Question: the global-readout VQC trains well in E2b despite gradient variance ~2^-d
(barren plateau). Is that only because gradients there are exact? Here the same model is
trained with shot-noise gradient estimates (S shots per circuit) and the gradient quality
(cosine with the exact gradient, relative error) is logged at every step.

Same hyper-parameters for every cell (no tuning, to isolate the effect of shots):
depth 2, batch 64, Adam lr 0.05, 300 steps, model selection on validation PR-AUC every 25 steps,
evaluation with exact expectation values. Qubit 0 gets the most informative feature.

  python scripts/10_vqc_shots.py --d 4 8 10 13 --readouts global local_mean \
         --shots exact 10000 1000 100 --splits 0 1 2 --device cuda
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import common, config, metrics, prep, store  # noqa: E402
from qmletd.config import get_device  # noqa: E402
from qmletd.shots import train_vqc_shots  # noqa: E402
from qmletd.vqc import predict  # noqa: E402

EXP = "E4_shots"

if __name__ == "__main__":
    ap = common.base_parser(__doc__)
    ap.add_argument("--readouts", nargs="+", default=["global", "local_mean"])
    ap.add_argument("--shots", nargs="+", default=["exact", "10000", "1000", "100"])
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--train-max", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=4096, help="rows per simulator call (lower if GPU OOM)")
    a = ap.parse_args()
    if a.splits is config.DEFAULT_SPLITS:   # not given on the command line
        a.splits = [0, 1, 2]
    device = get_device(a.device)
    cache = common.DataCache()
    for split, d in itertools.product(a.splits, a.d):
        P = cache.get(split, d)
        fl = list(P.feature_idx)
        perm = [fl.index(f) for f in cache._rank[split] if f in fl]          # MI order
        Xtr, ytr = prep.stratified_subsample(P.X_tr[:, perm], P.y_tr, a.train_max, seed=split + 7)
        Xva, Xte = P.X_va[:, perm], P.X_te[:, perm]
        for readout, s in itertools.product(a.readouts, a.shots):
            shots = None if s == "exact" else int(s)
            tag = f"{readout}_S{s}_{a.optimizer}_L{a.depth}"
            name = store.run_name("vqc_shots", d, a.train_max, split, tag)
            if store.exists(EXP, name) and not a.overwrite:
                continue
            m, info = train_vqc_shots(Xtr, ytr, Xva, P.y_va, d=d, depth=a.depth, readout=readout,
                                      shots=shots, steps=a.steps, batch=a.batch, lr=a.lr,
                                      optimizer=a.optimizer, seed=split, device=device, chunk=a.chunk)
            s_va, s_te = predict(m, Xva, device), predict(m, Xte, device)
            res = metrics.finalize(P.y_va, s_va, P.y_te, s_te)
            store.save(EXP, name, common.meta(P, "vqc_shots", a.train_max, len(ytr), readout=readout,
                                              shots=s, depth=a.depth, optimizer=a.optimizer, lr=a.lr,
                                              steps=a.steps, batch=a.batch, qubit_order="mi", **info),
                       res, **common.score_arrays(P, s_va, s_te))
            print(f"[E4] {name}: test PR-AUC {res['test']['pr_auc']:.4f} ROC {res['test']['roc_auc']:.4f} "
                  f"cos(g,g*) {info['cos_mean_all']:.3f} rel.err {info['rel_err_median_all']:.2f} "
                  f"({info['train_seconds']:.0f}s)")
