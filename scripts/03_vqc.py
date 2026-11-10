"""E2b - Variational quantum classifier, exact gradients (Adam), scaling over qubits.

For each (split, d, M, variant) the depth and learning rate are selected on validation
PR-AUC (training on a tuning subsample), then the selected model is retrained on the full
matched training set and evaluated once on test.

Variants (combine freely): --readouts local global, --losses bce mse, --reupload 0 1.
"thesis-like" variant: readout=local, loss=mse, reupload=0, entanglement=linear.

Example:
  python scripts/03_vqc.py --d 2 4 8 13 --readouts local global --losses bce --reupload 0 1
"""
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import common, metrics, store  # noqa: E402
from qmletd.config import get_device  # noqa: E402
from qmletd.vqc import predict, train_vqc  # noqa: E402

EXP = "E2_vqc"

if __name__ == "__main__":
    ap = common.base_parser(__doc__)
    ap.add_argument("--readouts", nargs="+", default=["local", "global"])
    ap.add_argument("--losses", nargs="+", default=["bce"])
    ap.add_argument("--reupload", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--entanglement", default="linear", choices=["linear", "ring", "full"])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.05, 0.01])
    ap.add_argument("--init", default="uniform", choices=["uniform", "small"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tune-epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tune-max", type=int, default=3000)
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--order", default="index", choices=["index", "mi"],
                    help="mi: qubit 0 gets the most informative feature (matters for linear circuits)")
    a = ap.parse_args()
    device = get_device(a.device)
    cache = common.DataCache()
    for split in a.splits:
        for d in a.d:
            P = cache.get(split, d)
            perm = list(range(d))
            if a.order == "mi":
                fl = list(P.feature_idx)
                perm = [fl.index(f) for f in cache._rank[split] if f in fl]
                P = P.__class__(P.X_tr[:, perm], P.y_tr, P.X_va[:, perm], P.y_va, P.X_te[:, perm],
                                P.y_te, P.feature_idx[perm], P.split, P.d)
            for M in common.parse_M(a.M):
                Xtr, ytr = cache.train_subset(P, M)
                Xtu, ytu = cache.tune_subset(Xtr, ytr, a.tune_max, split)
                if Xtu is None:
                    Xtu, ytu = Xtr, ytr
                for readout, loss, re in itertools.product(a.readouts, a.losses, a.reupload):
                    tag = f"{readout}_{loss}_re{re}_{a.entanglement}_{a.init}" + ("_mi" if a.order == "mi" else "") \
                        + (f"_{a.tag}" if a.tag else "")
                    name = store.run_name("vqc", d, M, split, tag)
                    if store.exists(EXP, name) and not a.overwrite:
                        continue
                    t0 = time.time()
                    best = (-1, None)
                    for depth, lr in itertools.product(a.depths, a.lrs):
                        _, info = train_vqc(Xtu, ytu, P.X_va, P.y_va, d=d, depth=depth,
                                            entanglement=a.entanglement, readout=readout,
                                            reupload=bool(re), loss=loss, lr=lr,
                                            epochs=a.tune_epochs, batch=a.batch, init=a.init,
                                            seed=split, device=device, input_scale=a.input_scale)
                        if info["best_val_pr_auc"] > best[0]:
                            best = (info["best_val_pr_auc"], {"depth": depth, "lr": lr})
                    t_tune = time.time() - t0
                    model, info = train_vqc(Xtr, ytr, P.X_va, P.y_va, d=d, depth=best[1]["depth"],
                                            entanglement=a.entanglement, readout=readout,
                                            reupload=bool(re), loss=loss, lr=best[1]["lr"],
                                            epochs=a.epochs, batch=a.batch, init=a.init,
                                            seed=split, device=device, input_scale=a.input_scale)
                    s_va = predict(model, P.X_va * a.input_scale, device)
                    s_te = predict(model, P.X_te * a.input_scale, device)
                    res = metrics.finalize(P.y_va, s_va, P.y_te, s_te)
                    store.save(EXP, name, common.meta(
                        P, "vqc", M, len(ytr), readout=readout, loss=loss, reupload=bool(re),
                        entanglement=a.entanglement, init=a.init, params=best[1], qubit_order=a.order,
                        val_pr_auc_selection=best[0], tune_seconds=t_tune, device=str(device),
                        **{k: v for k, v in info.items()}),
                        res, **common.score_arrays(P, s_va, s_te))
                    print(f"[E2b] {name}: {best[1]} test PR-AUC {res['test']['pr_auc']:.4f} "
                          f"ROC {res['test']['roc_auc']:.4f} pos_rate {res['test']['pos_rate']:.2f} "
                          f"(tune {t_tune:.0f}s, train {info['train_seconds']:.0f}s)")
