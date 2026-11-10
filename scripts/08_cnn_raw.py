"""E1b - Reference ceiling: 1D CNN on the raw daily series (+ missing-value mask channel).

Trained on the ORIGINAL (unbalanced) training partition with a positive-class weight;
same splits, same F2-threshold protocol. GPU strongly recommended.

  python scripts/08_cnn_raw.py --splits 0 1 2 3 4 5 6 7 8 9
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import baselines, common, data, metrics, prep, store  # noqa: E402
from qmletd.config import get_device  # noqa: E402

EXP = "E1_cnn_raw"

if __name__ == "__main__":
    ap = common.base_parser(__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    device = get_device(a.device)
    series, mask, y = data.load_series()
    for split in a.splits:
        name = store.run_name("cnn1d_raw", "raw", None, split, a.tag)
        if store.exists(EXP, name) and not a.overwrite:
            continue
        s = prep.make_split(y, split)
        s_va, s_te, info = baselines.train_cnn(series, mask, y, s, device, epochs=a.epochs, seed=split)
        res = metrics.finalize(y[s.val], s_va, y[s.test], s_te)
        store.save(EXP, name, {"model": "cnn1d_raw", "split": split, "d": "raw", "M": "full",
                               "n_train": len(s.train), **info}, res,
                   y_val=y[s.val], s_val=s_va, y_test=y[s.test], s_test=s_te)
        print(f"[E1b] {name}: test PR-AUC {res['test']['pr_auc']:.4f} ROC {res['test']['roc_auc']:.4f}")
