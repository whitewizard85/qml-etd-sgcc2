"""TDD2022 checks requested in review.

1. Per held-out consumer type: PR-AUC of every primary-family configuration and of its control, and the
   difference, computed separately on the test days of each consumer type (the test set holds whole types).
   With only a handful of held-out types this is descriptive: it shows whether the sign of the difference is
   consistent across types, which the day-level bootstrap cannot show.
2. Temporal structure of the raw file: each consumer type has 35,040 hourly rows = 4 x 8,760. For each type we
   compare the four 8,760-hour segments on days that are normal in both segments: correlation of daily totals
   with segment 1, and the share of days that are exactly identical.

  QMLETD_DATASET=tdd2022 QMLETD_PROTOCOL=holdout python scripts/16_tdd_checks.py
Output: <results>/summary/tdd_by_type.csv, tdd_segments.csv, tdd_checks.md
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import config, data, prep, tdd  # noqa: E402

_spec = importlib.util.spec_from_file_location("ho", Path(__file__).with_name("13_holdout_eval.py"))
ho = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ho)
OUT = config.RESULTS / "summary"


def by_type():
    comp = pd.read_csv(OUT / "holdout_comparisons.csv", dtype={"d": str, "M": str})
    runs = ho.agg.load_runs()
    _, y = data.load_features()
    groups = prep._load_groups(len(y))
    _, te = prep.holdout_test(y, groups)
    ctype = np.load(config.DATA_PROCESSED / "consumer_type.npy", allow_pickle=True)[te]
    rows = []
    for _, r in comp[comp.primary].iterrows():
        pairs = [p for p in ho.agg.comparison_pairs(runs)
                 if p[1] == r.config and str(p[2]) == r.d and str(p[3]) == r.M and p[4] == r.reference]
        if not pairs:
            continue
        exp, cfg, d, M, ref, g, c = pairs[0]
        q, k = ho.ensemble(g), ho.ensemble(c)
        if not np.array_equal(q["y_test"], y[te]):
            raise RuntimeError("test indices do not match the stored predictions")
        for t in sorted(set(ctype)):
            sel = ctype == t
            yt = q["y_test"][sel]
            if yt.min() == yt.max():
                continue
            a, b = average_precision_score(yt, q["s_test"][sel]), average_precision_score(yt, k["s_test"][sel])
            rows.append({"config": cfg, "reference": ref, "d": d, "M": M, "consumer_type": t,
                         "n_days": int(sel.sum()), "prevalence": float(yt.mean()),
                         "pr_auc_q": a, "pr_auc_ref": b, "diff": a - b})
    return pd.DataFrame(rows)


def segments():
    df = tdd.read_table(config.TDD_FILE)
    det = tdd.detect_columns(df)
    df = tdd.drop_invalid_types(df, det["type"])
    rows = []
    for t, d in df.groupby(det["type"], sort=False):
        e = pd.to_numeric(d[det["elec"]], errors="coerce").to_numpy(float)
        lab = tdd.binary_label(d[det["label"]])
        n_seg = len(e) // 8760
        days = e[: n_seg * 8760].reshape(n_seg, 365, 24)
        normal = (lab[: n_seg * 8760].reshape(n_seg, 365, 24) == 0).all(axis=2)
        for k in range(1, n_seg):
            both = normal[0] & normal[k]
            tot0, totk = days[0][both].sum(axis=1), days[k][both].sum(axis=1)
            same = np.all(np.isclose(days[0][both], days[k][both], rtol=0, atol=1e-9), axis=1)
            rows.append({"consumer_type": t, "segment": k + 1, "days_normal_in_both": int(both.sum()),
                         "corr_daily_total_vs_seg1": float(np.corrcoef(tot0, totk)[0, 1]),
                         "share_identical_days": float(same.mean()),
                         "mean_ratio_vs_seg1": float(totk.mean() / tot0.mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if config.DATASET != "tdd2022" or config.PROTOCOL != "holdout":
        sys.exit("set QMLETD_DATASET=tdd2022 QMLETD_PROTOCOL=holdout")
    bt = by_type()
    bt.to_csv(OUT / "tdd_by_type.csv", index=False)
    sg = segments()
    sg.to_csv(OUT / "tdd_segments.csv", index=False)
    with open(OUT / "tdd_checks.md", "w") as f:
        f.write("# TDD2022 checks\n\n## Primary family by held-out consumer type (difference q - control)\n\n")
        piv = bt.pivot_table(index="config", columns="consumer_type", values="diff")
        f.write(piv.round(4).to_string() + "\n\n")
        f.write("## Control PR-AUC by held-out consumer type\n\n")
        f.write(bt.pivot_table(index="reference", columns="consumer_type", values="pr_auc_ref").round(4).to_string())
        f.write("\n\n## Segments of 8,760 hours vs segment 1 (days normal in both)\n\n")
        f.write(sg.round(4).to_string(index=False) + "\n")
    print((OUT / "tdd_checks.md").read_text())
