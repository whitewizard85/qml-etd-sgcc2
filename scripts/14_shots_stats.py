"""E4 statistics: cost of finite-shot training relative to exact gradients, paired on split.

For every (readout, d, shot budget): mean PR-AUC difference (shots - exact) over splits,
naive 95% CI and Nadeau-Bengio corrected 95% CI, and corrected TOST within +-margin.

  python scripts/14_shots_stats.py --margin 0.02
Output: <results>/summary/shots_stats.csv and shots_stats.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import config  # noqa: E402
from qmletd.stats import corrected_resampled_ttest, corrected_tost, holm, mean_ci, verdict2  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=config.EQUIVALENCE_MARGIN)
    a = ap.parse_args()
    rows = [json.loads(f.read_text()) for f in sorted((config.RESULTS / "E4_shots").glob("*.json"))]
    if not rows:
        sys.exit("no E4_shots results")
    k = pd.DataFrame([{"d": r["d"], "split": r["split"], "readout": r["readout"], "shots": str(r["shots"]),
                       "pr_auc": r["metrics"]["test"]["pr_auc"]} for r in rows])
    f_tr, _, f_te = config.SPLIT_FRACTIONS
    ratio = f_te / f_tr
    out = []
    for (ro, d), g in k.groupby(["readout", "d"]):
        ex = g[g.shots == "exact"].set_index("split").pr_auc
        for s, h in g[g.shots != "exact"].groupby("shots"):
            m = h.set_index("split").pr_auc.to_frame("q").join(ex.rename("e"), how="inner")
            if len(m) < 2:
                continue
            mu, lo, hi = mean_ci(m.q - m.e)
            _, p, ci = corrected_resampled_ttest(m.q, m.e, ratio)
            pt, _ = corrected_tost(m.q, m.e, a.margin, ratio)
            out.append({"readout": ro, "d": d, "shots": int(s), "n_splits": len(m),
                        "pr_auc_exact": m.e.mean(), "pr_auc_shots": m.q.mean(), "diff": mu,
                        "ci95_naive_lo": lo, "ci95_naive_hi": hi, "ci95_corr_lo": ci[0],
                        "ci95_corr_hi": ci[1], "p_corr": p, "p_tost_corr": pt})
    o = pd.DataFrame(out).sort_values(["readout", "d", "shots"], ascending=[True, True, False])
    # Holm over all (readout, d, shots) cells, for the difference and the equivalence tests
    o["p_corr_holm"] = holm(o.p_corr.to_numpy())
    o["p_tost_corr_holm"] = holm(o.p_tost_corr.to_numpy())
    o["verdict"] = [verdict2(p, t, d) for p, t, d in zip(o.p_corr_holm, o.p_tost_corr_holm, o["diff"])]
    dst = config.RESULTS / "summary"
    dst.mkdir(parents=True, exist_ok=True)
    o.to_csv(dst / "shots_stats.csv", index=False)
    with open(dst / "shots_stats.md", "w") as f:
        f.write(f"# Finite-shot training: PR-AUC(shots) - PR-AUC(exact), margin {a.margin}\n\n")
        try:
            f.write(o.to_markdown(index=False, floatfmt=".4f"))
        except ImportError:                     # 'tabulate' not installed
            f.write(o.to_string(index=False))
    print(o.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
