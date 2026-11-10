"""Robustness of the statistical conclusions to the dependence between repeated splits.

Re-analyses the per-split results of the main (10-split) protocol. No model is re-run.

For every matched comparison (same pairs as 09_aggregate.py) it reports
  * the naive paired t-test (as in the original analysis),
  * the Nadeau-Bengio corrected resampled t-test, which inflates the variance by
    (1/J + n_test/n_train) to account for overlapping training/test sets,
  * TOST equivalence with the same corrected variance,
and three multiplicity schemes:
  family   Holm within (configuration, control, M) across d   (original analysis)
  primary  Holm over the pre-specified primary family: every quantum configuration at the
           largest d with its matched control, M = full (QUBO-SVM: its bagged setting)
  global   Holm over all matched comparisons at once (most conservative)

  python scripts/12_robust_stats.py --margin 0.02
Output: results/summary/robust_comparisons.csv and robust_report.md
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import config  # noqa: E402
from qmletd.stats import corrected_resampled_ttest, corrected_tost, holm  # noqa: E402

_spec = importlib.util.spec_from_file_location("agg", Path(__file__).with_name("09_aggregate.py"))
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)

OUT = config.RESULTS / "summary"
MATCHED_REFS = ("rbf_matched", "svm_rbf", "svm_bagged")
LATEST_TAGS = ("bwext2", "bwext")


def ratio_from_protocol():
    f_tr, _, f_te = config.SPLIT_FRACTIONS
    return f_te / f_tr


def analyse(df, margin, ratio):
    rows = []
    for exp, cfg, d, M, ref, g, r in agg.comparison_pairs(df):
        m = g.merge(r, on="split", suffixes=("", "_ref"))
        if len(m) < 3:
            continue
        naive_p = float(agg.stats.ttest_rel(m.pr_auc, m.pr_auc_ref).pvalue)
        diff, p_c, ci95 = corrected_resampled_ttest(m.pr_auc, m.pr_auc_ref, ratio)
        p_tost_c, ci90 = corrected_tost(m.pr_auc, m.pr_auc_ref, margin, ratio)
        rows.append({"experiment": exp, "config": cfg, "d": str(d), "M": str(M), "reference": ref,
                     "n_splits": len(m), "d_pr_auc": diff, "ci95_lo_corr": ci95[0],
                     "ci95_hi_corr": ci95[1], "ci90_lo_corr": ci90[0], "ci90_hi_corr": ci90[1],
                     "p_naive": naive_p, "p_corr": p_c, "p_tost_corr": p_tost_c})
    c = pd.DataFrame(rows)
    c = c[c.reference.isin(MATCHED_REFS)]
    # kernels: keep only the most recent (widest-grid) campaign, as in the paper
    for tag in LATEST_TAGS:
        if c.config.str.contains(rf"\[{tag}\]").any():
            c = c[~c.config.str.startswith("qk_") | c.config.str.contains(rf"\[{tag}\]")]
            break
    c = c.reset_index(drop=True)
    c["d_num"] = pd.to_numeric(c.d, errors="coerce")

    # family Holm (original scheme), with naive and corrected p-values
    c["family"] = c.config + "|" + c.reference + "|" + c.M
    for _, g in c.groupby("family"):
        c.loc[g.index, "p_naive_holm_family"] = holm(g.p_naive.to_numpy())
        c.loc[g.index, "p_corr_holm_family"] = holm(g.p_corr.to_numpy())
    # global Holm
    c["p_corr_holm_global"] = holm(c.p_corr.to_numpy())
    # pre-specified primary family
    dmax = c.d_num.max()
    prim = (c.d_num == dmax) & ((c.M == "full") | c.config.str.startswith("qmsvm"))
    c["primary"] = prim
    c.loc[prim, "p_corr_holm_primary"] = holm(c.loc[prim, "p_corr"].to_numpy())

    def verdict(p_adj, p_tost, diff):
        if p_adj < 0.05:
            return "Q>C" if diff > 0 else "Q<C"
        return "equivalent" if p_tost < 0.05 else "inconclusive"

    c["verdict_naive_family"] = [verdict(p, t, d) for p, t, d in
                                 zip(c.p_naive_holm_family, agg_tost_naive(c, df, margin), c.d_pr_auc)]
    c["verdict_corr_family"] = [verdict(p, t, d) for p, t, d in
                                zip(c.p_corr_holm_family, c.p_tost_corr, c.d_pr_auc)]
    c["verdict_corr_global"] = [verdict(p, t, d) for p, t, d in
                                zip(c.p_corr_holm_global, c.p_tost_corr, c.d_pr_auc)]
    return c


def agg_tost_naive(c, df, margin):
    """Naive TOST p-values, recomputed for the same rows (as in 09_aggregate)."""
    out = []
    pairs = {(str(exp), str(cfg), str(d), str(M), ref): (g, r)
             for exp, cfg, d, M, ref, g, r in agg.comparison_pairs(df)}
    for _, row in c.iterrows():
        g, r = pairs[(row.experiment, row.config, row.d, row.M, row.reference)]
        m = g.merge(r, on="split", suffixes=("", "_ref"))
        out.append(agg.tost_paired(m.pr_auc, m.pr_auc_ref, margin)[0])
    return out


def report(c, margin, ratio, path):
    def counts(col):
        return c[col].value_counts().reindex(["equivalent", "Q<C", "inconclusive", "Q>C"]).fillna(0).astype(int)

    lines = ["# Robustness of the matched comparisons to split dependence", "",
             f"Matched comparisons: {len(c)}. Equivalence margin: {margin} PR-AUC. "
             f"Nadeau-Bengio correction factor n_test/n_train = {ratio:.3f}.", "",
             "| verdict | naive t, Holm per family | corrected t, Holm per family | corrected t, Holm global |",
             "|---|---|---|---|"]
    a, b, g = counts("verdict_naive_family"), counts("verdict_corr_family"), counts("verdict_corr_global")
    for k in a.index:
        lines.append(f"| {k} | {a[k]} | {b[k]} | {g[k]} |")
    ch = c[c.verdict_naive_family != c.verdict_corr_family]
    lines += ["", f"## Comparisons whose verdict changes with the correction ({len(ch)})", ""]
    cols = ["config", "reference", "d", "M", "d_pr_auc", "ci95_lo_corr", "ci95_hi_corr",
            "verdict_naive_family", "verdict_corr_family"]
    lines.append(agg.to_markdown(ch.sort_values(["config", "M", "d_num"]), cols))
    lines += ["", "## Pre-specified primary family (largest d), corrected t, Holm within the family", ""]
    pc = ["config", "reference", "d", "M", "d_pr_auc", "ci95_lo_corr", "ci95_hi_corr",
          "ci90_lo_corr", "ci90_hi_corr", "p_corr_holm_primary", "p_tost_corr"]
    lines.append(agg.to_markdown(c[c.primary].sort_values("config"), pc))
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--runs", default=None, help="all_runs.csv (default: results/summary/all_runs.csv)")
    a = ap.parse_args()
    runs = Path(a.runs) if a.runs else OUT / "all_runs.csv"
    df = pd.read_csv(runs, dtype={"d": str, "M": str, "tag": str}, keep_default_na=False,
                     na_values={c: ["", "nan"] for c in ("pr_auc", "roc_auc", "f2")})
    df["tag"] = df["tag"].fillna("")
    ratio = ratio_from_protocol()
    c = analyse(df, a.margin, ratio)
    out = runs.parent
    c.drop(columns=["d_num"]).to_csv(out / "robust_comparisons.csv", index=False)
    report(c, a.margin, ratio, out / "robust_report.md")
    print((out / "robust_report.md").read_text())
