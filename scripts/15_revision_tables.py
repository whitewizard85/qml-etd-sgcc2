"""LaTeX tables for the revision: primary family on both hold-out test sets, finite-shot over 10 splits.

  python scripts/15_revision_tables.py --sgcc results_sgcc_holdout/summary \
      --tdd results_tdd2022_holdout/summary --shots results/summary/shots_stats.csv --out paper/tables
"""
import argparse
from pathlib import Path

import pandas as pd

NAMES = {"qk_fidelity": "Fidelity kernel", "qk_projected": "Projected kernel", "qmsvm[S20_m80]": "QUBO-SVM (SA)",
         "vqc[global_bce_re0_linear_uniform]": "VQC global",
         "vqc[global_bce_re1_linear_uniform]": "VQC global, re-upl.",
         "vqc[local_bce_re0_linear_uniform]": "VQC local",
         "vqc[local_bce_re1_linear_uniform]": "VQC local, re-upl.",
         "vqc[local_bce_re0_linear_uniform_mi]": "VQC local, MI",
         "vqc[local_bce_re1_linear_uniform_mi]": "VQC local, re-upl., MI",
         "vqc[local_mean_bce_re0_linear_uniform_mi]": "VQC local-mean, MI",
         "vqc[local_mean_bce_re1_linear_uniform_mi]": "VQC local-mean, re-upl., MI"}
REF = {"rbf_matched": "RBF", "svm_bagged": "exact SVM", "svm_rbf": "SVM RBF"}


def verdict(r):
    lab = {"Q<C": "worse", "Q>C": "better", "equivalent": "equivalent", "equivalent*": "equiv.$^{\\dagger}$",
           "inconclusive": "inconcl."}
    return lab[r.verdict]


def primary(path):
    c = pd.read_csv(Path(path) / "holdout_comparisons.csv", dtype={"d": str, "M": str})
    c = c[c.primary].copy()
    c["key"] = c.config.str.replace(r"\[bwext2\]", "", regex=True)
    return c.set_index("key")


def table_holdout(sgcc, tdd, out):
    a, b = primary(sgcc), primary(tdd)
    lines = [r"\begin{tabular}{lrlrl}", r"\toprule",
             r" & \multicolumn{2}{c}{SGCC} & \multicolumn{2}{c}{TDD2022} \\",
             r"Model & $\Delta$ [95\% CI] & Verdict & $\Delta$ [95\% CI] & Verdict \\", r"\midrule"]
    for k, name in NAMES.items():
        if k not in a.index or k not in b.index:
            continue
        x, y = a.loc[k], b.loc[k]
        cell = lambda r: f"{r.d_pr_auc:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] & {verdict(r)}"
        lines.append(f"{name} & {cell(x)} & {cell(y)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_holdout.tex").write_text("\n".join(lines) + "\n")


def table_shots(path, out):
    s = pd.read_csv(path)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Readout & $d$ & exact & $S=10^4$ & $S=10^3$ & $S=10^2$ \\", r"\midrule"]
    for (ro, d), g in s.groupby(["readout", "d"]):
        g = g.set_index("shots")
        cells = []
        for S in (10000, 1000, 100):
            r = g.loc[S]
            p_eq = r.p_tost_corr_holm if "p_tost_corr_holm" in r.index else r.p_tost_corr
            mark = r"$^{\mathrm{a}}$" if p_eq < 0.05 else ""
            cells.append(f"{r['diff']:+.3f}{mark}")
        lab = "global" if ro == "global" else "local mean"
        lines.append(f"{lab} & {d} & {g.pr_auc_exact.iloc[0]:.3f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_shots.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgcc", required=True)
    ap.add_argument("--tdd", required=True)
    ap.add_argument("--shots", required=True)
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table_holdout(a.sgcc, a.tdd, out)
    table_shots(a.shots, out)
    print(f"tables -> {out}")
