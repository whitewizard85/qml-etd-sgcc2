"""Generate LaTeX tables and paper figures from results/summary/*.csv.

Every number in the paper tables comes from here: re-run after any new experiment.
  python scripts/11_paper_assets.py --summary results/summary --out paper
writes paper/tables/*.tex and paper/figures/*.pdf (copies + the re-plotted finite-shot figure).
"""
import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DS = [2, 4, 6, 8, 10, 13]


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def ci(r):
    return f"{r.pr_auc:.3f} [{r.pr_auc_lo:.3f}, {r.pr_auc_hi:.3f}]"


def prefer_tag(df, col, tags=("bwext2", "bwext")):
    """Use the most recent extended-grid runs of the kernel models (and of their RBF control)."""
    df = df.copy()
    for base in ("qk_fidelity", "qk_projected", "rbf_matched"):
        for tag in tags:
            if (df[col] == f"{base}[{tag}]").any():
                drop = [base] + [f"{base}[{t}]" for t in tags if t != tag]
                df = df[~df[col].isin(drop)]
                df.loc[df[col] == f"{base}[{tag}]", col] = base
                break
    return df


def table_main(s, out):
    s = prefer_tag(s, "config")
    s["dn"] = pd.to_numeric(s.d, errors="coerce")
    rows = [
        ("1D CNN, raw daily series", s[s.config == "cnn1d_raw"]),
        ("LightGBM", s[(s.config == "lgbm") & (s.dn == 13) & (s.M == "full")]),
        ("Random forest", s[(s.config == "rf") & (s.dn == 13) & (s.M == "full")]),
        ("SVM, RBF kernel", s[(s.config == "svm_rbf") & (s.dn == 13) & (s.M == "full")]),
        ("SVM, RBF (precomputed, matched)", s[(s.config == "rbf_matched") & (s.dn == 13) & (s.M == "full")]),
        ("Fidelity quantum kernel (ZZ)", s[(s.config == "qk_fidelity") & (s.dn == 13) & (s.M == "full")]),
        ("Projected quantum kernel", s[(s.config == "qk_projected") & (s.dn == 13) & (s.M == "full")]),
        ("VQC, global readout", s[(s.config == "vqc[global_bce_re0_linear_uniform]") & (s.dn == 13)]),
        ("VQC, local readout (as in thesis)", s[(s.config == "vqc[local_bce_re0_linear_uniform]") & (s.dn == 13)]),
        ("VQC, local-mean readout, re-upload", s[(s.config == "vqc[local_mean_bce_re1_linear_uniform_mi]") & (s.dn == 13)]),
        ("SVM, exact, bagged (20$\\times$80)", s[(s.config == "svm_bagged[S20_m80]") & (s.dn == 13)]),
        ("QUBO-SVM, SA, bagged (20$\\times$80)", s[(s.config == "qmsvm[S20_m80]") & (s.dn == 13)]),
    ]
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & PR-AUC [95\% CI] & ROC-AUC & $F_2$ & MCC & P@5\% \\", r"\midrule"]
    for i, (name, g) in enumerate(rows):
        if not len(g):
            continue
        r = g.iloc[0]
        lines.append(f"{name} & {ci(r)} & {f3(r.roc_auc)} & {f3(r.f2)} & {f3(r.mcc)} & {f3(r.precision_at_5pct)} \\\\")
        if i in (0, 4, 6, 9):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_main.tex").write_text("\n".join(lines) + "\n")


def table_matched(c, out):
    c = prefer_tag(c, "config")
    c["dn"] = c.d.astype(int)
    fams = [
        ("Fidelity kernel", "qk_fidelity", "rbf_matched", "full"),
        ("Fidelity kernel", "qk_fidelity", "rbf_matched", "1040"),
        ("Fidelity kernel", "qk_fidelity", "rbf_matched", "150"),
        ("Projected kernel", "qk_projected", "rbf_matched", "full"),
        ("Projected kernel", "qk_projected", "rbf_matched", "1040"),
        ("Projected kernel", "qk_projected", "rbf_matched", "150"),
        ("QUBO-SVM (SA)", "qmsvm[S20_m80]", "svm_bagged", "1600"),
        ("VQC global", "vqc[global_bce_re0_linear_uniform]", "svm_rbf", "full"),
        ("VQC local, index order", "vqc[local_bce_re0_linear_uniform]", "svm_rbf", "full"),
        ("VQC local, MI order", "vqc[local_bce_re0_linear_uniform_mi]", "svm_rbf", "full"),
        ("VQC local-mean, re-upl., MI", "vqc[local_mean_bce_re1_linear_uniform_mi]", "svm_rbf", "full"),
    ]
    # Springer style: asterisk = significant (direction given by the sign); superscript a = equivalent
    mark = {"Q>C": r"$^{*}$", "Q<C": r"$^{*}$", "equivalent": r"$^{\mathrm{a}}$", "inconclusive": ""}
    lines = [r"\begin{tabular}{llc" + "r" * len(DS) + "}", r"\toprule",
             r"Model & Control & $M$ & " + " & ".join(f"$d={d}$" for d in DS) + r" \\", r"\midrule"]
    ref_name = {"rbf_matched": "RBF", "svm_bagged": "exact SVM", "svm_rbf": "SVM RBF"}
    for name, cfg, ref, M in fams:
        g = c[(c.config == cfg) & (c.reference == ref) & (c.M.astype(str) == M)]
        if not len(g):
            continue
        cells = []
        for d in DS:
            h = g[g.dn == d]
            if True:
                v = 0.0 if not len(h) else round(float(h.iloc[0].d_pr_auc), 3) + 0.0
                cells.append("--" if not len(h) else f"{v:+.3f}{mark[h.iloc[0].verdict]}")
        Mlab = "all" if M == "full" else M
        lines.append(f"{name} & {ref_name[ref]} & {Mlab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_matched.tex").write_text("\n".join(lines) + "\n")


def table_gradvar(gf, out):
    lab = {"local": "local $\\langle Z_0\\rangle$", "local_mean": "local mean $\\frac1d\\sum_q\\langle Z_q\\rangle$",
           "global": "global $\\langle Z^{\\otimes d}\\rangle$"}
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Readout & $L=1$ & $L=2$ & $L=4$ & $L=8$ \\", r"\midrule"]
    for ro in ("local", "local_mean", "global"):
        g = gf[gf.readout == ro].set_index("depth")
        if not len(g):
            continue
        lines.append(f"{lab[ro]} & " + " & ".join(f"{g.loc[L].decay_factor_per_qubit:.2f}" if L in g.index else "--"
                                                  for L in (1, 2, 4, 8)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_gradvar.tex").write_text("\n".join(lines) + "\n")


def shots_frame(runs):
    s = runs[runs.experiment == "E4_shots"].copy()
    if not len(s):
        return None
    s["readout"] = s.tag.str.split("_S").str[0]
    s["shots"] = s.tag.str.extract(r"_S(\w+?)_")[0]
    s["d"] = s.d.astype(int)
    return s


def table_shots(runs, shots_csv, out):
    s = shots_frame(runs)
    if s is None:
        return
    p = s.pivot_table(index=["readout", "d", "split"], columns="shots", values="pr_auc")
    agg = shots_csv.copy()
    agg["shots"] = agg.shots.astype(str)
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & & & \multicolumn{3}{c}{$\Delta$PR-AUC vs.\ exact (mean over splits)} & \\",
             r"\cmidrule(lr){4-6}",
             r"Readout & $d$ & PR-AUC exact & $10^4$ shots & $10^3$ & $10^2$ & cos$(\hat g,g)$ at $10^2$ \\",
             r"\midrule"]
    for ro in ("global", "local_mean"):
        for d in sorted(s[s.readout == ro].d.unique()):
            q = p.loc[(ro, d)]
            ex = q["exact"].mean()
            diffs = [(q[k] - q["exact"]).mean() for k in ("10000", "1000", "100")]
            cs = agg[(agg.readout == ro) & (agg.d == d) & (agg.shots == "100")].cos
            name = "global" if ro == "global" else "local mean"
            lines.append(f"{name} & {d} & {ex:.3f} & " + " & ".join(f"{v:+.3f}" for v in diffs)
                         + f" & {cs.iloc[0]:.2f} \\\\")
        if ro == "global":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_shots.tex").write_text("\n".join(lines) + "\n")


def fig_shots(runs, shots_csv, out):
    s = shots_frame(runs)
    if s is None:
        return
    order = ["exact", "10000", "1000", "100"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), sharex=True, squeeze=False)
    for j, ro in enumerate(("global", "local_mean")):
        for i, sh in enumerate(order):
            g = s[(s.readout == ro) & (s.shots == sh)].groupby("d").pr_auc.agg(["mean", "min", "max"]).reset_index()
            col = "#8a8980" if sh == "exact" else SEQ[i]
            off = (i - 1.5) * 0.12
            lab = "exact gradient" if sh == "exact" else f"{int(sh):,} shots"
            axes[0][j].errorbar(g.d + off, g["mean"], yerr=[g["mean"] - g["min"], g["max"] - g["mean"]],
                                color=col, marker="o", ms=4, lw=1.8, capsize=2, elinewidth=1,
                                ls=":" if sh == "exact" else "-", label=lab)
            a = shots_csv[(shots_csv.readout == ro) & (shots_csv.shots.astype(str) == sh)].sort_values("d")
            axes[1][j].plot(a.d, a.cos, color=col, marker="o", ms=4, lw=1.8, ls=":" if sh == "exact" else "-")
        axes[0][j].set_ylim(0.0, 0.30)
        axes[0][j].axhline(0.0853, color="#c3c2b7", lw=1)
        axes[0][j].set_title("global readout" if ro == "global" else "local-mean readout", fontsize=9)
        axes[1][j].set_ylim(0, 1.05)
        axes[1][j].set_xlabel("qubits $d$")
        for ax in (axes[0][j], axes[1][j]):
            ax.grid(True, color="#e6e5e0", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("test PR-AUC (mean, min-max)")
    axes[1][0].set_ylabel(r"cos(estimated, exact gradient)")
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig6_finite_shots.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    S, O = Path(a.summary), Path(a.out)
    (O / "tables").mkdir(parents=True, exist_ok=True)
    (O / "figures").mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(S / "summary.csv", dtype={"d": str, "M": str})
    comp = pd.read_csv(S / "comparisons.csv", dtype={"M": str})
    runs = pd.read_csv(S / "all_runs.csv", dtype={"d": str, "M": str}, low_memory=False)
    table_main(summ, O / "tables")
    table_matched(comp, O / "tables")
    if (S / "gradvar_fit.csv").exists():
        table_gradvar(pd.read_csv(S / "gradvar_fit.csv"), O / "tables")
    if (S / "shots_summary.csv").exists():
        sc = pd.read_csv(S / "shots_summary.csv", dtype={"shots": str})
        table_shots(runs, sc, O / "tables")
        fig_shots(runs, sc, O / "figures")
    for f in ("fig1_prauc_vs_d", "fig2_prauc_vs_M", "fig3_gradient_variance", "fig4_kernel_diagnostics",
              "fig5_precision_at_budget"):
        if (S / f"{f}.pdf").exists():
            shutil.copy(S / f"{f}.pdf", O / "figures" / f"{f}.pdf")
    print(f"tables -> {O / 'tables'}, figures -> {O / 'figures'}")
