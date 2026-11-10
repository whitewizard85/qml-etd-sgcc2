"""Aggregate all results: summary tables with 95% CIs, paired statistical comparisons,
paper figures. Output in results/summary/.

  python scripts/09_aggregate.py --margin 0.02 --bootstrap 1000

Comparisons (primary metric PR-AUC, paired on split):
  qk_fidelity / qk_projected  vs rbf_matched     (same d, same M)       [E2_qkernel]
  vqc (each variant)          vs svm_rbf (E1)    (same d, same M)
  qmsvm                       vs svm_bagged      (same d, subsets)      [E2_qmsvm]
  every quantum config        vs lgbm (E1)       (same d, M = full)     strongest tabular baseline
Per comparison: mean difference + 95% CI over splits, Wilcoxon signed-rank (Holm-corrected
over the whole family), TOST equivalence within +-margin, one-sided non-superiority test,
per-split DeLong (ROC-AUC) and stratified paired bootstrap (PR-AUC) on the test set.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import config  # noqa: E402
from qmletd.stats import (delong, holm, mean_ci, noninferiority_upper,  # noqa: E402
                          paired_bootstrap, tost_paired, wilcoxon)

OUT = config.RESULTS / "summary"
PRED_EXPS = ["E1_classical", "E1_cnn_raw", "E2_qkernel", "E2_vqc", "E2_qmsvm", "E4_shots", "E5_dwave"]
METRICS = ["pr_auc", "roc_auc", "f2", "mcc", "balanced_acc", "precision", "recall", "pos_rate",
           "precision_at_1pct", "precision_at_5pct", "precision_at_10pct", "recall_at_5pct", "prevalence"]
# Fixed categorical order (colour follows the entity, never its rank).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ENTITY_COLOR = {"svm_rbf": PALETTE[0], "lgbm": PALETTE[1], "rbf_matched": PALETTE[2],
                "qk_fidelity": PALETTE[3], "qk_projected": PALETTE[4], "vqc": PALETTE[5],
                "qmsvm": PALETTE[6], "svm_bagged": PALETTE[7]}
SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]      # one-hue ordinal ramp (blue)


def seq_color(k, n):
    """Spread n ordinal levels over the whole light->dark ramp."""
    return SEQ[int(round(k * (len(SEQ) - 1) / max(1, n - 1)))] if n <= len(SEQ) else SEQ[min(k, len(SEQ) - 1)]
FIG_BWS = (0.001, 0.01, 0.1, 1.0)   # bandwidths shown in the kernel-diagnostics figure (one-hue ramp, 4 levels)
LATEST_TAGS = ("bwext2", "bwext")   # most recent extended-grid runs of the kernel models, preferred in figures
QUANTUM = ("qk_fidelity", "qk_projected", "vqc", "qmsvm", "qmsvm_dwave", "qmsvm_sa")  # vqc_shots: own figure


def parse_name(name):
    parts = name.split("__")
    return {"model": parts[0], "d": parts[1][1:], "M": parts[2][1:], "split": int(parts[3][1:]),
            "tag": "__".join(parts[4:])}


def load_runs():
    rows = []
    for exp in PRED_EXPS:
        for f in sorted((config.RESULTS / exp).glob("*.json")):
            j = json.loads(f.read_text())
            if not j.get("metrics"):
                continue
            r = {"experiment": exp, "name": j["name"], "file": str(f.with_suffix(".npz")),
                 **parse_name(j["name"]), "val_pr_auc": j["metrics"]["val"]["pr_auc"]}
            r.update({m: j["metrics"]["test"].get(m, np.nan) for m in METRICS})
            r["config"] = r["model"] + (f"[{r['tag']}]" if r["tag"] else "")
            rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["d_num"] = pd.to_numeric(df["d"], errors="coerce")
    return df


def summary_table(df):
    out = []
    for (exp, cfg, d, M), g in df.groupby(["experiment", "config", "d", "M"]):
        r = {"experiment": exp, "config": cfg, "d": d, "M": M, "n_splits": len(g)}
        for m in METRICS + ["val_pr_auc"]:
            mu, lo, hi = mean_ci(g[m])
            r[m], r[f"{m}_lo"], r[f"{m}_hi"] = mu, lo, hi
        out.append(r)
    return pd.DataFrame(out)


def comparison_pairs(df):
    pairs = []
    q = df[df.model.isin(QUANTUM)]
    for (exp, cfg, d, M), g in q.groupby(["experiment", "config", "d", "M"]):
        model, tag = g.model.iloc[0], g.tag.iloc[0]
        refs = []
        if model in ("qk_fidelity", "qk_projected"):
            r = df[(df.experiment == exp) & (df.model == "rbf_matched") & (df.d == d) & (df.M == M)]
            same = r[r.tag == tag]                     # control tuned with the same (extended) grids
            refs.append(("rbf_matched", same if len(same) else r[r.tag == ""]))
        if model == "vqc":
            refs.append(("svm_rbf", df[(df.experiment == "E1_classical") & (df.model == "svm_rbf")
                                       & (df.d == d) & (df.M == M)]))
        if model.startswith("qmsvm"):
            refs.append(("svm_bagged", df[(df.experiment == exp) & (df.model == "svm_bagged")
                                          & (df.d == d) & (df.M == M) & (df.tag == tag)]))
        refs.append(("lgbm_full", df[(df.experiment == "E1_classical") & (df.model == "lgbm")
                                     & (df.d == d) & (df.M == "full")]))
        for ref_name, r in refs:
            if len(r):
                pairs.append((exp, cfg, d, M, ref_name, g, r))
    return pairs


def compare(df, margin, n_boot):
    rows = []
    for exp, cfg, d, M, ref, g, r in comparison_pairs(df):
        m = g.merge(r, on="split", suffixes=("", "_ref"))
        if len(m) == 0:
            continue
        row = {"experiment": exp, "config": cfg, "d": d, "M": M, "reference": ref, "n_splits": len(m)}
        for met in ("pr_auc", "roc_auc", "f2"):
            diff = m[met] - m[f"{met}_ref"]
            mu, lo, hi = mean_ci(diff)
            row.update({f"{met}_q": m[met].mean(), f"{met}_ref": m[f"{met}_ref"].mean(),
                        f"d_{met}": mu, f"d_{met}_lo": lo, f"d_{met}_hi": hi})
        if len(m) >= 5:
            row["p_wilcoxon"] = wilcoxon(m["pr_auc"], m["pr_auc_ref"])
            row["p_ttest"] = float(stats.ttest_rel(m["pr_auc"], m["pr_auc_ref"]).pvalue)
        if len(m) >= 3:
            row["p_tost"], _, ci90 = tost_paired(m["pr_auc"], m["pr_auc_ref"], margin)
            row["tost_ci90_lo"], row["tost_ci90_hi"] = ci90
            row["p_not_superior"] = noninferiority_upper(m["pr_auc"], m["pr_auc_ref"], margin)
        dl, bs = [], []
        for _, x in m.iterrows():
            a, b = np.load(x["file"]), np.load(x["file_ref"])
            if not np.array_equal(a["y_test"], b["y_test"]):
                raise RuntimeError(f"unpaired test sets: {x['name']} vs {x['name_ref']}")
            dl.append(delong(a["y_test"], a["s_test"], b["s_test"])[3])
            if n_boot:
                o, l_, h_ = paired_bootstrap(a["y_test"], a["s_test"], b["s_test"], "pr_auc", n_boot,
                                             seed=int(x["split"]))
                bs.append((l_ > 0) - (h_ < 0))           # +1 better, -1 worse, 0 n.s.
        row["delong_p_median"] = float(np.median(dl))
        row["delong_sig_splits"] = int(np.sum(np.array(dl) < 0.05))
        if n_boot:
            row["boot_better_splits"] = int(np.sum(np.array(bs) == 1))
            row["boot_worse_splits"] = int(np.sum(np.array(bs) == -1))
        rows.append(row)
    c = pd.DataFrame(rows)
    # Holm correction WITHIN each family = (configuration, reference, M) across the d values.
    # (A single family over all comparisons is too conservative: with 10 splits the smallest
    # attainable Wilcoxon p is ~0.002, so ~130 comparisons could never be significant.)
    if len(c) and "p_wilcoxon" in c:
        c["family"] = c.config + "|" + c.reference + "|" + c.M.astype(str)
        for _, g in c.groupby("family"):
            ok = g.index[g.p_wilcoxon.notna()]
            if len(ok):
                c.loc[ok, "p_wilcoxon_holm"] = holm(c.loc[ok, "p_wilcoxon"].to_numpy())
                c.loc[ok, "p_ttest_holm"] = holm(c.loc[ok, "p_ttest"].to_numpy())
        c["verdict"] = np.where(c.p_ttest_holm < 0.05, np.where(c.d_pr_auc > 0, "Q>C", "Q<C"),
                                np.where(c.p_tost < 0.05, "equivalent", "inconclusive"))
    return c


# ----------------------------------------------------------------------------- figures

def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def fig_prauc_vs_d(summ, df):
    s = summ[pd.to_numeric(summ.d, errors="coerce").notna()].copy()
    s["d_num"] = s.d.astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # best VQC variant per d is chosen by mean VALIDATION PR-AUC (no test peeking)
    vq = s[s.config.str.startswith("vqc")]
    if len(vq):
        vq = vq[vq.M == "full"] if (vq.M == "full").any() else vq
        best = vq.loc[vq.groupby("d_num").val_pr_auc.idxmax()]
        best = best.assign(config="vqc")
    else:
        best = vq
    sel = []
    for ent in ("svm_rbf", "lgbm", "rbf_matched", "qk_fidelity", "qk_projected", "qmsvm", "svm_bagged"):
        g = s[s.config.str.split("[").str[0] == ent]
        if ent in ("qk_fidelity", "qk_projected", "rbf_matched"):
            for t in LATEST_TAGS:
                if (g.config == f"{ent}[{t}]").any():
                    g = g[g.config == f"{ent}[{t}]"]
                    break
        if len(g):
            Mpick = "full" if (g.M == "full").any() else g.M.iloc[0]
            g = g[g.M == Mpick]
            g = g.loc[g.groupby("d_num").val_pr_auc.idxmax()]   # variant chosen on validation
            sel.append((ent, g))
    sel.insert(5, ("vqc", best))
    sel = [(e, g) for e, g in sel if len(g)]
    for k, (ent, g) in enumerate(sel):
        g = g.sort_values("d_num")
        quantum = ent in QUANTUM
        x = g.d_num + (k - (len(sel) - 1) / 2) * 0.09          # dodge so CIs do not overlap
        yerr = np.vstack([g.pr_auc - g.pr_auc_lo, g.pr_auc_hi - g.pr_auc]).clip(min=0)
        ax.errorbar(x, g.pr_auc, yerr=np.nan_to_num(yerr), color=ENTITY_COLOR[ent], linewidth=2,
                    marker="o", markersize=5, capsize=2, elinewidth=1,
                    linestyle="-" if quantum else "--", label=ent + ("" if quantum else " (classical)"))
    cnn = df[df.model == "cnn1d_raw"]
    if len(cnn):
        ax.axhline(cnn.pr_auc.mean(), color="#8a8980", linewidth=1.2, linestyle=":",
                   label="1D CNN on raw series (reference)")
    ax.axhline(df.prevalence.mean(), color="#c3c2b7", linewidth=1, label="prevalence (random ranking)")
    _style(ax, "number of features = qubits (d)", "test PR-AUC (mean, 95% CI over splits)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _save(fig, "fig1_prauc_vs_d")


def fig_prauc_vs_M(summ):
    s = summ[(summ.experiment == "E2_qkernel")]
    if not len(s):
        return
    ds = sorted(pd.to_numeric(s.d).unique())
    fig, axes = plt.subplots(1, len(ds), figsize=(3.0 * len(ds), 3.2), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], ds):
        g = s[pd.to_numeric(s.d) == d].copy()
        g["M_num"] = g.M.replace({"full": np.nan}).astype(float)
        full_n = 14460
        g["M_num"] = g["M_num"].fillna(full_n)
        for ent in ("rbf_matched", "qk_fidelity", "qk_projected"):
            h = g[g.config == ent].sort_values("M_num")
            if len(h):
                ax.plot(h.M_num, h.pr_auc, marker="o", markersize=5, linewidth=2,
                        color=ENTITY_COLOR[ent], label=ent)
        ax.set_xscale("log")
        ax.set_title(f"d = {int(d)}", fontsize=9)
        _style(ax, "training size M", "test PR-AUC" if d == ds[0] else "")
    axes[0][0].legend(fontsize=7, frameon=False)
    _save(fig, "fig2_prauc_vs_M")


def load_plain(exp):
    rows = [json.loads(f.read_text()) for f in sorted((config.RESULTS / exp).glob("*.json"))]
    return pd.DataFrame(rows)


def fig_gradvar():
    g = load_plain("E3_gradvar")
    if not len(g):
        return None
    agg = g.groupby(["readout", "depth", "d"]).var_expval_mean.mean().reset_index()
    fits = []
    readouts = [r for r in ("local", "local_mean", "global") if r in set(agg.readout)]
    fig, axes = plt.subplots(1, len(readouts), figsize=(4.0 * len(readouts), 3.4), sharey=True, squeeze=False)
    for ax, ro in zip(axes[0], readouts):
        depths = sorted(agg[agg.readout == ro].depth.unique())
        for k, L in enumerate(depths):
            h = agg[(agg.readout == ro) & (agg.depth == L)].sort_values("d")
            ax.plot(h.d, h.var_expval_mean, marker="o", markersize=5, linewidth=2,
                    color=seq_color(k, len(depths)), label=f"L = {L}")
            if len(h) >= 3:
                slope, icpt = np.polyfit(h.d, np.log10(h.var_expval_mean), 1)
                fits.append({"readout": ro, "depth": L, "log10_var_slope_per_qubit": slope,
                             "decay_factor_per_qubit": 10 ** slope})
        ax.set_yscale("log")
        ax.set_title(f"{ro} cost", fontsize=9)
        _style(ax, "qubits (d)", "Var[dC/dθ] over random inits" if ro == readouts[0] else "")
        ax.legend(fontsize=7, frameon=False)
    _save(fig, "fig3_gradient_variance")
    f = pd.DataFrame(fits)
    f.to_csv(OUT / "gradvar_fit.csv", index=False)
    return f


def fig_kernels():
    k = load_plain("E3_kernels")
    if not len(k):
        return
    agg = k.groupby(["bandwidth", "d"]).mean(numeric_only=True).reset_index()
    bws = [b for b in sorted(agg.bandwidth.unique()) if b in FIG_BWS] or sorted(agg.bandwidth.unique())[:4]
    gcol = max((c for c in agg.columns if c.startswith("g_rbf_fidelity_reg")),
                key=lambda c: float(c.split("reg")[1]))  # most regularised = most stable
    panels = [("fidelity_offdiag_var", "Var of off-diagonal K (fidelity)", True),
              ("fidelity_kta", "centered KTA (fidelity)", False),
              (gcol, f"g(K_C || K_Q), {gcol.split('_')[-1]}", False)]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), squeeze=False)
    for ax, (col, lab, logy) in zip(axes[0], panels):
        for kbw, bw in enumerate(bws):
            h = agg[agg.bandwidth == bw].sort_values("d")
            ax.plot(h.d, h[col], marker="o", markersize=5, linewidth=2,
                    color=seq_color(kbw, len(bws)), label=f"bandwidth {bw}")
        if col == "fidelity_kta":
            h = agg[agg.bandwidth == bws[0]].sort_values("d")
            ax.plot(h.d, h["rbf_kta"], color="#8a8980", linestyle="--", linewidth=1.5, label="classical RBF")
        if col == gcol:
            ax.axhline(np.sqrt(k.n.iloc[0]), color="#8a8980", linestyle=":", linewidth=1.2, label="sqrt(N)")
            ax.set_ylim(0, np.sqrt(k.n.iloc[0]) * 1.25)
        if logy:
            ax.set_yscale("log")
        _style(ax, "qubits (d)", lab)
        ax.legend(fontsize=7, frameon=False, loc="best")
    _save(fig, "fig4_kernel_diagnostics")


def fig_budget(summ):
    s = summ[summ.M.isin(["full"]) | summ.experiment.eq("E2_qmsvm")]
    s = s[pd.to_numeric(s.d, errors="coerce").notna()]
    if not len(s):
        return
    # use the d covered by the largest number of model families (ties -> largest d)
    cov = s.assign(fam=s.config.str.split("[").str[0], dn=pd.to_numeric(s.d)).groupby("dn").fam.nunique()
    dmax = cov[cov == cov.max()].index.max()
    s = s[pd.to_numeric(s.d) == dmax]
    s = s.assign(fam=s.config.str.split("[").str[0])
    order = [e for e in ENTITY_COLOR if e in set(s.fam) and e != "vqc"]
    vq = s[s.fam == "vqc"]
    rows = []
    for e in order:                      # several variants of a family -> best on VALIDATION
        g = s[s.fam == e]
        r = g.loc[g.val_pr_auc.idxmax()].copy()
        r["config"] = e
        rows.append(r)
    if len(vq):
        rows.insert(min(5, len(rows)), vq.loc[vq.val_pr_auc.idxmax()].copy())
        rows[min(5, len(rows) - 1)]["config"] = "vqc"
    budgets = ["precision_at_1pct", "precision_at_5pct", "precision_at_10pct"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.8 / max(1, len(rows))
    for i, r in enumerate(rows):
        x = np.arange(len(budgets)) + i * w
        ax.bar(x, [r[b] for b in budgets], width=w - 0.02, color=ENTITY_COLOR.get(r["config"], "#8a8980"),
               label=r["config"])
    ax.axhline(s.prevalence.mean(), color="#c3c2b7", linewidth=1, label="prevalence")
    ax.set_xticks(np.arange(len(budgets)) + 0.4 - w / 2)
    ax.set_xticklabels(["top 1%", "top 5%", "top 10%"])
    _style(ax, f"inspection budget (d = {int(dmax)})", "precision among inspected")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _save(fig, "fig5_precision_at_budget")


def fig_shots():
    """E4: test PR-AUC and gradient quality vs d, one line per shot budget, one column per readout."""
    rows = [json.loads(f.read_text()) for f in sorted((config.RESULTS / "E4_shots").glob("*.json"))]
    if not rows:
        return None
    k = pd.DataFrame([{"d": r["d"], "split": r["split"], "readout": r["readout"], "shots": r["shots"],
                       "pr_auc": r["metrics"]["test"]["pr_auc"], "cos": r["cos_mean_all"],
                       "rel_err": r["rel_err_median_all"]} for r in rows])
    order = ["exact"] + sorted([s for s in k.shots.unique() if s != "exact"], key=lambda v: -int(v))
    agg = k.groupby(["readout", "shots", "d"]).agg(pr_auc=("pr_auc", "mean"), cos=("cos", "mean"),
                                                   rel_err=("rel_err", "mean")).reset_index()
    agg.to_csv(OUT / "shots_summary.csv", index=False)
    readouts = list(dict.fromkeys(agg.readout))
    fig, axes = plt.subplots(2, len(readouts), figsize=(4.2 * len(readouts), 6.2), squeeze=False, sharex=True)
    for j, ro in enumerate(readouts):
        for i, s in enumerate(order):
            h = agg[(agg.readout == ro) & (agg.shots == s)].sort_values("d")
            if not len(h):
                continue
            col = "#8a8980" if s == "exact" else seq_color(i - 1, len(order) - 1)
            ls = ":" if s == "exact" else "-"
            lab = "exact gradient" if s == "exact" else f"{int(s):,} shots"
            axes[0][j].plot(h.d, h.pr_auc, marker="o", markersize=5, linewidth=2, color=col, ls=ls, label=lab)
            axes[1][j].plot(h.d, h.cos, marker="o", markersize=5, linewidth=2, color=col, ls=ls, label=lab)
        axes[0][j].set_title(f"{ro} readout", fontsize=9)
        _style(axes[0][j], "", "test PR-AUC" if j == 0 else "")
        _style(axes[1][j], "qubits (d)", "cos(estimated, exact gradient)" if j == 0 else "")
        axes[1][j].set_ylim(-0.05, 1.05)
    axes[0][0].legend(fontsize=7, frameon=False)
    _save(fig, "fig6_finite_shots")
    return agg


def to_markdown(df, cols, floatfmt=".4f"):
    d = df[cols].copy()
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) and not np.isnan(v) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.02, help="equivalence margin on PR-AUC")
    ap.add_argument("--bootstrap", type=int, default=1000, help="0 disables per-split bootstrap")
    ap.add_argument("--figs-only", action="store_true", help="redraw figures, keep existing comparisons.csv")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    if not len(df):
        sys.exit("no prediction results found")
    df.to_csv(OUT / "all_runs.csv", index=False)
    summ = summary_table(df)
    summ.to_csv(OUT / "summary.csv", index=False)
    if a.figs_only and (OUT / "comparisons.csv").exists():
        try:
            comp = pd.read_csv(OUT / "comparisons.csv")
        except pd.errors.EmptyDataError:
            comp = pd.DataFrame()
    else:
        comp = compare(df, a.margin, a.bootstrap)
        comp.to_csv(OUT / "comparisons.csv", index=False)
    fig_prauc_vs_d(summ, df)
    fig_prauc_vs_M(summ)
    fits = fig_gradvar()
    fig_kernels()
    fig_budget(summ)
    shots = fig_shots()
    with open(OUT / "REPORT.md", "w") as f:
        f.write("# Results summary\n\nAll values: test set, mean over splits [95% CI].\n\n")
        cols = ["experiment", "config", "d", "M", "n_splits", "pr_auc", "pr_auc_lo", "pr_auc_hi",
                "roc_auc", "f2", "mcc", "precision_at_5pct"]
        srt = summ.assign(_d=pd.to_numeric(summ.d, errors="coerce").fillna(-1))
        f.write(to_markdown(srt.sort_values(["_d", "experiment", "config"]), cols))
        if len(comp):
            f.write(f"\n\n# Paired comparisons (PR-AUC), equivalence margin {a.margin}\n\n")
            ccols = [c for c in ["experiment", "config", "d", "M", "reference", "n_splits", "d_pr_auc",
                                 "d_pr_auc_lo", "d_pr_auc_hi", "p_ttest_holm", "p_wilcoxon_holm", "p_tost", "verdict",
                                 "p_not_superior", "delong_sig_splits", "boot_better_splits",
                                 "boot_worse_splits"] if c in comp.columns]
            f.write(to_markdown(comp, ccols))
        if shots is not None and len(shots):
            f.write("\n\n# Finite-shot training (E4): mean over splits\n\n")
            f.write(to_markdown(shots, list(shots.columns)))
        if fits is not None and len(fits):
            f.write("\n\n# Gradient-variance decay (fit of log10 Var vs d)\n\n")
            f.write(to_markdown(fits, list(fits.columns)))
    print(f"[agg] {len(df)} runs -> {OUT}")
