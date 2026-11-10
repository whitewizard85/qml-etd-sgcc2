"""Hold-out protocol: final evaluation on the fixed, untouched test set.

Run the experiment scripts first with QMLETD_PROTOCOL=holdout (splits 0..4 = inner folds);
each fold run selects hyperparameters and the F2 threshold on its own validation fold and
scores the common test set. This script then, for every configuration,

  1. standardises each fold model's scores with the mean/sd of ITS validation scores
     (no test information), and averages the K fold models -> one ensemble score per test
     customer (a cross-validation committee);
  2. picks the F2 threshold on the pooled out-of-fold validation scores and applies it once
     to the ensemble test scores;
  3. compares every quantum configuration with its matched control ON THE SAME TEST
     CUSTOMERS: stratified paired bootstrap of the PR-AUC difference (95% CI, 90% CI for
     equivalence), DeLong for ROC-AUC;
  4. multiplicity: Holm over the PRE-SPECIFIED primary family (all quantum configurations
     at the largest d, M = full; QUBO-SVM at its bagged setting, each against its matched
     control). Everything else is secondary: reported with CIs, Holm over all of them too.

  QMLETD_PROTOCOL=holdout python scripts/13_holdout_eval.py --boot 10000
Output: <results>/summary/holdout_{runs,summary,comparisons}.csv, holdout_report.md
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import config, data, metrics, prep  # noqa: E402
from qmletd.stats import bootstrap_diff_fast, delong, holm, verdict  # noqa: E402

_spec = importlib.util.spec_from_file_location("agg", Path(__file__).with_name("09_aggregate.py"))
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)

OUT = config.RESULTS / "summary"
MATCHED_REFS = ("rbf_matched", "svm_rbf", "svm_bagged")
REPORT_METRICS = ("pr_auc", "roc_auc", "f2", "mcc", "precision_at_5pct", "recall_at_5pct")


def ensemble(g):
    """g: fold runs of one configuration. Returns dict with y_test, s_test, oof val, metrics."""
    zs_te, zv, yv, y_te, folds = [], [], [], None, []
    for _, r in g.sort_values("split").iterrows():
        a = np.load(r["file"])
        mu, sd = a["s_val"].mean(), a["s_val"].std() + 1e-12
        zs_te.append((a["s_test"] - mu) / sd)
        zv.append((a["s_val"] - mu) / sd)
        yv.append(a["y_val"])
        if y_te is None:
            y_te = a["y_test"]
        elif not np.array_equal(y_te, a["y_test"]):
            raise RuntimeError(f"fold runs of {r['config']} do not share the test set")
        folds.append(int(r["split"]))
    s_te = np.mean(zs_te, axis=0)
    thr, _ = metrics.best_fbeta_threshold(np.concatenate(yv), np.concatenate(zv), beta=2.0)
    return {"y_test": y_te, "s_test": s_te, "metrics": metrics.evaluate(y_te, s_te, thr),
            "folds": folds}


def per_theft_type(cache, c):
    """Only for datasets that store the theft type of every unit (TDD2022)."""
    f = config.DATA_PROCESSED / "theft_type.npy"
    if not f.exists():
        return None
    from sklearn.metrics import average_precision_score
    _, y = data.load_features()
    _, te = prep.holdout_test(y, prep._load_groups(len(y)))
    tt = np.load(f, allow_pickle=True)[te]
    keys = set()
    for _, r in c[c.primary].iterrows():
        keys.add((r.experiment, r.config, r.d, r.M))
    ref_keys = [k for k in cache if k[1] in MATCHED_REFS and k[2] in {kk[2] for kk in keys}]
    rows = []
    for k in sorted(keys) + sorted(set(ref_keys)):
        e = cache[k]
        if not np.array_equal(e["y_test"], y[te]):
            raise RuntimeError("test indices do not match the stored predictions")
        row = {"config": k[1], "d": k[2], "M": k[3]}
        neg = e["y_test"] == 0
        for t in sorted(set(tt[e["y_test"] == 1])):
            sel = neg | (tt == t)
            row[str(t)] = average_precision_score(e["y_test"][sel], e["s_test"][sel])
        rows.append(row)
    return pd.DataFrame(rows)


def main(margin, n_boot, require_folds):
    df = agg.load_runs()
    if not len(df):
        sys.exit(f"no runs found in {config.RESULTS}")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "holdout_runs.csv", index=False)

    cache, summ = {}, []
    for key, g in df.groupby(["experiment", "config", "d", "M"]):
        if len(g) < require_folds:
            print(f"[holdout] skip {key}: {len(g)} folds < {require_folds}")
            continue
        e = ensemble(g)
        cache[key] = e
        row = dict(zip(["experiment", "config", "d", "M"], key), n_folds=len(g))
        row.update({m: e["metrics"][m] for m in REPORT_METRICS})
        row.update({f"{m}_fold_mean": g[m].mean() for m in ("pr_auc", "roc_auc")})
        row["pr_auc_fold_sd"] = g["pr_auc"].std(ddof=1)
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "holdout_summary.csv", index=False)

    rows = []
    for exp, cfg, d, M, ref, g, r in agg.comparison_pairs(df):
        if ref not in MATCHED_REFS:
            continue
        kq, kr = (exp, cfg, d, M), (r.experiment.iloc[0], r.config.iloc[0], d, r.M.iloc[0])
        if kq not in cache or kr not in cache:
            continue
        q, c = cache[kq], cache[kr]
        if not np.array_equal(q["y_test"], c["y_test"]):
            raise RuntimeError(f"unpaired test sets: {kq} vs {kr}")
        b = bootstrap_diff_fast(q["y_test"], q["s_test"], c["s_test"], n_boot=n_boot, seed=0)
        auc_q, auc_c, _, p_dl = delong(q["y_test"], q["s_test"], c["s_test"])
        rows.append({"experiment": exp, "config": cfg, "d": d, "M": M, "reference": ref,
                     "pr_auc_q": q["metrics"]["pr_auc"], "pr_auc_ref": c["metrics"]["pr_auc"],
                     "d_pr_auc": b["diff"], "ci95_lo": b["ci95"][0], "ci95_hi": b["ci95"][1],
                     "ci90_lo": b["ci90"][0], "ci90_hi": b["ci90"][1], "p_boot": b["p"],
                     "d_roc_auc": auc_q - auc_c, "p_delong": p_dl,
                     "d_precision_at_5pct": q["metrics"]["precision_at_5pct"]
                     - c["metrics"]["precision_at_5pct"]})
    c = pd.DataFrame(rows)
    if not len(c):
        sys.exit("no complete matched comparisons yet")
    for tag in ("bwext2", "bwext"):                 # kernels: widest grid campaign only
        if c.config.str.contains(rf"\[{tag}\]").any():
            c = c[~c.config.str.startswith("qk_") | c.config.str.contains(rf"\[{tag}\]")]
            break
    c = c.reset_index(drop=True)
    c["d_num"] = pd.to_numeric(c.d, errors="coerce")
    dmax = c.d_num.max()
    c["primary"] = (c.d_num == dmax) & ((c.M.astype(str) == "full") | c.config.str.startswith("qmsvm"))
    c.loc[c.primary, "p_holm_primary"] = holm(c.loc[c.primary, "p_boot"].to_numpy())
    sec = ~c.primary
    if sec.any():
        c.loc[sec, "p_holm_secondary"] = holm(c.loc[sec, "p_boot"].to_numpy())
    c["p_adj"] = c.p_holm_primary.fillna(c.get("p_holm_secondary"))
    c["verdict"] = [verdict(p, d_, (lo, hi), margin)
                    for p, d_, lo, hi in zip(c.p_adj, c.d_pr_auc, c.ci90_lo, c.ci90_hi)]
    c.drop(columns=["d_num"]).to_csv(OUT / "holdout_comparisons.csv", index=False)

    n_te = len(next(iter(cache.values()))["y_test"])
    prev = next(iter(cache.values()))["y_test"].mean()
    lines = ["# Hold-out evaluation", "",
             f"Dataset: {config.DATASET}. Untouched test set: {n_te} units, prevalence {prev:.4f}. "
             f"Inner folds: {config.HOLDOUT_FOLDS}. Bootstrap replicates: {n_boot}. "
             f"Equivalence margin: {margin} PR-AUC.", "",
             "## Primary family (pre-specified), Holm-adjusted", ""]
    cols = ["config", "reference", "d", "M", "pr_auc_q", "pr_auc_ref", "d_pr_auc", "ci95_lo",
            "ci95_hi", "ci90_lo", "ci90_hi", "p_adj", "verdict", "d_precision_at_5pct"]
    lines.append(agg.to_markdown(c[c.primary].sort_values("config"), cols))
    lines += ["", "## Secondary comparisons (Holm over all secondary comparisons)", ""]
    lines.append(agg.to_markdown(c[~c.primary].sort_values(["config", "M", "d"]), cols))
    lines += ["", "## Verdict counts", "", c.groupby(["primary", "verdict"]).size().to_string(), "",
              "## Test-set performance of the fold ensembles", ""]
    sc = ["experiment", "config", "d", "M", "n_folds"] + list(REPORT_METRICS) + ["pr_auc_fold_mean",
                                                                                 "pr_auc_fold_sd"]
    srt = summ.assign(_d=pd.to_numeric(summ.d, errors="coerce").fillna(-1))
    lines.append(agg.to_markdown(srt.sort_values(["_d", "experiment", "config"]), sc))
    pt = per_theft_type(cache, c)
    if pt is not None:
        pt.to_csv(OUT / "holdout_per_theft_type.csv", index=False)
        lines += ["", "## PR-AUC by theft type (positives of one type vs all negatives), primary family", "",
                  agg.to_markdown(pt, list(pt.columns))]
    (OUT / "holdout_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12 + int(c.primary.sum())]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=config.EQUIVALENCE_MARGIN)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--require-folds", type=int, default=config.HOLDOUT_FOLDS,
                    help="configurations with fewer completed folds are skipped")
    a = ap.parse_args()
    if config.PROTOCOL != "holdout":
        sys.exit("set QMLETD_PROTOCOL=holdout")
    main(a.margin, a.boot, a.require_folds)
