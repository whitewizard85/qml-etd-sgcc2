"""Statistical comparison tools.

* mean_ci: mean and t-based 95% CI across splits.
* delong: paired DeLong test for two ROC-AUCs on the same test set (Sun & Xu 2014 fast version).
* paired_bootstrap: bootstrap CI of a metric difference on the same test set.
* wilcoxon_holm: Wilcoxon signed-rank across splits, Holm-corrected over a family.
* tost_paired: two one-sided paired t-tests for equivalence within +-margin.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def mean_ci(x, level=0.95):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = x.mean()
    if n == 1:
        return m, np.nan, np.nan
    h = stats.t.ppf((1 + level) / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = T
    return out


def delong(y, s1, s2):
    """Returns (auc1, auc2, z, p) for H0: AUC1 == AUC2 (paired)."""
    y = np.asarray(y).astype(int)
    order = np.r_[np.flatnonzero(y == 1), np.flatnonzero(y == 0)]
    m = int(y.sum())
    preds = np.vstack([np.asarray(s1, float)[order], np.asarray(s2, float)[order]])
    n = preds.shape[1] - m
    tx = np.array([_midrank(p[:m]) for p in preds])
    ty = np.array([_midrank(p[m:]) for p in preds])
    tz = np.array([_midrank(p) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))


def paired_bootstrap(y, s1, s2, metric="pr_auc", n_boot=2000, seed=0):
    """Stratified bootstrap CI of metric(s1) - metric(s2)."""
    f = average_precision_score if metric == "pr_auc" else roc_auc_score
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        diffs[b] = f(y[idx], s1[idx]) - f(y[idx], s2[idx])
    obs = f(y, s1) - f(y, s2)
    return float(obs), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def tost_paired(a, b, margin):
    """Equivalence of mean(a - b) within [-margin, +margin]. Returns (p_tost, mean_diff, ci90)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return (0.0 if abs(m) < margin else 1.0), float(m), (float(m), float(m))
    p_low = stats.t.sf((m + margin) / se, n - 1)
    p_up = stats.t.cdf((m - margin) / se, n - 1)
    h = stats.t.ppf(0.95, n - 1) * se
    return float(max(p_low, p_up)), float(m), (float(m - h), float(m + h))


def noninferiority_upper(a, b, margin):
    """One-sided test H0: mean(a - b) >= margin  vs  H1: mean(a - b) < margin.
    Small p => a is NOT better than b by more than `margin` ("not superior")."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if d.mean() < margin else 1.0
    return float(stats.t.cdf((d.mean() - margin) / se, n - 1))
