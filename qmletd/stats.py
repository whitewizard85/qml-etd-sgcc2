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


# ----------------------------------------------------------------------------------------
# Inference that accounts for the overlap between repeated random splits.
# With repeated hold-out the J per-split differences are positively correlated (training and
# test sets overlap), so the naive paired t-test underestimates the variance. Nadeau & Bengio
# (2003) replace Var/J by Var * (1/J + n_test/n_train). This "corrected resampled t-test" is
# conservative under the usual assumptions and is the standard fix for this design.

def corrected_resampled_ttest(a, b, test_train_ratio):
    """Two-sided corrected resampled t-test of mean(a - b) = 0.

    Returns (mean_diff, p_value, (ci95_lo, ci95_hi)).
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    J = len(d)
    if J < 2:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    m = d.mean()
    var = d.var(ddof=1)
    se = np.sqrt(var * (1.0 / J + test_train_ratio))
    if se == 0:
        return float(m), (1.0 if m == 0 else 0.0), (float(m), float(m))
    t = m / se
    p = 2 * stats.t.sf(abs(t), J - 1)
    h = stats.t.ppf(0.975, J - 1) * se
    return float(m), float(p), (float(m - h), float(m + h))


def corrected_tost(a, b, margin, test_train_ratio):
    """Equivalence within +-margin with the Nadeau-Bengio variance. Returns (p_tost, ci90)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    J = len(d)
    if J < 2:
        return float("nan"), (float("nan"), float("nan"))
    m = d.mean()
    se = np.sqrt(d.var(ddof=1) * (1.0 / J + test_train_ratio))
    if se == 0:
        return (0.0 if abs(m) < margin else 1.0), (float(m), float(m))
    p_low = stats.t.sf((m + margin) / se, J - 1)
    p_up = stats.t.cdf((m - margin) / se, J - 1)
    h = stats.t.ppf(0.95, J - 1) * se
    return float(max(p_low, p_up)), (float(m - h), float(m + h))


# ----------------------------------------------------------------------------------------
# Inference on ONE fixed, untouched test set (hold-out protocol). The unit of resampling is
# the test customer; both models are scored on the same customers, so the bootstrap is paired.

def bootstrap_diff(y, s1, s2, metric="pr_auc", n_boot=10_000, seed=0):
    """Stratified paired bootstrap of metric(s1) - metric(s2) on a single test set.

    Returns dict: diff, ci95, ci90 (percentile), p (two-sided, from the bootstrap distribution
    shifted to the null), se.
    """
    f = average_precision_score if metric == "pr_auc" else roc_auc_score
    y = np.asarray(y).astype(int)
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        diffs[b] = f(y[idx], s1[idx]) - f(y[idx], s2[idx])
    obs = f(y, s1) - f(y, s2)
    centred = diffs - diffs.mean()
    p = float((np.sum(np.abs(centred) >= abs(obs)) + 1) / (n_boot + 1))
    return {"diff": float(obs), "se": float(diffs.std(ddof=1)), "p": p,
            "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
            "ci90": (float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95)))}


def verdict(p_adj, diff, ci90, margin, alpha=0.05):
    """Q>C / Q<C if the (adjusted) difference test rejects; 'equivalent' if the 90% CI lies in
    (-margin, +margin) (TOST at level 0.05); otherwise 'inconclusive'."""
    if p_adj < alpha:
        return "Q>C" if diff > 0 else "Q<C"
    if -margin < ci90[0] and ci90[1] < margin:
        return "equivalent"
    return "inconclusive"


def _weighted_ap(y, s, W):
    """Average precision (sklearn definition) for each row of the weight matrix W [B, n].

    Ties in s are grouped as in sklearn. Used for fast bootstrap: W holds resampling counts."""
    order = np.argsort(-s, kind="stable")
    s_o, y_o = s[order], y[order].astype(float)
    W = W[:, order].astype(float)
    last = np.r_[s_o[1:] != s_o[:-1], True]              # last index of each distinct score
    tp = np.cumsum(W * y_o, axis=1)[:, last]
    fp = np.cumsum(W * (1 - y_o), axis=1)[:, last]
    P = tp[:, -1:]
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, P, out=np.zeros_like(tp), where=P > 0)
    drec = np.diff(np.concatenate([np.zeros((W.shape[0], 1)), rec], axis=1), axis=1)
    return (drec * prec).sum(axis=1)


def bootstrap_diff_fast(y, s1, s2, n_boot=10_000, seed=0, chunk=500, margin=0.02):
    """Same as bootstrap_diff for PR-AUC, vectorised (resampling counts as weights)."""
    y = np.asarray(y).astype(int)
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = []
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        W = np.zeros((b, len(y)), dtype=np.int32)
        W[:, pos] = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=b)
        W[:, neg] = rng.multinomial(len(neg), np.full(len(neg), 1 / len(neg)), size=b)
        diffs.append(_weighted_ap(y, s1, W) - _weighted_ap(y, s2, W))
        done += b
    diffs = np.concatenate(diffs)
    obs = float(average_precision_score(y, s1) - average_precision_score(y, s2))
    centred = diffs - diffs.mean()
    p = float((np.sum(np.abs(centred) >= abs(obs)) + 1) / (n_boot + 1))
    # bootstrap TOST: share of the bootstrap distribution at or beyond each margin (the larger one)
    p_tost = float(max(np.sum(diffs <= -margin) + 1, np.sum(diffs >= margin) + 1) / (n_boot + 1))
    return {"diff": obs, "se": float(diffs.std(ddof=1)), "p": p, "p_tost": p_tost,
            "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
            "ci90": (float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95)))}


def verdict2(p_diff_adj, p_tost_adj, diff, alpha=0.05):
    """Verdict from multiplicity-adjusted difference and equivalence p-values.

    Q<C / Q>C   : difference significant and equivalence not established
    equivalent* : difference significant but also equivalent within the margin (negligible)
    equivalent  : equivalence established, difference not significant
    inconclusive: neither
    """
    sig, eq = p_diff_adj < alpha, p_tost_adj < alpha
    if sig and eq:
        return "equivalent*"
    if sig:
        return "Q>C" if diff > 0 else "Q<C"
    return "equivalent" if eq else "inconclusive"
