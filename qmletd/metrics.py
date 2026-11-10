"""Threshold selection and evaluation metrics for imbalanced binary classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config


def best_fbeta_threshold(y: np.ndarray, s: np.ndarray, beta: float = 2.0):
    """Exhaustive search over the thresholds induced by the observed scores.

    Rule: predict positive iff score >= threshold. Between consecutive distinct scores the
    confusion matrix does not change, so this search is exact. Ties: highest threshold wins.
    Returns (threshold, best_fbeta).
    """
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    ss, ys = s[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    P = ys.sum()
    last_of_tie = np.r_[ss[1:] != ss[:-1], True]
    tp, fp, thr = tp[last_of_tie], fp[last_of_tie], ss[last_of_tie]
    fn = P - tp
    b2 = beta ** 2
    denom = (1 + b2) * tp + b2 * fn + fp
    f = np.divide((1 + b2) * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    i = int(np.argmax(f))
    return float(thr[i]), float(f[i])


def _at_budget(y, s, frac):
    n = len(y)
    k = max(1, int(round(frac * n)))
    top = np.argsort(-s, kind="stable")[:k]
    tp = y[top].sum()
    return tp / k, tp / max(1, y.sum())


def evaluate(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    pred = (s >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if 4 * prec + rec else 0.0
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0.0
    out = {
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, s)),
        "precision": prec, "recall": rec, "specificity": spec, "f1": f1, "f2": f2,
        "mcc": float(mcc), "balanced_acc": (rec + spec) / 2,
        "pos_rate": float(pred.mean()), "prevalence": float(y.mean()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "threshold": float(threshold),
    }
    for b in config.BUDGETS:
        p, r = _at_budget(y, s, b)
        out[f"precision_at_{int(b * 100)}pct"] = float(p)
        out[f"recall_at_{int(b * 100)}pct"] = float(r)
    return out


def finalize(y_va, s_va, y_te, s_te) -> dict:
    """Pick F2 threshold on validation, evaluate on test once."""
    thr, f2_val = best_fbeta_threshold(y_va, s_va, beta=2.0)
    res = {"val": evaluate(y_va, s_va, thr), "test": evaluate(y_te, s_te, thr)}
    res["val"]["f2_selected"] = f2_val
    return res
