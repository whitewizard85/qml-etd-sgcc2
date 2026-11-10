"""Loading the SGCC dataset and building the 13 tabular features used in the thesis.

Differences from the thesis code (documented on purpose):
* date columns are parsed and SORTED chronologically before interpolation. In the public
  SGCC CSV the date columns are not guaranteed to be in chronological order; interpolating
  along an unsorted axis would mix non-adjacent days.
* everything is vectorised (no per-customer Python loops).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config

FEATURE_NAMES = [
    "mean", "std", "cv", "median", "min", "max", "skewness", "kurtosis",
    "pct_zero", "pct_missing", "trend_slope", "max_daily_drop", "autocorr_lag1",
]


def load_sgcc(csv_path=None):
    """Return (X_raw [n, T] float64 with NaNs, y [n] int8, ids [n], dates [T])."""
    csv_path = csv_path or config.SGCC_CSV
    df = pd.read_csv(csv_path)
    id_col = "CONS_NO" if "CONS_NO" in df.columns else df.columns[0]
    label_col = "FLAG" if "FLAG" in df.columns else df.columns[1]
    other = [c for c in df.columns if c not in (id_col, label_col)]
    dates = pd.to_datetime(pd.Index(other), errors="coerce")
    valid = ~dates.isna()
    if valid.sum() < len(other):
        print(f"[data] warning: {len(other) - valid.sum()} non-date columns ignored")
    cols = np.array(other)[valid]
    dates = dates[valid]
    order = np.argsort(dates.values)
    cols, dates = cols[order], dates[order]
    if df[id_col].duplicated().any():
        n_dup = int(df[id_col].duplicated().sum())
        print(f"[data] warning: {n_dup} duplicated customer ids, keeping first occurrence")
        df = df.drop_duplicates(subset=id_col, keep="first")
    X = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = df[label_col].astype(int).to_numpy().astype(np.int8)
    return X, y, df[id_col].to_numpy(), dates


def interpolate_series(X: np.ndarray):
    """Linear interpolation along time per customer, then zero-fill residual NaNs.

    Returns (X_filled, missing_mask). Per-row operation: no information crosses customers,
    so doing it before the train/val/test split does not leak labels.
    """
    mask = np.isnan(X)
    filled = pd.DataFrame(X).interpolate(axis=1, limit_direction="both").to_numpy()
    filled = np.nan_to_num(filled, nan=0.0)
    return filled, mask


def engineer_features(X_filled: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    x = X_filled
    n, T = x.shape
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    cv = np.divide(std, mean, out=np.zeros_like(std), where=mean > 0)
    median = np.median(x, axis=1)
    mn, mx = x.min(axis=1), x.max(axis=1)
    with np.errstate(all="ignore"):
        skew = np.nan_to_num(stats.skew(x, axis=1), nan=0.0)
        kurt = np.nan_to_num(stats.kurtosis(x, axis=1), nan=0.0)
    pct_zero = (x == 0).mean(axis=1)
    pct_missing = mask.mean(axis=1)
    t = np.arange(T, dtype=np.float64)
    tc = t - t.mean()
    slope = (x - mean[:, None]) @ tc / (tc @ tc)
    drops = x[:, :-1] - x[:, 1:]
    max_drop = np.clip(drops.max(axis=1), 0, None)
    a, b = x[:, :-1], x[:, 1:]
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
    ac1 = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    feats = np.column_stack([mean, std, cv, median, mn, mx, skew, kurt, pct_zero,
                             pct_missing, slope, max_drop, ac1])
    return pd.DataFrame(feats, columns=FEATURE_NAMES)


def prepare_and_cache(csv_path=None) -> None:
    config.ensure_dirs()
    X, y, ids, dates = load_sgcc(csv_path)
    print(f"[data] {X.shape[0]} customers, {X.shape[1]} days "
          f"({dates.min().date()} .. {dates.max().date()}), positives={y.mean():.4%}, "
          f"missing={np.isnan(X).mean():.2%}")
    filled, mask = interpolate_series(X)
    feats = engineer_features(filled, mask)
    feats.insert(0, "customer_id", ids)
    feats["label"] = y
    feats.to_parquet(config.FEATURES_FILE, index=False)
    np.save(config.SERIES_FILE, filled.astype(np.float32))
    np.save(config.MASK_FILE, mask)
    np.save(config.LABELS_FILE, y)
    print(f"[data] cached features -> {config.FEATURES_FILE}")


def load_features():
    df = pd.read_parquet(config.FEATURES_FILE)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy().astype(np.int64)
    return X, y


def load_series():
    return (np.load(config.SERIES_FILE, mmap_mode="r"),
            np.load(config.MASK_FILE, mmap_mode="r"),
            np.load(config.LABELS_FILE).astype(np.int64))
