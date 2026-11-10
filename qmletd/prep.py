"""Splits, feature selection, scaling, balancing and matched subsampling.

Every data-dependent transformation is fitted on the training partition only.
The same split seed always yields the same train/val/test indices for all models,
so test-set predictions of different models are paired (needed for DeLong/bootstrap).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class SplitIdx:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_split(y: np.ndarray, seed: int) -> SplitIdx:
    idx = np.arange(len(y))
    f_tr, f_va, f_te = config.SPLIT_FRACTIONS
    tr, tmp = train_test_split(idx, test_size=f_va + f_te, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=f_te / (f_va + f_te), stratify=y[tmp],
                              random_state=seed + 10_000)
    return SplitIdx(np.sort(tr), np.sort(va), np.sort(te))


def mi_ranking(X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> np.ndarray:
    """Feature indices sorted by mutual information with the label (train only)."""
    mi = mutual_info_classif(X_tr, y_tr, random_state=seed)
    return np.argsort(-mi, kind="stable"), mi


def balance(X: np.ndarray, y: np.ndarray, seed: int, mode: str = "rus_smote"):
    if mode == "none":
        return X, y
    if mode == "rus":
        return RandomUnderSampler(sampling_strategy=1.0, random_state=seed).fit_resample(X, y)
    if mode == "rus_smote":
        Xr, yr = RandomUnderSampler(sampling_strategy=config.RUS_RATIO,
                                    random_state=seed).fit_resample(X, y)
        return SMOTE(sampling_strategy=1.0, random_state=seed).fit_resample(Xr, yr)
    raise ValueError(mode)


def stratified_subsample(X, y, M, seed):
    """Balanced subsample of size M (M//2 per class) from an already-balanced set.

    Returned in a deterministic order: identical for every model given (split, M, seed).
    """
    if M is None or M >= len(y):
        return X, y
    rng = np.random.default_rng(seed + 777)
    out = []
    for c in (0, 1):
        idx_c = np.flatnonzero(y == c)
        k = min(len(idx_c), M // 2 if c == 0 else M - M // 2)
        out.append(rng.choice(idx_c, size=k, replace=False))
    sel = np.sort(np.concatenate(out))
    return X[sel], y[sel]


@dataclass
class Prepared:
    """Preprocessed partitions for one (split, d)."""
    X_tr: np.ndarray          # balanced training set, values in (-pi/2, pi/2)
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    feature_idx: np.ndarray
    split: int
    d: int


def preprocess(X, y, split_seed: int, d: int, balance_mode: str = "rus_smote",
               ranking_cache: dict | None = None) -> Prepared:
    s = make_split(y, split_seed)
    X_tr_raw, y_tr = X[s.train], y[s.train]
    key = split_seed
    if ranking_cache is not None and key in ranking_cache:
        order = ranking_cache[key]
    else:
        order, _ = mi_ranking(X_tr_raw, y_tr, seed=split_seed)
        if ranking_cache is not None:
            ranking_cache[key] = order
    fidx = np.sort(order[:d]) if d < X.shape[1] else np.arange(X.shape[1])
    scaler = StandardScaler().fit(X_tr_raw[:, fidx])

    def tf(A):
        return np.arctan(scaler.transform(A[:, fidx]))

    X_tr, X_va, X_te = tf(X_tr_raw), tf(X[s.val]), tf(X[s.test])
    X_tr, y_tr = balance(X_tr, y_tr, seed=split_seed, mode=balance_mode)
    return Prepared(X_tr, np.asarray(y_tr), X_va, y[s.val], X_te, y[s.test], fidx,
                    split_seed, d)
