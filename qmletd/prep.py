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


def _load_groups(n):
    """Groups that must never be split across partitions (e.g. TDD2022 consumer types)."""
    if not config.GROUPS_FILE.exists():
        return None
    g = np.load(config.GROUPS_FILE, allow_pickle=True)
    if len(g) != n:
        raise RuntimeError(f"groups file has {len(g)} rows, labels have {n}")
    return g


def _strat_split(idx, y, frac, seed, groups):
    """Split idx into (rest, part) with |part| ~ frac*|idx|, stratified on y.

    Without groups: sklearn train_test_split. With groups: StratifiedGroupKFold with
    round(1/frac) folds, first fold taken as `part` (whole groups only)."""
    if groups is None:
        rest, part = train_test_split(idx, test_size=frac, stratify=y[idx], random_state=seed)
        return np.sort(rest), np.sort(part)
    from sklearn.model_selection import StratifiedGroupKFold
    k = max(2, int(round(1 / frac)))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed % (2**32 - 1))
    rest, part = next(cv.split(idx, y[idx], groups[idx]))
    return np.sort(idx[rest]), np.sort(idx[part])


def holdout_test(y, groups=None):
    """The fixed, untouched test set of the hold-out protocol (same for every model/run)."""
    idx = np.arange(len(y))
    return _strat_split(idx, y, config.HOLDOUT_TEST_FRACTION, config.HOLDOUT_SEED, groups)


def make_split(y: np.ndarray, seed: int, groups=None) -> SplitIdx:
    """Train/val/test indices for 'split' `seed` under the active protocol.

    splits : repeated stratified 60/20/20 split with seed `seed` (original protocol).
    holdout: test = fixed hold-out set (independent of `seed`); the remaining data are cut
             into HOLDOUT_FOLDS stratified folds (fixed seed) and fold `seed` is the
             validation set. Hyperparameters and thresholds therefore never see the test set.
    If a groups file exists, whole groups are kept together in every partition.
    """
    if groups is None:
        groups = _load_groups(len(y))
    idx = np.arange(len(y))
    if config.PROTOCOL == "splits":
        f_tr, f_va, f_te = config.SPLIT_FRACTIONS
        if groups is None:      # exactly the original code: results stay paired with old runs
            tr, tmp = train_test_split(idx, test_size=f_va + f_te, stratify=y, random_state=seed)
            va, te = train_test_split(tmp, test_size=f_te / (f_va + f_te), stratify=y[tmp],
                                      random_state=seed + 10_000)
            return SplitIdx(np.sort(tr), np.sort(va), np.sort(te))
        tr, tmp = _strat_split(idx, y, f_va + f_te, seed, groups)
        va, te = _strat_split(tmp, y, f_te / (f_va + f_te), seed + 10_000, groups)
        return SplitIdx(tr, va, te)
    if not 0 <= seed < config.HOLDOUT_FOLDS:
        raise ValueError(f"hold-out protocol: split must be in 0..{config.HOLDOUT_FOLDS - 1}, got {seed}")
    dev, te = holdout_test(y, groups)
    if groups is None:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(config.HOLDOUT_FOLDS, shuffle=True, random_state=config.HOLDOUT_SEED + 1)
        folds = list(cv.split(dev, y[dev]))
    else:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(config.HOLDOUT_FOLDS, shuffle=True, random_state=config.HOLDOUT_SEED + 1)
        folds = list(cv.split(dev, y[dev], groups[dev]))
    tr, va = folds[seed]
    return SplitIdx(np.sort(dev[tr]), np.sort(dev[va]), te)


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
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        if min(n1, n0) / max(n1, n0) < config.RUS_RATIO:    # only if more imbalanced than RUS_RATIO
            X, y = RandomUnderSampler(sampling_strategy=config.RUS_RATIO,
                                      random_state=seed).fit_resample(X, y)
        return SMOTE(sampling_strategy=1.0, random_state=seed).fit_resample(X, y)
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
