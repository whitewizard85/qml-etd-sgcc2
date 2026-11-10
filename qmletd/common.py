"""Shared CLI options and the prepared-data cache used by all experiment scripts."""
from __future__ import annotations

import argparse

import numpy as np

from . import config, data, prep


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--splits", type=int, nargs="+", default=config.DEFAULT_SPLITS)
    p.add_argument("--d", type=int, nargs="+", default=[2, 4, 8, 13],
                   help="number of features = number of qubits")
    p.add_argument("--M", type=str, nargs="+", default=["full"],
                   help="training-set sizes after balancing, e.g. 150 1040 full")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--tag", default="")
    return p


def parse_M(values):
    return [None if v == "full" else int(v) for v in values]


class DataCache:
    """Loads features once and caches preprocessed partitions per (split, d)."""

    def __init__(self, balance_mode="rus_smote"):
        self.X, self.y = data.load_features()
        self.balance_mode = balance_mode
        self._rank = {}
        self._prep = {}

    def get(self, split: int, d: int) -> prep.Prepared:
        key = (split, d)
        if key not in self._prep:
            self._prep[key] = prep.preprocess(self.X, self.y, split, d, self.balance_mode,
                                              ranking_cache=self._rank)
        return self._prep[key]

    def train_subset(self, P: prep.Prepared, M):
        """Matched subsample: identical for every model given (split, d, M)."""
        return prep.stratified_subsample(P.X_tr, P.y_tr, M, seed=P.split)

    def tune_subset(self, X, y, tune_max, split):
        if tune_max is None or len(y) <= tune_max:
            return None, None
        return prep.stratified_subsample(X, y, tune_max, seed=split + 31)


def meta(P: prep.Prepared, model: str, M, n_train: int, **extra) -> dict:
    return {"model": model, "split": P.split, "d": P.d, "M": "full" if M is None else M,
            "n_train": int(n_train), "feature_idx": P.feature_idx.tolist(),
            "feature_names": [data.FEATURE_NAMES[i] for i in P.feature_idx], **extra}


def score_arrays(P: prep.Prepared, s_va, s_te) -> dict:
    return {"y_val": P.y_va.astype(np.int8), "s_val": np.asarray(s_va, np.float64),
            "y_test": P.y_te.astype(np.int8), "s_test": np.asarray(s_te, np.float64)}
