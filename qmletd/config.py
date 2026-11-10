"""Global configuration and helpers shared by all experiments."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"

# ----------------------------------------------------------------------------------------
# Dataset and evaluation protocol are selected with environment variables, so that every
# experiment script runs unchanged under each combination:
#   QMLETD_DATASET  = sgcc (default) | tdd2022
#   QMLETD_PROTOCOL = splits (default: 10 repeated stratified 60/20/20 splits)
#                   | holdout (one fixed, untouched 20% test set; K-fold CV on the rest,
#                              "split" s = inner fold s used for validation)
DATASET = os.environ.get("QMLETD_DATASET", "sgcc").lower()
PROTOCOL = os.environ.get("QMLETD_PROTOCOL", "splits").lower()
if DATASET not in ("sgcc", "tdd2022"):
    raise ValueError(f"QMLETD_DATASET={DATASET!r}")
if PROTOCOL not in ("splits", "holdout"):
    raise ValueError(f"QMLETD_PROTOCOL={PROTOCOL!r}")

# The original SGCC/splits layout is kept unchanged for backward compatibility.
DATA_PROCESSED = ROOT / "data" / "processed" if DATASET == "sgcc" else ROOT / "data" / "processed" / DATASET
_default_results = ROOT / "results" if (DATASET, PROTOCOL) == ("sgcc", "splits") \
    else ROOT / f"results_{DATASET}_{PROTOCOL}"
RESULTS = Path(os.environ.get("QMLETD_RESULTS", _default_results))

# Default location of the SGCC file (Kaggle / original release name).
SGCC_CSV = Path(os.environ.get("SGCC_CSV", DATA_RAW / "data set.csv"))
TDD_FILE = Path(os.environ.get("TDD_FILE", DATA_RAW / "tdd2022"))   # file or folder

FEATURES_FILE = DATA_PROCESSED / "features.parquet"
SERIES_FILE = DATA_PROCESSED / "series.npy"          # interpolated series (float32)
MASK_FILE = DATA_PROCESSED / "missing_mask.npy"      # original missing-value mask (bool)
LABELS_FILE = DATA_PROCESSED / "labels.npy"
GROUPS_FILE = DATA_PROCESSED / "groups.npy"          # optional: split units that must not be separated

# Protocol constants (pre-specified; change only with a documented reason).
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
HOLDOUT_TEST_FRACTION = 0.2
HOLDOUT_SEED = 20261110                 # fixes the untouched test set once and for all
HOLDOUT_FOLDS = 5
DEFAULT_SPLITS = list(range(10)) if PROTOCOL == "splits" else list(range(HOLDOUT_FOLDS))
PRIMARY_METRIC = "pr_auc"
EQUIVALENCE_MARGIN = 0.02               # PR-AUC
BUDGETS = (0.01, 0.05, 0.10)            # inspection budgets for precision@k / recall@k
RUS_RATIO = 0.30                        # thesis protocol: RUS to 0.30, then SMOTE to 1.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str = "auto"):
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cuda":
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cpu")


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROCESSED, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
