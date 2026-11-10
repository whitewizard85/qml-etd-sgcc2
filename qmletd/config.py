"""Global configuration and helpers shared by all experiments."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

# Default location of the SGCC file (Kaggle / original release name).
SGCC_CSV = Path(os.environ.get("SGCC_CSV", DATA_RAW / "data set.csv"))

FEATURES_FILE = DATA_PROCESSED / "features.parquet"
SERIES_FILE = DATA_PROCESSED / "series.npy"          # interpolated daily series (float32)
MASK_FILE = DATA_PROCESSED / "missing_mask.npy"      # original missing-value mask (bool)
LABELS_FILE = DATA_PROCESSED / "labels.npy"

# Protocol constants (pre-registered; change only with a documented reason).
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
DEFAULT_SPLITS = list(range(10))        # 10 repeated stratified splits
PRIMARY_METRIC = "pr_auc"
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
