"""TDD2022 (Zidi et al. 2023): synthetic thefts injected into OEDI hourly consumption profiles.

Source: Mendeley Data, doi:10.17632/c3c7329tjj (CC BY 4.0). One row = one hour of one consumer
type; theft is applied to whole daily blocks (24 h) of the electricity channel only, and every
hour of a manipulated day carries the theft label (Alfaverh et al. 2026, Energies 19:2045).

Unit of analysis here = one DAY (24 hourly electricity readings), so that the same 13
features as for SGCC can be computed on a time series. Gas channels are not used (theft is
never injected there).

The exact column names are not documented on the dataset page, so the loader detects them
and `scripts/00b_prepare_tdd2022.py --inspect` shows what it found; every column can be
forced from the command line.

Leakage: theft days are transformations of normal days of the same consumer type. Splits
therefore keep whole consumer types together (groups.npy); `--group-by none` is available
only as a sensitivity analysis.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data

HOURS = 24


def find_file(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    cands = sorted([p for p in path.rglob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls")],
                   key=lambda p: -p.stat().st_size)
    if not cands:
        raise FileNotFoundError(f"no .csv/.xlsx file under {path}")
    return cands[0]


def read_table(path: Path, nrows=None) -> pd.DataFrame:
    path = find_file(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=nrows)
    return pd.read_excel(path, nrows=nrows)


def _norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def detect_columns(df: pd.DataFrame) -> dict:
    """Best-effort detection of the electricity, consumer-type and label columns."""
    cols = list(df.columns)
    out = {"elec": None, "type": None, "label": None, "theft_type": None}
    for c in cols:
        if _norm(c) in ("electricityfacility", "electricityfacilitykwhhourly"):
            out["elec"] = c
            break
    if out["elec"] is None:
        el = [c for c in cols if "electricity" in _norm(c) and "facility" in _norm(c)]
        out["elec"] = el[0] if el else None
    obj = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    for c in obj:
        vals = df[c].astype(str).str.lower()
        if vals.str.contains("theft").any() or vals.str.contains("normal").any():
            if out["label"] is None:
                out["label"] = c
            elif out["theft_type"] is None:
                out["theft_type"] = c
    for c in obj:
        if c in (out["label"], out["theft_type"]):
            continue
        n = df[c].nunique()
        if 2 < n <= 64:
            out["type"] = c
            break
    if out["label"] is None:                       # numeric label: 0 = normal, k = theft type
        for c in cols:
            if any(k in _norm(c) for k in ("theft", "label", "target", "class", "fraud")):
                if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 8:
                    out["label"] = c
                    break
    return out


def binary_label(v: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(v):
        return (v.to_numpy() != 0).astype(np.int8)
    s = v.astype(str).str.strip().str.lower()
    return (~s.isin(["normal", "0", "honest", "benign", "no", "false"])).astype(np.int8).to_numpy()


def daily_blocks(df, elec, typ, label, theft_type=None):
    """Cut each consumer type's rows (in file order) into consecutive 24-hour blocks.

    Returns series [n_days, 24], y [n_days], groups [n_days], theft [n_days], report dict.
    Blocks whose hourly labels are not constant are dropped and counted (this also tells us
    whether the 24-hour alignment is right: with a wrong offset most theft blocks are mixed).
    """
    X, Y, G, T = [], [], [], []
    n_mixed = n_tail = 0
    for g, d in df.groupby(typ, sort=False):
        e = pd.to_numeric(d[elec], errors="coerce").to_numpy(np.float64)
        yb = binary_label(d[label])
        tt = d[theft_type].astype(str).to_numpy() if theft_type else d[label].astype(str).to_numpy()
        n = len(d) // HOURS
        n_tail += len(d) - n * HOURS
        e, yb, tt = e[: n * HOURS].reshape(n, HOURS), yb[: n * HOURS].reshape(n, HOURS), \
            tt[: n * HOURS].reshape(n, HOURS)
        const = (yb == yb[:, :1]).all(axis=1) & (tt == tt[:, :1]).all(axis=1)
        n_mixed += int((~const).sum())
        X.append(e[const]); Y.append(yb[const, 0]); G.append(np.full(const.sum(), str(g), object))
        T.append(tt[const, 0])
    rep = {"days": int(sum(len(y) for y in Y)), "mixed_blocks_dropped": n_mixed,
           "tail_hours_dropped": n_tail}
    return np.vstack(X), np.concatenate(Y), np.concatenate(G), np.concatenate(T), rep


def prepare_and_cache(path, cols: dict, group_by: str = "type") -> None:
    if config.DATASET != "tdd2022":
        raise RuntimeError("set QMLETD_DATASET=tdd2022 before preparing TDD2022")
    config.ensure_dirs()
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = read_table(path)
    det = detect_columns(df)
    det.update({k: v for k, v in cols.items() if v})
    missing = [k for k in ("elec", "type", "label") if not det[k]]
    if missing:
        raise SystemExit(f"could not detect columns {missing}; run with --inspect and pass them "
                         f"explicitly. Columns: {list(df.columns)}")
    print(f"[tdd] columns: {det}")
    X, y, groups, theft, rep = daily_blocks(df, det["elec"], det["type"], det["label"],
                                            det["theft_type"])
    mask = np.isnan(X)
    filled, _ = data.interpolate_series(X)
    print(f"[tdd] {rep}; consumer types={len(set(groups))}; positives={y.mean():.4f}; "
          f"missing={mask.mean():.4%}")
    if rep["mixed_blocks_dropped"] > 0.02 * rep["days"]:
        print("[tdd] WARNING: >2% of 24-hour blocks have mixed labels: the rows may not be ordered "
              "by time within each consumer type, or days do not start at row 0. Check --inspect.")
    feats = data.engineer_features(filled, mask)
    feats.insert(0, "customer_id", [f"{g}#{i}" for i, g in enumerate(groups)])
    feats["label"] = y
    feats.to_parquet(config.FEATURES_FILE, index=False)
    np.save(config.SERIES_FILE, filled.astype(np.float32))
    np.save(config.MASK_FILE, mask)
    np.save(config.LABELS_FILE, y)
    np.save(config.DATA_PROCESSED / "theft_type.npy", theft.astype(str))
    np.save(config.DATA_PROCESSED / "consumer_type.npy", groups.astype(str))
    if group_by == "type":
        np.save(config.GROUPS_FILE, groups.astype(str))
    elif config.GROUPS_FILE.exists():
        config.GROUPS_FILE.unlink()
    pd.Series(theft).value_counts().to_csv(config.DATA_PROCESSED / "theft_type_counts.csv")
    print(f"[tdd] cached -> {config.DATA_PROCESSED} (group_by={group_by})")
