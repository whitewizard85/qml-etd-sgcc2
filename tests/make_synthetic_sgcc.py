"""Generate a synthetic CSV with the same layout as SGCC 'data set.csv' (shuffled date
columns, CONS_NO, FLAG, NaNs) to test the full pipeline before using the real data.

  python tests/make_synthetic_sgcc.py --n 3000 --out data/raw/synthetic.csv
"""
import argparse

import numpy as np
import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--days", type=int, default=1034)
    ap.add_argument("--prevalence", type=float, default=0.0853)
    ap.add_argument("--out", default="data/raw/synthetic.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    dates = pd.date_range("2014-01-01", periods=a.days, freq="D")
    y = (rng.random(a.n) < a.prevalence).astype(int)
    base = rng.lognormal(1.5, 0.8, size=(a.n, 1))
    season = 1 + 0.3 * np.sin(2 * np.pi * np.arange(a.days) / 365.0)
    X = base * season * rng.lognormal(0, 0.35, size=(a.n, a.days))
    pos = np.flatnonzero(y)
    thieves = pos[rng.random(len(pos)) < 0.6]         # 40% of positive labels carry no pattern
    for i in thieves:                                 # theft: mild drop / sporadic zeros
        t0 = rng.integers(100, a.days - 100)
        X[i, t0:] *= rng.uniform(0.5, 0.95)
        z = rng.random(a.days - t0) < rng.uniform(0.0, 0.08)
        X[i, t0:][z] = 0
    X[rng.random(X.shape) < 0.25] = np.nan
    cols = [f"{d.year}/{d.month}/{d.day}" for d in dates]
    df = pd.DataFrame(X, columns=cols)
    df = df[list(rng.permutation(cols))]             # shuffled column order, as in SGCC
    df.insert(0, "FLAG", y)
    df.insert(0, "CONS_NO", [f"C{i:06d}" for i in range(a.n)])
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {a.n} customers, prevalence {y.mean():.4f}")
