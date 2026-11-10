"""Synthetic file with the TDD2022 layout (hourly rows, consumer type, theft label) for tests."""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--types", type=int, default=16)
ap.add_argument("--days", type=int, default=60)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = np.random.default_rng(0)
rows = []
h = np.arange(24)
for t in range(a.types):
    base = 50 + 30 * t + 40 * np.sin((h - 6) / 24 * 2 * np.pi).clip(0)
    for d in range(a.days):
        x = base * rng.uniform(0.8, 1.2) + rng.normal(0, 3, 24)
        k = int(rng.integers(0, 7)) if rng.random() < 0.4 else 0
        if k == 1:
            x = x * rng.uniform(0.1, 0.8)
        elif k == 2:
            s = rng.integers(0, 20); x[s:s + rng.integers(2, 5)] = 0
        elif k == 3:
            x = x * rng.uniform(0.1, 0.8, 24)
        elif k == 4:
            x = np.full(24, x.mean() * rng.uniform(0.1, 0.8))
        elif k == 5:
            x = np.full(24, x.mean())
        elif k == 6:
            x = x[::-1]
        for i in range(24):
            rows.append({"Electricity:Facility [kW](Hourly)": x[i], "Gas:Facility [kW](Hourly)": rng.random(),
                         "class": f"Type{t}", "theft": "Normal" if k == 0 else f"Theft{k}"})
pd.DataFrame(rows).to_csv(a.out)
