"""Step 0b: TDD2022 -> daily 24-hour series, the same 13 features as SGCC, consumer-type groups.

  export QMLETD_DATASET=tdd2022
  python scripts/00b_prepare_tdd2022.py --inspect                 # look at the file first
  python scripts/00b_prepare_tdd2022.py                           # build the cache
  python scripts/00b_prepare_tdd2022.py --elec-col "Electricity:Facility [kW](Hourly)" \
         --type-col Class --label-col Theft                       # if detection fails

Input: TDD_FILE (default data/raw/tdd2022/, a file or a folder containing it).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import config, tdd  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(config.TDD_FILE))
    ap.add_argument("--inspect", action="store_true", help="print columns, detection and value counts")
    ap.add_argument("--elec-col")
    ap.add_argument("--type-col")
    ap.add_argument("--label-col")
    ap.add_argument("--theft-type-col")
    ap.add_argument("--group-by", default="type", choices=["type", "none"],
                    help="keep consumer types together in splits (default) or not (sensitivity only)")
    ap.add_argument("--day-label", default="pure", choices=["pure", "any"],
                    help="pure: drop days mixing normal and theft hours; any: theft if any hour is theft")
    ap.add_argument("--layout", action="store_true", help="only print the label-run diagnostics")
    a = ap.parse_args()
    if a.inspect:
        f = tdd.find_file(Path(a.file))
        df = tdd.read_table(f)
        print(f"file: {f}\nshape: {df.shape}\n\ncolumns and dtypes:\n{df.dtypes.to_string()}\n")
        print(f"first rows:\n{df.head(30).to_string()}\n")
        det = tdd.detect_columns(df)
        print(f"detected: {det}\n")
        for k in ("type", "label", "theft_type"):
            if det[k]:
                print(f"value counts of {det[k]!r}:\n{df[det[k]].value_counts().head(20).to_string()}\n")
        sys.exit(0)
    if a.layout:
        df = tdd.read_table(Path(a.file))
        det = tdd.detect_columns(df)
        det.update({k: v for k, v in {"elec": a.elec_col, "type": a.type_col, "label": a.label_col}.items() if v})
        df = tdd.drop_invalid_types(df, det["type"])
        print(tdd.label_runs_report(df, det["type"], det["label"]))
        for g, d in list(df.groupby(det["type"], sort=False))[:2]:
            v = d[det["label"]].astype(str).to_numpy()
            st = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
            print(g, "first runs (value, start, length):",
                  [(v[i], int(i), int(j - i)) for i, j in zip(st[:12], np.r_[st[1:13], len(v)])])
        sys.exit(0)
    tdd.prepare_and_cache(Path(a.file), {"elec": a.elec_col, "type": a.type_col, "label": a.label_col,
                                         "theft_type": a.theft_type_col}, group_by=a.group_by,
                          day_label=a.day_label)
