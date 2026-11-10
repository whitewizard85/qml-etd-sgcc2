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
    tdd.prepare_and_cache(Path(a.file), {"elec": a.elec_col, "type": a.type_col, "label": a.label_col,
                                         "theft_type": a.theft_type_col}, group_by=a.group_by)
