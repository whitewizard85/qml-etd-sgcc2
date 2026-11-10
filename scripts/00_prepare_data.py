"""Step 0: read the SGCC CSV, sort dates, interpolate, build the 13 features, cache to disk."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import config, data  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.SGCC_CSV))
    a = ap.parse_args()
    data.prepare_and_cache(a.csv)
