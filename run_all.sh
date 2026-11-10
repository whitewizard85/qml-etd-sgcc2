#!/usr/bin/env bash
# Full experimental campaign. Every step is resumable: finished runs are skipped
# (results/*/*.json already present) unless you add --overwrite to that step.
#
#   ./run_all.sh                 # full campaign on the real SGCC data
#   QUICK=1 ./run_all.sh         # ~5-10 min smoke test on synthetic data (no SGCC needed)
#   STEPS="e1 e2k" ./run_all.sh  # only some steps
#
# Long runs: launch inside tmux/screen, e.g.  tmux new -s qml './run_all.sh 2>&1 | tee run.log'
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

DEVICE="${DEVICE:-auto}"
STEPS="${STEPS:-prep e1 e1cnn e2k e2v e2q e3g e3k e4 e5 agg}"

if [ "${QUICK:-0}" = "1" ]; then
  echo ">>> QUICK smoke test on synthetic data"
  python tests/make_synthetic_sgcc.py --n 3000 --out data/raw/synthetic.csv
  export SGCC_CSV=data/raw/synthetic.csv
  SPLITS="0 1 2"; D="2 4 8"; M="150 full"
  E2K_EXTRA="--bandwidths 0.25 1.0 --C 1 10 --proj-gammas 1 --tune-max 500"
  E2V_EXTRA="--depths 1 2 --lrs 0.05 --epochs 4 --tune-epochs 2 --tune-max 600"
  E2Q_EXTRA="--K 2 --xi 1 --gamma-mult 1 --n-subsets 6 --subset-size 60 --num-reads 50 --num-sweeps 500 --tune-subsets 2"
  E3G_EXTRA="--d 2 4 6 8 --depths 1 4 --n-init 50 --n-samples 64"
  E3K_EXTRA="--d 2 4 6 8 --n 300 --bandwidths 0.1 0.5 1.0"
  E5_EXTRA="--d 3 --n-train 16 --n-test 20 --shots 500"
  CNN_EXTRA="--epochs 3"
  E4_EXTRA="--splits 0 --d 4 6 --shots exact 1000 100 --steps 50 --train-max 800"
else
  SPLITS="0 1 2 3 4 5 6 7 8 9"; D="2 4 6 8 10 13"; M="150 1040 full"
  E2K_EXTRA=""
  E2V_EXTRA=""
  E2Q_EXTRA=""
  E3G_EXTRA="--d 2 4 6 8 10 12 14 --depths 1 2 4 8 --n-init 200 --splits 0 1 2"
  E3K_EXTRA="--d 2 4 6 8 10 13 --n 800 --splits 0 1 2"
  E5_EXTRA="--d 4 --n-train 60 --n-test 120 --shots 2000"
  CNN_EXTRA=""
fi

has() { [[ " $STEPS " == *" $1 "* ]]; }
ts() { date "+%F %T"; }

has prep  && { echo "[$(ts)] prep";  python scripts/00_prepare_data.py --csv "${SGCC_CSV:-data/raw/data set.csv}"; }
has e1    && { echo "[$(ts)] E1 classical"; python scripts/01_classical.py --splits $SPLITS --d $D --M $M; }
has e1cnn && { echo "[$(ts)] E1b CNN raw";  python scripts/08_cnn_raw.py --splits $SPLITS --device "$DEVICE" $CNN_EXTRA; }
has e2k   && { echo "[$(ts)] E2 quantum kernels"; python scripts/02_qkernel.py --splits $SPLITS --d $D --M $M --device "$DEVICE" $E2K_EXTRA; }
has e2v   && { echo "[$(ts)] E2b VQC"; python scripts/03_vqc.py --splits $SPLITS --d $D --M full --device "$DEVICE" $E2V_EXTRA; }
has e2q   && { echo "[$(ts)] E2c QUBO-SVM"; python scripts/04_qmsvm.py --splits $SPLITS --d $D $E2Q_EXTRA; }
has e3g   && { echo "[$(ts)] E3 gradient variance"; python scripts/05_diagnostics.py --part gradvar --device "$DEVICE" $E3G_EXTRA; }
has e3k   && { echo "[$(ts)] E3 kernel diagnostics"; python scripts/05_diagnostics.py --part kernels --device "$DEVICE" $E3K_EXTRA; }
has e4    && { echo "[$(ts)] E4 finite-shot VQC"; python scripts/10_vqc_shots.py --device "$DEVICE" ${E4_EXTRA:-}; }
has e5    && { echo "[$(ts)] E5 noisy kernel (fake backend)"; python scripts/06_noisy_kernel.py $E5_EXTRA; }
has agg   && { echo "[$(ts)] aggregate"; python scripts/09_aggregate.py; }
echo "[$(ts)] done. See results/summary/REPORT.md and the fig*.pdf files."
