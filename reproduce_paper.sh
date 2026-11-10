#!/usr/bin/env bash
# Exact sequence of commands that produced the results reported in the paper
# (two kernel campaigns included: the narrow-grid one is reported as a pitfall).
# Every step is resumable: runs whose results/<experiment>/<name>.json exists are skipped.
#
#   ./reproduce_paper.sh              # everything, in order (about one day on 32 vCPU + 1 GPU)
#   STEPS="kernels_final agg" ./reproduce_paper.sh
#
# Requires the SGCC file in data/raw/ (see README) and a CUDA GPU for the VQC steps (DEVICE=cpu works, slowly).
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-prep classical cnn kernels_narrow vqc qubo diag_grad diag_kern noise vqc_mi diag_grad_lm shots kernels_wide1 kernels_final agg}"
SPLITS="0 1 2 3 4 5 6 7 8 9"; D="2 4 6 8 10 13"; M="150 1040 full"
has() { [[ " $STEPS " == *" $1 "* ]]; }
ts() { date "+%F %T"; }
py() { echo "[$(ts)] python $*"; python -u "$@"; }

has prep        && py scripts/00_prepare_data.py --csv "${SGCC_CSV:-data/raw/data set.csv}"
# E1: classical baselines and CNN on raw series
has classical   && py scripts/01_classical.py --splits $SPLITS --d $D --M $M
has cnn         && py scripts/08_cnn_raw.py --splits $SPLITS --device "$DEVICE"
# E2 kernels, first campaign (narrow grid: bandwidth >= 0.1, C <= 100) - reported as a pitfall
has kernels_narrow && py scripts/02_qkernel.py --splits $SPLITS --d $D --M $M --device "$DEVICE"
# E2 VQC (local/global readout, re-upload 0/1, features in column order) and QUBO-SVM
has vqc         && py scripts/03_vqc.py --splits $SPLITS --d $D --M full --device "$DEVICE"
has qubo        && py scripts/04_qmsvm.py --splits $SPLITS --d $D
# E3 diagnostics
has diag_grad   && py scripts/05_diagnostics.py --part gradvar --d 2 4 6 8 10 12 14 --depths 1 2 4 8 --n-init 200 --splits 0 1 2 --device "$DEVICE"
has diag_kern   && py scripts/05_diagnostics.py --part kernels --d $D --n 800 --splits 0 1 2 --device "$DEVICE"
# E5 noisy fidelity kernel (simulated IBM backend)
has noise       && py scripts/06_noisy_kernel.py --d 4 --n-train 60 --n-test 120 --shots 2000
# VQC with features ordered by mutual information and local-mean readout
has vqc_mi      && py scripts/03_vqc.py --splits $SPLITS --d $D --M full --readouts local local_mean --reupload 0 1 --order mi --device "$DEVICE"
has diag_grad_lm && py scripts/05_diagnostics.py --part gradvar --readouts local_mean --d 2 4 6 8 10 12 14 --depths 1 2 4 8 --n-init 200 --splits 0 1 2 --device "$DEVICE"
# E4 finite-shot training
has shots       && py scripts/10_vqc_shots.py --d 4 8 10 13 --readouts global local_mean --shots exact 10000 1000 100 --splits 0 1 2 --device "$DEVICE"
# Kernels, intermediate grid (bandwidth >= 0.01) - superseded by the final grid
if has kernels_wide1; then
  py scripts/02_qkernel.py --splits $SPLITS --d $D --M $M --models qk_fidelity qk_projected \
     --bandwidths 0.01 0.025 0.05 0.1 0.25 0.5 1.0 --tag bwext --device "$DEVICE"
  py scripts/05_diagnostics.py --part kernels --d $D --n 800 --splits 0 1 2 --bandwidths 0.01 0.025 0.05 --device "$DEVICE"
fi
# Kernels, final grid (quantum kernels AND classical control), double precision - used in the paper
if has kernels_final; then
  py scripts/02_qkernel.py --splits $SPLITS --d $D --M $M --models qk_fidelity qk_projected rbf_matched \
     --bandwidths 0.001 0.0025 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1.0 --C 0.1 1 10 100 1000 10000 \
     --proj-gammas 0.1 1 10 30 100 300 --rbf-mults 0.01 0.03 0.1 0.3 1 3 10 30 100 \
     --precision double --tag bwext2 --device "$DEVICE"
  py scripts/05_diagnostics.py --part kernels --d $D --n 800 --splits 0 1 2 --bandwidths 0.001 0.0025 0.005 --device "$DEVICE"
fi
# Aggregation, statistics, figures and the LaTeX tables of the paper
if has agg; then
  py scripts/09_aggregate.py --bootstrap 300
  py scripts/11_paper_assets.py --summary results/summary --out paper
fi
echo "[$(ts)] done"
