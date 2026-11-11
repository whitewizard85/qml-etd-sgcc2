#!/usr/bin/env bash
# Experiments added for the major revision. Every step is resumable (finished runs are skipped).
#
#   ./run_revision.sh                          # all steps, in this order
#   STEPS="robust shots" ./run_revision.sh     # only some steps
#
# Steps
#   robust      dependence-aware re-analysis of the existing 10-split results (seconds, no training)
#   shots       finite-shot experiment extended from 3 to 10 splits, then its statistics
#   sgcc_hold   SGCC, hold-out protocol: fixed untouched 20% test set, 5 inner folds
#   tdd_prep    TDD2022: daily series + features + consumer-type groups (needs data/raw/tdd2022/)
#   tdd_hold    TDD2022, hold-out protocol, consumer types never shared between partitions
#   eval        final evaluation of both hold-out campaigns on their test sets
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-robust shots sgcc_hold tdd_prep tdd_hold eval}"
D="2 4 6 8 10 13"; M="150 1040 full"; FOLDS="0 1 2 3 4"
ts() { date "+%F %T"; }
py() { echo "[$(ts)] ${QMLETD_DATASET:-sgcc}/${QMLETD_PROTOCOL:-splits}: python $*"; python -u "$@"; }

# The same models and the same (final) hyperparameter grids as in the paper.
campaign() {
  py scripts/01_classical.py --splits $FOLDS --d $D --M $M
  py scripts/08_cnn_raw.py --splits $FOLDS --device "$DEVICE"
  py scripts/02_qkernel.py --splits $FOLDS --d $D --M $M --models qk_fidelity qk_projected rbf_matched \
     --bandwidths 0.001 0.0025 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1.0 --C 0.1 1 10 100 1000 10000 \
     --proj-gammas 0.1 1 10 30 100 300 --rbf-mults 0.01 0.03 0.1 0.3 1 3 10 30 100 \
     --precision double --tag bwext2 --device "$DEVICE"
  py scripts/03_vqc.py --splits $FOLDS --d $D --M full --device "$DEVICE"
  py scripts/03_vqc.py --splits $FOLDS --d $D --M full --readouts local local_mean --reupload 0 1 --order mi --device "$DEVICE"
  py scripts/04_qmsvm.py --splits $FOLDS --d $D
}

step_robust()    { py scripts/09_aggregate.py --figs-only; py scripts/12_robust_stats.py; }
step_shots()     { py scripts/10_vqc_shots.py --d 4 8 10 13 --readouts global local_mean --shots exact 10000 1000 100 \
                     --splits 0 1 2 3 4 5 6 7 8 9 --device "$DEVICE"; py scripts/14_shots_stats.py; }
step_sgcc_hold() { ( export QMLETD_DATASET=sgcc QMLETD_PROTOCOL=holdout; campaign ); }
step_tdd_prep()  { ( export QMLETD_DATASET=tdd2022; py scripts/00b_prepare_tdd2022.py --day-label any ); }
step_tdd_hold()  { ( export QMLETD_DATASET=tdd2022 QMLETD_PROTOCOL=holdout; campaign ); }
step_eval() {
  ( export QMLETD_DATASET=sgcc QMLETD_PROTOCOL=holdout; py scripts/13_holdout_eval.py --boot 10000 )
  ( export QMLETD_DATASET=tdd2022 QMLETD_PROTOCOL=holdout; py scripts/13_holdout_eval.py --boot 10000 )
}
# steps run in the order given in STEPS
for st in $STEPS; do "step_$st"; done
echo "[$(ts)] done. Reports: results/summary/robust_report.md, results/summary/shots_stats.md,"
echo "          results_sgcc_holdout/summary/holdout_report.md, results_tdd2022_holdout/summary/holdout_report.md"
