# Quantum machine learning for electricity-theft detection under matched evaluation

Code accompanying the article *"Quantum machine learning for electricity-theft detection under matched evaluation:
kernel concentration, readout light cones and finite-shot training"* (G. Grimaldi, A. Esposito; submitted to
*Quantum Machine Intelligence*).

It benchmarks fidelity and projected quantum kernels, variational quantum classifiers (VQCs) and a QUBO formulation of
the SVM against classical controls that share **features, training subsets, tuning protocol and decision-threshold
rule**, on the public SGCC smart-meter dataset, with 10 repeated stratified splits, paired tests (Holm correction),
TOST equivalence tests, DeLong tests and paired bootstrap.

## Contents

| Path | Purpose |
|---|---|
| `qmletd/sim.py` | Batched exact statevector simulator (PyTorch, CPU/GPU): ZZ feature map, Pauli expectations, gates |
| `qmletd/kernels.py` | Fidelity / projected quantum kernels, kernel diagnostics (concentration, KTA, geometric difference) |
| `qmletd/vqc.py` | VQC with local / local-mean / global readout, exact-gradient training, gradient-variance diagnostics |
| `qmletd/shots.py` | VQC training with finite-shot parameter-shift gradients |
| `qmletd/qubo_svm.py` | QUBO-SVM (Willsch et al. 2020) with bagging, seeded simulated annealing, matched exact-SVM control |
| `qmletd/baselines.py` | Logistic regression, SVM, random forest, LightGBM, MLP, 1D CNN on raw series |
| `qmletd/prep.py`, `metrics.py`, `stats.py` | Splits (repeated or hold-out, optionally grouped), exact F2 threshold search, statistical tests (incl. Nadeau-Bengio corrected resampled t-test, paired bootstrap on a fixed test set) |
| `qmletd/tdd.py` | TDD2022 loader: hourly rows to daily 24-hour series, consumer-type groups |
| `scripts/00`–`11` | One script per experiment (see `reproduce_paper.sh`) |
| `scripts/00b`, `12`–`14` | TDD2022 preparation, dependence-aware statistics, hold-out evaluation, finite-shot statistics (see `run_revision.sh`) |
| `tests/` | Unit tests: simulator vs Qiskit (1e-10), parameter shift vs autograd, QUBO vs SVM dual, metrics |

## Installation

Ubuntu 22.04/24.04, Python >= 3.11. A CUDA GPU is recommended for the VQC experiments.

```bash
./setup_env.sh          # creates .venv, installs dependencies (CUDA build of torch if nvidia-smi works), runs the tests
source .venv/bin/activate
python -m pytest -q tests     # 27 tests
```
Versions used for the paper: Python 3.12, torch 2.x (CUDA), qiskit 2.x, qiskit-aer 0.17+, scikit-learn 1.7+,
dwave-samplers 1.x (exact versions are stored in every result file, field `env`).

## Data

The SGCC dataset (42,372 customers, 1,034 days, binary label `FLAG`) is not redistributed here. Place the CSV file in
`data/raw/data set.csv` or set `SGCC_CSV=/path/to/file.csv`.

Source: <https://github.com/henryRDlab/ElectricityTheftDetection>. The CSV is distributed as a split archive
(`data.zip`, `data.z01`, `data.z02`); download all three parts into one folder and extract them together, e.g.
`7z x data.zip` (or `zip -s 0 data.zip --out full.zip && unzip full.zip`). If you use the dataset, cite:

> Z. Zheng, Y. Yang, X. Niu, H.-N. Dai, Y. Zhou, "Wide and Deep Convolutional Neural Networks for Electricity-Theft
> Detection to Secure Smart Grids," *IEEE Transactions on Industrial Informatics*, 14(4):1606–1615, 2018.
> doi:10.1109/TII.2017.2785963
Date columns are sorted chronologically before interpolation (in the public CSV they are not guaranteed to be ordered).

Per-run outputs of the paper (metrics, selected hyperparameters, test-set scores):
<https://doi.org/10.5281/zenodo.24109562>.

## Reproducing the paper

```bash
QUICK=1 ./run_all.sh        # 5-10 min smoke test on synthetic data with the SGCC layout (no real data needed)
rm -rf results/* data/processed/*
./reproduce_paper.sh        # exact sequence of the paper's experiments (resumable)
```
Outputs: `results/<experiment>/<run>.json` (hyper-parameters, metrics, timings, library versions) and `.npz` (validation
and test scores). `scripts/09_aggregate.py` writes `results/summary/` (tables with 95% CIs, `comparisons.csv` with all
paired tests, figures). `scripts/11_paper_assets.py` writes the LaTeX tables and figures used in the article.

Wall-clock times on 32 vCPUs + NVIDIA RTX 5000 Ada: classical baselines 27 min, CNN 3.5 min, kernels (first grid)
34 min, VQC 2 h 40 min, QUBO-SVM 3 h 37 min, diagnostics 25 min, finite-shot VQC about 5 min per run at 13 qubits.

## Hold-out protocol and second dataset

Two environment variables select the dataset and the evaluation protocol; every experiment script runs unchanged:

| Variable | Values | Effect |
|---|---|---|
| `QMLETD_DATASET` | `sgcc` (default), `tdd2022` | processed data in `data/processed/` or `data/processed/tdd2022/` |
| `QMLETD_PROTOCOL` | `splits` (default), `holdout` | 10 repeated 60/20/20 splits, or one fixed untouched 20% test set with 5 inner folds (`--splits 0..4`) |

Results go to `results/` (SGCC, splits) or `results_<dataset>_<protocol>/`. Under `holdout`, each fold model tunes
hyperparameters and the F2 threshold on its own validation fold; `scripts/13_holdout_eval.py` averages the fold models
(scores standardised with validation statistics only) and compares models on the same test units with a stratified
paired bootstrap, with Holm correction over a pre-specified primary family.

TDD2022 (Zidi et al. 2023, Mendeley Data, doi:10.17632/c3c7329tjj, CC BY 4.0): download all files from the Mendeley page,
put them in `data/raw/tdd2022/` (unzipped), then `QMLETD_DATASET=tdd2022 python scripts/00b_prepare_tdd2022.py --inspect`.
The unit is one day (24 hourly readings of the electricity channel); thefts are synthetic. Whole consumer types are kept
in one partition, so no test day is a transformation of a training day of the same consumer type.

```bash
./run_revision.sh           # robust statistics, 10-split finite-shot, SGCC hold-out, TDD2022 hold-out, evaluation
```

## Notes and limitations

* Real-hardware scripts (`06_noisy_kernel.py --ibm-backend`, `07_dwave_qmsvm.py`) were not run for the paper; the
  simulated-noise path of `06` was.
* The kernel experiments were run three times with increasingly wide hyper-parameter grids (tags: none, `bwext`,
  `bwext2`). The paper uses `bwext2`, where quantum kernels and the RBF control share the extended grids; the first
  campaign is reported as an example of how a narrow grid produces a spurious degradation with qubit count.
* The SVM parameter C reached the upper end of its grid (1e4) in many kernel runs, for quantum and classical kernels alike.

## License and citation

Code released under the MIT License (see `LICENSE`). If you use it, please cite the article (see `CITATION.cff`).
