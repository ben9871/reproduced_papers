# QCNN Data Classification

This reproduction follows the standard MerLin template structure. It bundles two
complementary workflows:

1. **QCNN Benchmarks** — wrappers around the original TensorFlow/Keras QCNN
   implementation released with *Cong et al., 2019*. These scripts keep parity
   with the authors' code and summarise accuracies across multiple seeds.
2. **Merlin/Perceval Reproduction** — a self-contained PyTorch + Perceval
   rewrite of the parallel-columns photonic circuit using Merlin's
   `QuantumLayer`. It operates on PCA-compressed MNIST digits (0 vs 1) and can
   optionally compare quantum pseudo-convolutions against their classical
   counterparts while keeping identical kernel counts.

## Folder Layout

```
QCNN_data_classification/
├── configs/                # Ready-to-run JSON presets (see example.json)
├── data/                   # Local datasets, cache files, or preprocessing artefacts
├── lib/                    # Legacy reference scripts (e.g. legacy_merlin_reproduction.py)
├── model/                  # Source modules (SingleGI, QConvModel, ...)
├── models/                 # Checkpoints saved from experiments
├── results/                # CSV / JSON summaries, plots, etc.
├── tests/                  # Lightweight smoke tests (pytest-compatible)
├── utils/                  # Helper scripts and shell utilities
├── implementation.py       # Main CLI entrypoint (Merlin reproduction)
├── notebook.ipynb          # Scratchpad for exploratory analysis
├── README.md               # You are here
└── requirements.txt        # Minimal dependency list
```

## Quick Start

```bash
cd QCNN_data_classification
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Merlin / Perceval reproduction

Run the main experiment with the template-style entrypoint:

```bash
python implementation.py \
  --steps 200 \
  --seeds 3 \
  --n_modes 8 \
  --n_features 8 \
  --qconv_kernels 4 \
  --qconv_kernel_size 2 \
  --compare_classical
```

Key options (see `implementation.py` or modules under `model/`, `utils/`, and `data/` for details):

- `--model`: `qconv` (default) or `single` for the baseline single-layer GI.
- `--compare_classical`: evaluate quantum and classical pseudo-convolutions
  back-to-back using the very same kernel count and stride.
- `--qconv_kernel_size`, `--qconv_kernel_features`: must match so that every
  quantum kernel receives exactly the features it expects.
- `--n_modes`, `--n_features`: define the circuit depth/width. They must produce
  a required input size equal to the PCA dimension (or, for qconv, the kernel
  feature count).

You can also load hyperparameters from JSON and forward them to the CLI, e.g.:

```bash
python implementation.py --config configs/example.json
```

Command-line flags still override values loaded from the JSON file. Use
`python implementation.py --help` for the complete option list.

### Original QCNN benchmarking scripts

Utilities that interact with the official QCNN repository live under `utils/`:

- `utils/run_experiment.py`: multi-seed wrapper around `QCNN/Benchmarking.py`.
- `utils/run_exp.sh`: example shell sweep over the ansätze listed in the paper.

Place/clone the original QCNN repo so that `QCNN_data_classification/QCNN` is a
valid path (or ensure it is on `PYTHONPATH`). Then run, for example:

```bash
python utils/run_experiment.py --dataset mnist --classes 0,1 --ansatz 7 --encoding pca8
```

Results are appended to the upstream `QCNN/Result/` folder and summarised in
per-ansatz text files.

## Result Review

The directories `results-mnist/` and `results-fashionmnist/` contain MerLin
reproduction sweeps from 5 Nov 2025. Each run keeps the PCA dimension fixed at
8 while varying the number of quantum kernels per convolutional layer. Key
findings:

- **MNIST (digits 0 vs 1)** — 12 runs average `0.963 ± 0.047` mean accuracy.
  Accuracy improves sharply with capacity: one-kernel circuits average `0.915`
  mean accuracy, while four kernels climb to `0.994`. The best run
  (`run_20251105-180951`, four kernels / 56 modes) reaches `0.995` mean accuracy
  with only `0.002` standard deviation across seeds. The weakest configuration
  (`run_20251105-184526`, one kernel) falls to `0.824` mean accuracy, with seed
  outcomes spanning `0.668` to `0.967`, highlighting sensitivity to
  initialisation at low kernel counts.

Per-configuration metrics grouped by `qconv_kernels` and `qconv_kernel_modes`
(PCA = 8, one run per combination):

| qconv_kernels | qconv_kernel_modes | runs | mean_mean_acc | between_run_std | avg_seed_std | min_seed_acc | max_seed_acc |
|---------------|--------------------|------|---------------|-----------------|--------------|--------------|--------------|
| 1             | 8                  | 1    | 0.957         | 0.000           | 0.014        | 0.948        | 0.974        |
| 1             | 16                 | 1    | 0.963         | 0.000           | 0.025        | 0.936        | 0.987        |
| 1             | 18                 | 1    | 0.824         | 0.000           | 0.150        | 0.668        | 0.967        |
| 2             | 8                  | 1    | 0.921         | 0.000           | 0.085        | 0.823        | 0.979        |
| 2             | 16                 | 1    | 0.975         | 0.000           | 0.010        | 0.967        | 0.986        |
| 2             | 18                 | 1    | 0.970         | 0.000           | 0.007        | 0.963        | 0.977        |
| 3             | 8                  | 1    | 0.985         | 0.000           | 0.015        | 0.969        | 0.998        |
| 3             | 16                 | 1    | 0.985         | 0.000           | 0.013        | 0.971        | 0.996        |
| 3             | 18                 | 1    | 0.991         | 0.000           | 0.007        | 0.983        | 0.996        |
| 4             | 8                  | 1    | 0.995         | 0.000           | 0.002        | 0.993        | 0.997        |
| 4             | 16                 | 1    | 0.994         | 0.000           | 0.002        | 0.992        | 0.996        |
| 4             | 18                 | 1    | 0.994         | 0.000           | 0.004        | 0.989        | 0.997        |

- **Fashion-MNIST (classes 0 vs 1)** — 12 runs average `0.930 ± 0.029` mean
  accuracy. As on MNIST, one-kernel circuits underperform (`0.900` mean accuracy
  and `0.171` seed-level deviation), whereas two to four kernels stay in the
  mid-`0.93` range. The best run (`run_20251105-162727`, two kernels / 28 modes)
  records `0.953` mean accuracy with modest variance, while the weakest
  (`run_20251105-171858`, one kernel) drops to `0.842` after a seed stalls at
  `0.645`.

Per-configuration metrics grouped by `qconv_kernels` and `qconv_kernel_modes`
(PCA = 8, one run per combination):

| qconv_kernels | qconv_kernel_modes | runs | mean_mean_acc | between_run_std | avg_seed_std | min_seed_acc | max_seed_acc |
|---------------|--------------------|------|---------------|-----------------|--------------|--------------|--------------|
| 1             | 8                  | 1    | 0.938         | 0.000           | 0.009        | 0.930        | 0.948        |
| 1             | 16                 | 1    | 0.919         | 0.000           | 0.042        | 0.872        | 0.955        |
| 1             | 18                 | 1    | 0.842         | 0.000           | 0.171        | 0.644        | 0.944        |
| 2             | 8                  | 1    | 0.953         | 0.000           | 0.014        | 0.937        | 0.963        |
| 2             | 16                 | 1    | 0.925         | 0.000           | 0.016        | 0.913        | 0.943        |
| 2             | 18                 | 1    | 0.936         | 0.000           | 0.009        | 0.927        | 0.945        |
| 3             | 8                  | 1    | 0.947         | 0.000           | 0.008        | 0.942        | 0.956        |
| 3             | 16                 | 1    | 0.920         | 0.000           | 0.045        | 0.869        | 0.951        |
| 3             | 18                 | 1    | 0.939         | 0.000           | 0.013        | 0.924        | 0.948        |
| 4             | 8                  | 1    | 0.951         | 0.000           | 0.009        | 0.946        | 0.961        |
| 4             | 16                 | 1    | 0.949         | 0.000           | 0.007        | 0.943        | 0.957        |
| 4             | 18                 | 1    | 0.942         | 0.000           | 0.010        | 0.933        | 0.952        |

Overall, the sweeps confirm that adding quantum kernels stabilises training and
pushes accuracies close to classical baselines on both datasets, whereas
low-capacity circuits exhibit large variance and occasional collapse.

## Tests

Execute smoke tests with:

```bash
pytest tests
```

They confirm that the reproduction entry points import correctly. Extend this
folder with dataset-specific or regression tests as the reproduction evolves.

## Notes

- Keep virtual environments outside of the reproduction directory whenever
  possible (`.venv/` is suggested in the quick start above but can live at a
  higher level).
- Large result files belong in `results/`; checkpoints in `models/`.
- The included `notebook.ipynb` is a minimal placeholder to encourage tracking
  exploratory work close to the codebase.
