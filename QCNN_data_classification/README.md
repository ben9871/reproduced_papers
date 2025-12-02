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
We propose a photonic version of the kernel, sliding on the PCA components:

![Alt text](Photonic_QConv.png)

Run the main experiment with the template-style entrypoint:

```bash
python implementation.py \
  --steps 200 \
  --seeds 3 \
  --n_modes 8 \
  --n_features 8 \
  --nb_kernels 4 \
  --kernel_size 2 \
  --compare_classical
```

Key options (see `implementation.py` or modules under `model/`, `utils/`, and `data/` for details):

- `--model`: `qconv` (default) or `single` for the baseline single-layer GI.
- `--compare_classical`: evaluate quantum and classical pseudo-convolutions
  back-to-back using the very same kernel count and stride.
- `--nb_kernels`: number of trainable quantum convolution kernels (i.e.
  parallel photonic filters applied across the input patch grid).
- `--kernel_modes`: number of optical modes allocated per quantum kernel;
  it sets the circuit width for each filter and affects both parameter count and
  expressive power.
- `--kernel_size`: sliding patch width (and feature count) consumed by each
  kernel. This value is shared between the classical and quantum variants.
- `--conv_classical`: skip the quantum kernels entirely and train the classical
  pseudo-convolution only.
- `--n_modes`, `--n_features`: define the circuit depth/width. They must produce
  a required input size equal to the PCA dimension (or, for qconv, the kernel
  size).

### Model-specific parameter cheatsheet

**Single GI (`--model single`)**
- `--n_modes` / `--n_features`: configure the Gaussian interferometer width and depth; they must match the PCA dimension exactly.
- `--n_photons`, `--reservoir_mode`, `--state_pattern`: photonic hardware settings replicated from the original QCNN work.
- `--angle_scale`: rescales normalized inputs before they hit the circuit, matching the paper’s experimentation knobs.

**Quantum convolution (`--model qconv` with quantum kernels enabled)**
- `--nb_kernels`: number of parallel photonic kernels that slide across the PCA vector.
- `--kernel_size`: number of PCA features per patch (and per optical kernel input).
- `--stride`: displacement between consecutive patches; mirrors 1D convolution stride.
- `--kernel_modes`: optical modes allocated per kernel; defaults to `kernel_size` but can be larger for deeper photonic circuits.
- `--compare_classical`: evaluates the quantum kernels alongside their classical counterparts for apples-to-apples reporting.

**Classical qconv variant (`--conv_classical` or implicit when comparing)**
- Shares `--nb_kernels`, `--kernel_size`, and `--stride` with the quantum version so both models consume identical patches.
- Each kernel is a learnable dense weight vector with optional bias; enable via `--conv_classical` to train only the classical branch or combine with `--compare_classical` to log both.

You can also load hyperparameters from JSON and forward them to the CLI, e.g.:

```bash
python implementation.py --config configs/example.json
```

Command-line flags still override values loaded from the JSON file. Use
`python implementation.py --help` for the complete option list.

### QConv model

The default `qconv` model emulates a 1D convolution by sliding PCA-compressed
patches across the input vector. Each kernel can be either:

- **Quantum** — a photonic Gaussian interferometer (`QuantumLayer`) that acts on
  `kernel_modes` optical modes and consumes `kernel_size` features per patch.
  The outputs of all kernels are concatenated and fed to a linear classifier.
- **Classical** — a learnable weight vector of size `kernel_size`, mirroring
  the quantum receptive field so side-by-side comparisons remain fair.

The stride defaults to 1, so the model evaluates every overlapping patch. The
quantum kernels always expect `kernel_size` features, so the PCA dimension must
be large enough to supply that many inputs per patch.

To run the QConv model end-to-end:

1. **Activate your environment** and install dependencies (`pip install -r requirements.txt`).
2. **Choose a dataset** (`mnist`, `fashionmnist`, `kmnist`) and PCA dimension
   (`--pca_dim` must match `--n_features`/`--n_modes` requirements).
3. **Launch a quantum-only experiment**, e.g.:

   ```bash
   python implementation.py \
     --dataset fashionmnist \
     --pca_dim 8 \
     --nb_kernels 3 \
     --kernel_modes 16 \
     --kernel_size 2 \
     --steps 200 \
     --seeds 3
   ```

   This saves summaries under `results-fashionmnist/`.

4. **(Optional) Compare against the classical analogue** by adding
   `--compare_classical`, or force a classical-only run with `--conv_classical`.

5. **Inspect outputs**: every run logs its configuration in
   `results-<dataset>/run_<timestamp>/`. `run_summary.json` collects per-seed
   accuracies and effective feature dimensions.

## Choice of model

To chose the right parameters for our model, we proceeded with an MNIST (8-PCA components) hyperparameter sweep whose processed records live under `QCNN_data_classification/plots-mnist/outputs/` and whose visual summaries are rendered to `QCNN_data_classification/plots-mnist/plots/`. These figures (for example `pca8_kernel_modes.png`, `pca8_nb_kernels.png`, and `pareto_frontier.png`) highlight how the architectural knobs change both accuracy and parameter budgets:

| | |
|----------|----------|
| ![Impacts of the hyperparameters](./plots-mnist/plots/hyperparameter_impacts.png) | ![Correlation Matrix](./plots-mnist/plots/correlation_matrix.png) |
| ![Impacts of the hyperparameters](./plots-mnist/plots/quantum_vs_classical.png) | ![Correlation Matrix](./plots-mnist/plots/pareto_frontier.png) |

- **Kernel modes.** Modes control how many photonic degrees of freedom each convolution kernel manipulates. Accuracy rises almost linearly with the number of modes (mean accuracy climbs from 0.58 for 4 modes to 0.85 for 16), and the correlation with accuracy is the strongest among all tunables (r≈0.65 in `full_analysis_with_efficiency.csv`). The trade-off is model size: increasing from 4 to 16 modes multiplies the average parameter count from ~90 to ~1,000, so use higher modes only when your hardware budget allows it.
- **Number of kernels.** Adding kernels deepens the expressive power of the QCNN. The sweep (`pca8_nb_kernels.png`) shows that three kernels deliver ~0.81 mean accuracy versus 0.68 with a single kernel, but also push the average parameter count to roughly 700 parameters (`full_analysis_with_efficiency.csv`). Nearly all entries in `outputs/top_20_configurations.csv` therefore use three kernels when chasing the best accuracy.
- **Kernel size.** Larger spatial footprints do not automatically help: kernel sizes of 2–3 consistently outperform size-1 kernels by ~1% absolute accuracy, whereas size 4 starts to over-smooth and drops the mean accuracy back to ~0.75 despite the added complexity (`pca8_kernel_size.png`). Choose size 2 or 3 unless you have a specific reason to capture longer-range correlations.
- **Stride.** Stride 2 slightly outperforms stride 1 on MNIST (0.776 vs 0.747 mean accuracy) while also keeping the parameter count marginally lower (≈453 vs 478 on average, see `full_analysis_with_efficiency.csv`). The larger stride reduces circuit depth by skipping overlapping receptive fields, which appears to regularize training without sacrificing accuracy.
- **Parameter count vs. payoff.** The Pareto plot (`pareto_frontier.png`) shows that squeezing the most accuracy requires hundreds of parameters: the best MNIST run (0.98 ± 0.015 accuracy with 842 parameters, see `outputs/top_20_configurations.csv`) uses 3 kernels, kernel size 2, stride 2, and 12 modes. If parameter efficiency is the goal, the classical baseline still provides the highest accuracy-per-parameter ratio (0.65 accuracy with only 13 parameters), but its absolute accuracy plateaus near 0.64 (`summary_statistics.csv`).

Use these trends to pick a QCNN variant that matches the hardware envelope: start with 12 kernel modes, 3 kernels, kernel size 2–3, and stride 2 for state-of-the-art accuracy, then dial the modes or number of kernels down if you need to fit stricter parameter counts.

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
