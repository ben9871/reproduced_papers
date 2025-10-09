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
