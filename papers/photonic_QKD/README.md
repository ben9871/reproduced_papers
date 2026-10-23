# photonic_QKD: Photonic Quantum-Enhanced Knowledge Distillation (PQKD)

Status: planning complete (Phase 1, feasibility `PARTIALLY_FEASIBLE`). No PQKD method code, experiments or results exist yet.

## Reference and Attribution

- Paper: Photonic Quantum-Enhanced Knowledge Distillation (arXiv preprint, 2026)
- Authors: Kuan-Cheng Chen et al.
- DOI/ArXiv: https://arxiv.org/abs/2603.14898 (arXiv:2603.14898)
- Original repository: none published by the authors
- License and attribution notes: the paper is distributed under CC BY 4.0. Cite the arXiv record above when reusing figures, text or results.

## Overview

The paper distils a classical teacher network into a compact student whose dictionary convolutions are conditioned by a shot-limited photonic circuit. The photonic parameters are updated without gradients, and the paper studies shot-noise scaling and EMA smoothing on MNIST, Fashion-MNIST and CIFAR-10.

- Scope, deviations and environment: to be recorded in `LOG.md` during planning.
- Related reproduction: `papers/QRKD` (knowledge distillation on MNIST/CIFAR-10). Shared dataset loaders live in `papers/shared/QRKD/datasets.py` and are candidates for reuse.

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Main entry point: the repo-level `implementation.py`. The CLI is entirely described in `cli.json`, so updating/adding arguments does not require editing Python code.

```bash
# From inside papers/photonic_QKD
python ../../implementation.py --help

# From the repo root
python implementation.py --paper photonic_QKD --help
```

Example overrides (see `cli.json` for the authoritative list):

- `--config PATH` Load an additional JSON config (merged over `defaults.json`).
- `--seed INT` Random seed applied via the generic runtime.
- `--dtype STR` Force a global torch/NumPy dtype (e.g., `float32`, `float64`).
- `--device STR` Device override (`cpu`, `cuda:0`, `mps`, etc.).
- `--log-level LEVEL` Logging verbosity (`DEBUG`, `INFO`, ...).
- `--outdir DIR` Base output directory (default in `defaults.json`).
- `--epochs INT` Override `training.epochs`.
- `--batch-size INT` Override `dataset.batch_size`.
- `--lr FLOAT` Override `training.lr`.

Example runs:

```bash
# From a JSON config (inside the project)
python ../../implementation.py --config configs/example.json

# Override some parameters inline
python ../../implementation.py --config configs/example.json --epochs 50 --lr 1e-3

# Equivalent from the repo root
python implementation.py --paper photonic_QKD --config configs/example.json --epochs 50 --lr 1e-3
```

The script saves a snapshot of the resolved config alongside results and logs.

### Data location and shared code

- The shared data root defaults to `<repo>/data`; each paper is scoped to a subfolder named after the paper (e.g., `data/photonic_QKD/`). The runner resolves this via `runtime_lib.data_paths.paper_data_dir`, so you usually just pass `--paper` or set `dataset.root` to `null` and downloads land in the right place.
- Override the base root with `DATA_DIR=/abs/path` or `--data-root /abs/path` when calling `implementation.py`; the paper subfolder is appended automatically.
- No datasets are bundled. Download or place assets under the resolved paper-specific subfolder.
- Reusable helpers can live under `papers/shared/<paper>/...`; import them from your paper code to avoid duplication while keeping code on the Python path (e.g., `from papers.shared.<paper>.module import helper`).

### Output directory and generated files

At each run, a timestamped folder is created under the base `outdir` (default: `outdir`):

```
<outdir>/run_YYYYMMDD-HHMMSS/
├── config_snapshot.json   # Resolved configuration used for the run
└── done.txt               # Scaffold artifact (to be replaced by PQKD outputs)
```

Notes:
- Change the base output directory with `--outdir` or in `configs/example.json` (key `outdir`).
- Add your own artifacts in the training code (e.g. checkpoints, metrics, figures). `outdir/` is ignored by Git.

## Configuration

Place configuration files in `configs/`.

- `defaults.json` defines the default parameter values.
- `cli.json` declares the CLI arguments, their types, and the config keys they mutate.
- The shared runtime automatically loads `configs/defaults.json`, `cli.json`, and calls `lib/runner.py::train_and_evaluate`, so keep that entry point in every project.
- Any config value named `dtype` is normalized into a `(label, torch.dtype)` pair at runtime via `runtime_lib.dtypes`, so you can inspect both the requested string and the resolved `torch.dtype` without duplicating conversion logic.
- `example.json` shows the structure for a specific experiment.
- Keys typically include: dataset, model, training, evaluation, logging.
- Precision control: add a top-level `"dtype"` entry (mirrors `"seed"`) to force the entire pipeline (datasets + models) to run in a specific torch dtype; individual models may still expose a `model.dtype` override if required.

**Placeholder guard:** If any config value still contains a `<<...>>` placeholder (e.g., `"teacher_path": "<<TEACHER_PATH>>"`), the shared runner aborts early with a clear error. Replace these placeholders with real paths/values before launching a run.

## Results and Analysis

No results yet.

## Extensions and Next Steps

To be filled after the reproduction runs.

## Reproducibility Notes

Planning findings from the arXiv v1 TeX source, to be confirmed in implementation:

- **Code and splits:** no author code or split indices are published. The 45k/5k train/val splits are regenerated with a fixed seed from the canonical train sets, and the canonical test set is held out.
- **Unreported hyperparameters:** SPSA step sizes (a, c, θ_max), the shot budget S for Tables 1–2, the input photon pattern, and the feature scale γ. Each is set by a predeclared choice or a small, equally budgeted grid.
- **Parameter accounting:** the reported compression factors match R·k² + 2·C_out trainable parameters per compressed layer, not the R·k² + C_out stated in the text. The implementation follows the reported numbers.
- **Attribution baseline:** the paper's photonic-contribution control (z = 0) zeroes every compressed kernel, so it is not a meaningful "no-photonic" student. The paper also describes, but does not report, its own fair controls (dictionary-only, random-feature, fixed-θ). These are run here alongside a classical surrogate with the same parameter and SPSA budget.
- **Simulator:** the photonic sampler is simulated in MerLin (16-mode loop-based interferometer, Fock space, threshold detection) in place of the CUDA-Q ORCA simulator used in the paper.
- **Compute:** this machine is CPU-only, so first runs use reduced epochs and seeds and are labelled as partial runs.

## Testing

Run tests from inside the `papers/photonic_QKD/` directory:

```bash
cd papers/photonic_QKD
pytest -q
```

Notes:
- Tests are scoped to this paper folder and expect the current working directory to be `photonic_QKD/`.
- If `pytest` is not installed: `pip install pytest`.
