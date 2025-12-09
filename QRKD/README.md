# Quantum Relational Knowledge Distillation (QRKD)

## Reference and Attribution

- Paper: Quantum Relational Knowledge Distillation (preprint)
- arXiv: https://arxiv.org/abs/2508.13054
- Authors: Huiqiong Ji, et al. (see arXiv for full list)
- Original repository: not provided in the paper
- License/attribution: cite the arXiv preprint when using this reproduction

## Overview

This folder reproduces the classical MNIST baselines for QRKD:
- Teacher CNN, KD student, RKD student, and scratch student.
- Current scope: MNIST only; CIFAR and quantum layers are not yet integrated.
- Deviations: Uses lightweight CNNs and standard KD/RKD losses.

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Main entry point (from repo root): `python implementation.py --project QRKD [--config ...]`. All flags are defined in `configs/cli.json`.

```bash
python implementation.py --project QRKD --help
```

Example overrides (see `configs/cli.json` for the authoritative list):

- `--config PATH` Load an additional JSON config (merged over `defaults.json`), e.g. `--config QRKD/configs/ten_epochs.json`.
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
# Quick sanity (defaults: 1 epoch, teacher+KD+RKD+scratch)
python implementation.py --project QRKD

# Longer 10-epoch run
python implementation.py --project QRKD --config QRKD/configs/ten_epochs.json

# Students only (requires a teacher checkpoint path)
python implementation.py --project QRKD --config QRKD/configs/student_kd_10ep.json --training.teacher_path /path/to/teacher.pt
```

The script saves a snapshot of the resolved config alongside results and logs.

### Output directory and generated files

At each run, a timestamped folder is created under the base `outdir` (default: `outdir`):

```
<outdir>/run_YYYYMMDD-HHMMSS/
├── config_snapshot.json   # Resolved configuration used for the run
└── done.txt               # Placeholder artifact (replace with real outputs)
```

Notes:
- Change the base output directory with `--outdir` or by editing `configs/defaults.json` (key `outdir`).
- Add your own artifacts in the training code (e.g. checkpoints, metrics, figures). The template ignores `outdir/` in Git by default.
- Placeholder guard: if any config value contains `<<...>>` (e.g. `"<<TEACHER_PATH>>"`), the runner aborts with an error until you replace it with a real value.

## Configuration

Place configuration files in `configs/`.

- `defaults.json` defines the default parameter values.
- `cli.json` declares the CLI arguments, their types, and the config keys they mutate.
- The shared runtime automatically loads `configs/defaults.json`, `configs/cli.json`, and calls `lib/runner.py::train_and_evaluate`, so keep that entry point in every project.
- Any config value named `dtype` is normalized into a `(label, torch.dtype)` pair at runtime via `runtime_lib.dtypes`, so you can inspect both the requested string and the resolved `torch.dtype` without duplicating conversion logic.
- Keys typically include: dataset, model, training, evaluation, logging. A longer MNIST run is in `configs/ten_epochs.json`; student-only configs require providing `training.teacher_path`.
- Precision control: add a top-level `"dtype"` entry (mirrors `"seed"`) to force the entire pipeline (datasets + models) to run in a specific torch dtype; individual models may still expose a `model.dtype` override if required.

## Results and Analysis

- Where results are stored, how to reproduce key figures/tables
- Any divergence from reported metrics and possible causes
- Post-hoc analyses (ablation, sensitivity, robustness)

## Extensions and Next Steps

- Potential model variations to explore
- Additional datasets or tasks
- Improved training strategies or evaluation metrics

## Reproducibility Notes

- Random seed control
- Precision/number format control via `dtype` (global or per-model) so experiments can target `float32`, `float64`, `bfloat16`, etc.
- Determinism settings (if applicable)
- Exact versions of libraries (consider `pip freeze > results/requirements.txt`)

## Testing

Run tests from inside the `reproduction_template/` directory:

```bash
cd reproduction_template
pytest -q
```

Notes:
- Tests are scoped to this template folder and expect the current working directory to be `reproduction_template/`.
- If `pytest` is not installed: `pip install pytest`.
