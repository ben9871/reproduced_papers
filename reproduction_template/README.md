# <PAPER_SHORT_NAME> — Reproduction Template

Replace the placeholders below with the relevant information for the specific paper.

## Reference and Attribution

- Paper: <Paper title> (<venue, year>)
- Authors: <Authors list>
- DOI/ArXiv: <link>
- Original repository (if any): <URL>
- License and attribution notes: <how you attribute and cite>

## Overview

Briefly describe the paper’s goal and the scope of this reproduction.

- What was reproduced (datasets, models, metrics)
- Any deviations/assumptions
- Hardware/software environment

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Main entry point: `implementation.py`. The CLI is entirely described in `configs/cli.json`, so updating/adding arguments does not require editing Python code.

```bash
python implementation.py --help
```

Example overrides (see `configs/cli.json` for the authoritative list):

- `--config PATH` Load an additional JSON config (merged over `defaults.json`).
- `--seed INT` Random seed applied via the generic runtime.
- `--outdir DIR` Base output directory (default in `defaults.json`).
- `--epochs INT` Override `training.epochs`.
- `--batch-size INT` Override `dataset.batch_size`.
- `--lr FLOAT` Override `training.lr`.

Example runs:

```bash
# From a JSON config
python implementation.py --config configs/example.json

# Override some parameters inline
python implementation.py --config configs/example.json --epochs 50 --lr 1e-3
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
- Change the base output directory with `--outdir` or in `configs/example.json` (key `outdir`).
- Add your own artifacts in the training code (e.g. checkpoints, metrics, figures). The template ignores `outdir/` in Git by default.

## Configuration

Place configuration files in `configs/`.

- `defaults.json` defines the default parameter values.
- `cli.json` declares the CLI arguments, their types, and the config keys they mutate.
- `runtime.json` points the generic runner to `cli.json`, `defaults.json`, and the callable that implements the actual training loop (`lib/runner.py::train_and_evaluate` by default).
- `example.json` shows the structure for a specific experiment.
- Keys typically include: dataset, model, training, evaluation, logging.

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
