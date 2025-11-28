# MerLin Reproduced Papers

## About this repository


This repository contains implementations and resources for reproducing key quantum machine learning papers, with a focus on photonic and optical quantum computing.

It is part of the main MerLin project: [https://github.com/merlinquantum/merlin](https://github.com/merlinquantum/merlin)
and complements the online documentation available at:

[https://merlinquantum.ai/research/reproduced_papers.html](https://merlinquantum.ai/research/reproduced_papers.html)

Each paper reproduction is designed to be accessible, well-documented, and easy to extend. Contributions are welcome!

## Running existing reproductions

- Browse the up-to-date catalogue at [https://merlinquantum.ai/reproduced_papers/index.html](https://merlinquantum.ai/reproduced_papers/index.html) to pick the paper you want to execute. The `<NAME>` you pass to the CLI is simply the folder name under the repo root (e.g., `QLSTM/`, `QORC/`, `reproduction_template/`).
- `cd` into the chosen folder and install its dependencies: `pip install -r requirements.txt` (each reproduction keeps its own list).
- Launch training/eval runs through the shared CLI from the repo root:

	```bash
	python implementation.py --project <NAME> --config <NAME>/configs/<config>.json
	```

- If you prefer running from inside the project directory, reference the parent runner instead: `python ../implementation.py --project <NAME> --config configs/<config>.json`.

All logs, checkpoints, and figures land in `<NAME>/outdir/run_YYYYMMDD-HHMMSS/` unless the configs specify a different base path.

Need a quick tour of a project’s knobs? Run `python implementation.py --project <NAME> --help` to print the runtime-generated CLI for that reproduction (dataset switches, figure toggles, etc.) before launching a full experiment.

Universal CLI flags provided by the shared runner:
- `--seed INT` Reproducibility seed propagated to Python/NumPy/PyTorch backends.
- `--dtype STR` Force a global tensor dtype before model-specific overrides.
- `--device STR` Torch device string (`cpu`, `cuda:0`, `mps`, ...).
- `--log-level LEVEL` Runtime logging verbosity (`INFO` by default).
Project-specific `configs/cli.json` files only declare the extra paper knobs; the runner injects the global options automatically.

### Precision control (`dtype`)

- Every reproduction accepts an optional top-level `"dtype"` entry in its configs, just like `"seed"`. When present, the shared runner casts input tensors and initializes models in that dtype.
- Individual models can still override via `model.dtype`; if omitted, each reproduction picks a sensible default (e.g., `float64` for photonic MerLin layers).
- Use this to downgrade to `float32` for speed, experiment with `bfloat16`, or enforce `float64` reproducibility across classical/quantum variants.

## How to contribute a reproduced paper

We encourage contributions of new quantum ML paper reproductions. Please follow the guidelines below:

### Mandatory structure for a reproduction

```
NAME/                     # Non-ambiguous acronym or fullname of the reproduced paper
├── .gitignore            # specific .gitignore rules for clean repository
├── notebook.ipynb        # Interactive exploration of key concepts
├── README.md             # Paper overview and results overview
├── requirements.txt      # additional requirements for the scripts
├── configs/              # defaults + CLI/runtime descriptors consumed by the repo root runner
├── data/                 # Datasets and preprocessing if any
├── lib/                  # code used by the shared runner and notebooks - as an integrated library
├── models/               # Trained models 
├── results/              # Selected generated figures, tables, or outputs from trained models
├── tests/                # Validation tests
└── utils/                # additional commandline utilities for visualization, launch of multiple trainings, etc...
```

### Reproduction template (starter kit)

Use the ready-to-go template in `reproduction_template/` to bootstrap a new paper folder that follows the structure above.

Quick start:

```bash
# 1) Create your paper folder (replace NAME with a short, unambiguous id)
cp -R reproduction_template NAME

cd NAME

# 2) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional shared deps can go in the repo root, but each project keeps its own requirements.txt.

# 3) Run with the example config (JSON-only) via the repo-level runner
python ../implementation.py --project NAME --config configs/example.json

# 4) See outputs (default base outdir is `outdir/` inside NAME/)
ls outdir

# 5) Run tests (from inside NAME/)
pytest -q
```

You can also run from the repository root:

```bash
python implementation.py --project NAME --config NAME/configs/example.json
```

`--project` (or `--project-dir`) is mandatory so the shared runner knows which reproduction folder to load.

Then edit the placeholders in:
- `README.md` — paper reference/authors, reproduction details, CLI options, results analysis
- `configs/example.json` — dataset/model/training defaults (extend or add more configs)
- `configs/cli.json` + `configs/runtime.json` — CLI definitions plus metadata telling the shared runner which callable to import (set `runner_callable`, `defaults_path`, and optionally `seed_callable` pointing to your seeding helper)
- `lib/runner.py` and supporting modules inside `lib/` — dataset/model/training logic invoked by the shared runner
- `runtime_lib.config.load_config` / `.deep_update` handle JSON loading and overrides globally; the template already wires `lib.config` to these helpers so you must not add a custom `lib/config.py` (JSON is the only supported format).

> **Note:** Every reproduction has its own `requirements.txt`. Install the relevant file before running `implementation.py --project ...` to ensure dependencies are available.

Notes:
- Configs are JSON-only in the template.
- Each run creates a timestamped folder under the base `outdir` (default `outdir/`): `run_YYYYMMDD-HHMMSS/` with `config_snapshot.json` and your artifacts.
- Tests are intended to be run from inside the paper folder (e.g., `cd NAME && PYTHONPATH=. pytest -q`).

### Submission process

1. **Propose** the paper in our [GitHub Discussions](https://github.com/merlinquantum/merlin/discussions)
2. **Implement** using the repository tools, following the structure above
3. **Validate** results against the original paper
4. **Document** in Jupyter notebook format
5. **Submit** a pull request with the complete reproduction folder

### Contribution requirements

- High-impact quantum ML papers (>50 citations preferred)
- Photonic/optical quantum computing focus
- Implementable with current repository features
- Clear experimental validation

### Recognition

Contributors are recognized in:
- Paper reproduction documentation
- MerLin project contributors list
- Academic citations in MerLin publications

## Code Style and Quality

This repository uses [Ruff](https://docs.astral.sh/ruff/) for consistent code formatting and linting across all paper implementations.

### Usage

**Check code style:**
```bash
ruff check .
```

**Format code:**
```bash
ruff format .
```

**Install pre-commit hooks (recommended):**
```bash
pip install pre-commit
pre-commit install
```

### Configuration

- Code style rules are defined in `pyproject.toml`
- GitHub Actions automatically check all PRs and pushes
- Pre-commit hooks run ruff automatically before commits