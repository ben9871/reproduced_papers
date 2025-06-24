# reproduced_papers

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