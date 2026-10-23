# Photonic Quantum-Enhanced Knowledge Distillation (PQKD) - Reproduction

Status: implementation complete and tested (Phase 2). No experiment results yet; feasibility `PARTIALLY_FEASIBLE` (CPU-only budget, unreported hyperparameters).

## Reference and Attribution

- Paper: Photonic Quantum-Enhanced Knowledge Distillation (arXiv preprint, 2026)
- Authors: Kuan-Cheng Chen, Shang Yu, Chen-Yu Liu, Samuel Yen-Chi Chen, Huan-Hsin Tseng, Yen Jui Chang, Wei-Hao Huang, Felix Burt, Esperanza Cuenca Gomez, Zohim Chandani, William Clements, Ian Walmsley, Kin K. Leung
- arXiv: https://arxiv.org/abs/2603.14898 (DOI 10.48550/arXiv.2603.14898)
- Original repository: none published ("available upon reasonable request")
- License: the paper is CC BY 4.0. Cite the arXiv record when reusing figures, text or results.

## Original Paper

PQKD compresses a CNN student during knowledge distillation from a larger teacher. Selected convolutions become **dictionary convolutions**:

- each layer learns only R spatial basis filters B;
- the channel-mixing tensor M is not learned; it is generated from a global conditioning vector, vec(M) = A z(θ), with a fixed random matrix A.

The vector z(θ) ∈ R^512 comes from a programmable 16-mode, loop-based photonic interferometer with beam-splitter angles θ (dim θ ∈ {15, 30, 45}):

- S threshold-detection shots are taken;
- each 16-bit click pattern is split into two 8-bit halves;
- the two 256-bin histograms are concatenated and standardised.

Training alternates SPSA updates of θ, which are gradient-free and driven by a validation KD objective, with Adam epochs on the student. EMA smoothing of z across epochs is optional.

Datasets: MNIST, Fashion-MNIST and CIFAR-10, each with a 45k/5k train/validation split.

Main claims:
- **Table 1:** at 1.5× compression, PQKD matches the teacher on MNIST and Fashion-MNIST and lags it on CIFAR-10.
- **Table 2 / Fig. 2:** there is a smooth compression–accuracy frontier up to about 105× overall compression.
- **Fig. 4:** the photonic feature's test-accuracy gain follows δ∞ − k/√S, and EMA raises δ∞.

## Reproduction Scope (including Updates and Deviations)

Planned evidence (details in the git-ignored `LOG.md`; nothing has been run beyond smoke tests yet):

- **Table 1 (MNIST first):** teacher (64,128,128) and PQKD (conv1+conv2 compressed, R=4, dim θ=45). Fashion-MNIST and CIFAR-10 follow.
- **Fair baselines** at the same teacher, data and budget. Selected with `student.variant` / `conditioning.source`:

  | ID | Baseline | What it is |
  |---|---|---|
  | B0 | `zero` | the paper's z = 0 control (degenerate: zero kernels) |
  | B1 | `dense` | dense KD-only student at the same trainable-parameter budget |
  | B2 | `dictionary_trainable` | M learned directly |
  | B3 | `random_fixed` / `random_resampled` | Gaussian z |
  | B4 | `photonic_fixed` | θ frozen |
  | B5 | `surrogate` | classical iso-parameter generator with the same SPSA |
  | B6 | `classical_histogram` | independent-mode clicks through the same Φ and S |

- **Shot sweep with and without EMA** (Fig. 4), then the Table 2 operating points.

Updates and deviations:

- **Photonic simulation:** MerLin (`QuantumLayer`, full Fock space, threshold detection), in place of the CUDA-Q ORCA simulator used in the paper. Only MerLin 0.4.2 public API calls are used.
- **Assumed settings** (not reported in the paper): 8 photons in alternating input modes; S = 200; EMA β = 0.9 on; SPSA (a, c) = (0.05, 0.2) until a predeclared grid sets them.
- **Standardisation:** μ and σ are fitted on features at random θ. At the paper's initial θ = 0 the interferometer is the identity, so σ would be zero.
- **Parameter accounting:** each compressed layer has a bias **and a per-output-channel gain** (R·k² + 2·C_out trainable parameters). This is the only accounting that reproduces the paper's reported compression factors.
- **Compute budget:** first runs are CPU-only with reduced budgets (30 epochs, 3 seeds; paper: 100 epochs, 5 seeds) and are labelled partial. `*_original.json` configs keep the paper budget.
- **Metrics:** test accuracy is the primary metric. The paper reports validation accuracy, which SPSA also tunes on.

## Install and How to Run

```bash
pip install -r papers/photonic_QKD/requirements.txt
```

Every run trains one model. Students need a finished teacher run on the same dataset, widths and split seed.

Smoke run (seconds; checks the code path only):

```bash
python implementation.py --paper photonic_QKD --config configs/smoke_teacher.json
python implementation.py --paper photonic_QKD --config configs/smoke_student.json --teacher-path papers/photonic_QKD/outdir/<teacher run>
```

Reduced runs (defaults: 30 epochs) and paper-budget runs (100 epochs):

```bash
python implementation.py --paper photonic_QKD --seed 0                                    # teacher, MNIST
python implementation.py --paper photonic_QKD --config configs/teacher_original.json --seed 0
python implementation.py --paper photonic_QKD --config configs/pqkd_merlin.json --teacher-path <teacher run dir> --seed 0
python implementation.py --paper photonic_QKD --config configs/pqkd_original.json --teacher-path <teacher run dir> --seed 0
python implementation.py --paper photonic_QKD --config configs/baseline_surrogate_classical.json --teacher-path <teacher run dir> --seed 0
```

The same commands work from inside `papers/photonic_QKD` with `python ../../implementation.py --config configs/<name>.json ...`.

Each run writes `outdir/run_YYYYMMDD-HHMMSS/` with:
- `run.log`;
- `run_status.json`;
- `config_snapshot.json`;
- `metrics.json`;
- `teacher.pt` or `student.pt`;
- `feature_trace.npz` for conditioned students (raw and EMA z per epoch).

## Configuration

`configs/defaults.json` holds the paper hyperparameters: batch 64, Adam 1e-3, τ = 3, λ = 0.5, dropout 0.25, 10 SPSA steps per epoch, d = 512, 16 modes. It also holds the assumed values above.

| Config | Purpose |
|---|---|
| `teacher_original.json` | teacher, 100 epochs |
| `pqkd_merlin.json` / `pqkd_original.json` | PQKD student, 30 / 100 epochs |
| `baseline_*_{classical,merlin}.json` | baselines B0–B6 |
| `smoke_*.json` | tiny infrastructure runs |

Student configs contain `teacher.checkpoint: "<<RESULT_PATH>>"`. Pass `--teacher-path`; the runtime refuses to start until it is resolved.

Main flags (see `cli.json`): `--task`, `--dataset {mnist,fashion_mnist,cifar10,synthetic}`, `--teacher-path`, `--conditioning`, `--shots`, `--ema`, `--theta-dim`, `--spsa-a`, `--spsa-c`, `--epochs`, `--batch-size`, `--lr`. Global flags (`--seed`, `--device`, `--outdir`, `--data-root`, `--config`) come from the shared runtime.

## Data

The torchvision datasets live under `data/photonic_QKD/` and download on first use. The train/validation split is a seeded permutation (`dataset.split_seed = 0`) of the canonical training set; the canonical test set is held out. The paper's synthetic Gaussian-blob dataset (`--dataset synthetic`) is generated in memory.

## Results Obtained and Comparison with the Paper

No results yet. Phase 3 runs the teacher, the SPSA grid, PQKD and the baselines.

## Limitations

- **CPU-only:** reduced epochs and seeds until a GPU machine is available.
- **Unreported settings:** SPSA step sizes, shot budget, input photon pattern and the Fig. 4 configuration are not in the paper; the assumed values are listed above.
- **Simulation only:** no ORCA hardware was used.

## Tests

```bash
cd papers/photonic_QKD
python -m pytest -q
```

The 46 tests cover the paper's parameter counts and compression factors, the dictionary kernel (Eqs. 6–7), photonic circuit ordering and threshold detection, the feature map, the KD loss, SPSA, every conditioning source, and end-to-end teacher → student runs on the synthetic dataset.

## Citation and License

Cite arXiv:2603.14898. This reproduction's code follows the repository license.
