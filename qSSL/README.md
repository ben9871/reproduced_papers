# Quantum Self-Supervised Learning (qSSL)

Reproduction of: “Quantum Self-Supervised Learning” (Jaderberg et al.), arXiv:2103.14653 — https://arxiv.org/abs/2103.14653

In this folder, you will find an implementation and evaluation of the core ideas from the paper. It supports three representation networks under the same SSL pipeline: a photonic (MerLin/Perceval) model, a gate-model (Qiskit) model, and a classical MLP baseline.

— Default backend in this repo: MerLin (photonic).

## What is reproduced
- Dataset and task: CIFAR-10, restricted to the first k labels (e.g., k=5).
- Training: Self-supervised pretraining with InfoNCE on two augmented views (SimCLR-style), followed by linear evaluation with a frozen encoder.
- Models (representation network):
  - MerLin photonic circuit (Perceval + Merlin quantum layer)
  - Qiskit parameterized circuit (via `qSSL/qnn`)
  - Classical MLP baseline
- Metrics: SSL losses over epochs and linear-probing accuracy curves; checkpoints and run metadata saved per experiment.

## Project structure
- `implementation.py` — main entry point (replaces the old `main.py`)
- `lib/` — core library modules used by scripts
  - `data_utils.py` — datasets, transforms (SSL and linear eval)
  - `model.py` — backbone, representation networks (MerLin/Qiskit/Classical), projection head
  - `training_utils.py` — InfoNCE, training loops, metrics and results I/O
  - `config.py` — JSON config loading and defaults
- `configs/` — example JSON configs (default uses MerLin)
  - `qssl_default.json`
- Other
  - `linear_probing.py` — evaluate frozen features with a linear head
  - `requirements.txt` — Python dependencies
  - `utils/`, `tests/` — placeholders following the template

## Install
```bash
python -m venv ssl-venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start
- Run with default MerLin settings (from JSON config):
```bash
python implementation.py --config configs/qssl_default.json
```
- CLI alternatives (override or skip configs):
```bash
# MerLin (photonic)
python implementation.py --merlin --classes 5 --modes 10 --epochs 2 --batch_size 256 --ckpt-step 1

# Qiskit (gate-model)
python implementation.py --qiskit --classes 5 --epochs 2 --batch_size 256 --ckpt-step 1

# Classical baseline
python implementation.py --classes 5 --epochs 2 --batch_size 256 --ckpt-step 1
```

## Configuration (JSON)
See `configs/qssl_default.json`. Key fields:
- `dataset`: `root`, `classes`, `batch_size`
- `model`: `backend` (`merlin` | `qiskit` | `classical`), `width`, `loss_dim`, `batch_norm`, `temperature`
- Qiskit-specific: `layers`, `encoding`, `q_ansatz`, `q_sweeps`, `activation`, `shots`, `q_backend`
- MerLin-specific: `modes`, `no_bunching`
- `training`: `epochs`, `ckpt_step`, `le_epochs`

You can combine `--config` with CLI overrides. The runner resolves the final configuration and saves it to the results directory (`args.json`).

## Training pipeline (pedagogical overview)
1) SSL pretraining
- Input: for each image, generate two strong augmentations (query/key) using `TwoCropsTransform`.
- Backbone: ResNet18 (final FC replaced by Identity).
- Compression: Linear layer to `width` (quantum-friendly size).
- Representation network (choose one): MerLin, Qiskit, or Classical MLP.
- Projection head: MLP to `loss_dim` with BN + ReLU.
- Loss: InfoNCE (temperature τ) on the two views.

2) Linear evaluation
- Freeze backbone + compression + representation.
- Train a linear classifier on top using lightly augmented train data and minimal val transforms.
- Report accuracy curves and final/best validation accuracy.

## Models explained
- MerLin (default)
  - Photonic circuit built with Perceval: two trainable interferometers around a phase-encoding layer.
  - Features are Sigmoid-normalized and scaled by 1/π to map into phase parameters.
  - Parameters: `modes` (number of photonic modes), `no_bunching` (photon statistics), `width` (input feature size to the circuit), plus trainable circuit phases.

- Qiskit (gate-model)
  - Representation network `QNet` with `n_qubits = width`.
  - Configurable `encoding`, `q_ansatz`, `layers`, `q_sweeps`, `activation`, `shots`, and `q_backend` (e.g., `qasm_simulator`).

- Classical baseline
  - Simple MLP with `args.layers` repetitions of Linear(width, width) + LeakyReLU.

## Outputs and checkpoints
Results are saved under `results/<backend>/<timestamp>/`:
- `args.json` — resolved arguments used for the run
- `training_metrics.json` — SSL and linear-eval losses/accuracies over epochs
- `experiment_summary.json` — consolidated summary with final and best val accuracy
- `model-cl-<classes>-epoch-<n>.pth` — checkpoints saved every `ckpt_step` epochs

## Linear probing only
Evaluate pretrained encoders with a frozen representation and train a linear head:
```bash
# Single checkpoint file
python linear_probing.py --pretrained ./results/merlin/<timestamp>/model-cl-5-epoch-5.pth

# Or point to a results directory (auto-discovers .pth files)
python linear_probing.py --pretrained ./results/merlin/<timestamp>/
```

## Acknowledgments
- Original paper: Quantum Self-Supervised Learning — https://arxiv.org/abs/2103.14653
- Portions of the Qiskit pipeline and general approach are inspired by the original authors’ resources where relevant.

## Troubleshooting
- For Qiskit, ensure `qiskit-aer` is installed and the selected backend (e.g., `qasm_simulator`) is available.