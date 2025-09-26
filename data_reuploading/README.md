# Data re-uploading

## Reference and Attribution

- Paper: Experimental data re-uploading with provable enhanced learning capabilities (2025)
- Authors: Martin F. X. Mauser, Solène Four, Lena Marie Predl, Riccardo Albiero, Francesco Ceccarelli, Roberto Osellame, Philipp Petersen, Borivoje Dakić, Iris Agresti, Philip Walther
- DOI/ArXiv: 10.48550/arXiv.2507.05120
- License and attribution notes: This project reproduces experiments from Mauser et al., Experimental data re-uploading with provable enhanced learning capabilities (2025). Please cite [10.48550/arXiv.2507.05120] if you use this code or results.

## Overview

The reference paper's goal is to present a well performing and resource-efficient data re-uploading scheme on a photonic quantum processor. They also provide new theoretical insights about this algorithm.

The scope of this reproduction englobes the experiments on the 4 different binary classification tasks: circles, moons, tetromino and overhead MNIST. We report the training and test classification accuracies in function of the number of layers used. In addition, we have also visualized the decision boundary of the classifier (trained on the circles dataset) based on its number of layers.

A deviation from the paper is that we have included the variable alpha into the map function from data to phases instead of directly setting the scaling to $\pi/2$:

$x \to \phi=x\cdot\texttt{alpha}$

As for the hardware environment, the model has been implemented with MerLin, which utilizes backpropagation with gradient descent in a simulation, and with Perceval, which can either simulate or directly compute the algorithm on QPU with several optimization methods available.

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Prerequisites

You first have to download the overhead MNIST dataset and store it at two following directories:
1. data_reuploading/data/overhead/training/
2. data_reuploading/data/overhead/testing/

Note that only the two classes "ship" and "car" are used in this repository.

You can either navigate to this [link](https://www.kaggle.com/datasets/datamunge/overheadmnist) and download it manually

OR 

You can use this Kaggle's API by [creating your token, downloading it](https://www.kaggle.com/settings) and moving it to `~/.kaggle/`locally.

Then, you can run the bash command:
```bash
kaggle datasets download datamunge/overheadmnist -p data/overhead
unzip data/overhead.zip -d data/
```

### Main Experiments

The implementation supports three types of experiments:

#### 1. Reproduce Figure 5 (Default)
Reproduces the main result showing accuracy vs number of layers for a specific dataset:

```bash
# Default: Figure 5 reproduction on circles dataset
python implementation.py

# With moons dataset
python implementation.py --dataset moons

# With tetromino dataset
python implementation.py --dataset tetromino

# With overhead MNIST dataset
python implementation.py --dataset overhead

# Using a config file
python implementation.py --config configs/figure_5_circles.json

# Quick test with fewer epochs and layers
python implementation.py --config configs/quick_test.json
```

#### 2. Architecture Design Benchmark
Compares 9 different circuit designs (combinations of A, B, C data/trainable blocks):

```bash
# Run architecture benchmark on circles
python implementation.py --design-benchmark --dataset circles

# Run architecture benchmark on moons
python implementation.py --design-benchmark --dataset moons

# Using config file
python implementation.py --config configs/design_benchmark_circles.json
```

#### 3. Tau/Alpha Parameter Benchmark
Explores hyperparameter space of tau (Fisher loss parameter) and alpha (phase scaling):

```bash
# Run tau/alpha benchmark on moons
python implementation.py --tau-alpha-benchmark --dataset moons

# Using config file
python implementation.py --config configs/tau_alpha_benchmark_moons.json
```

### Command-line Options

Main entry point: `implementation.py`

```bash
python implementation.py --help
```

**General Options:**
- `--config PATH` Load config from JSON (example files in `configs/`)
- `--seed INT` Random seed for reproducibility
- `--outdir DIR` Output base directory (default: `results`)
- `--device STR` Device string (cpu, cuda:0, mps)

**Experiment Selection:**
- `--design-benchmark` Run architecture design benchmark (9 designs)
- `--tau-alpha-benchmark` Run tau/alpha parameter grid benchmark
- `--dataset {circles,moons}` Dataset to use

**Training Parameters:**
- `--epochs INT` Number of training epochs (default: 10000)
- `--batch-size INT` Batch size (default: 400)
- `--lr FLOAT` Learning rate (default: 0.001)
- `--layers INT` Max number of layers for Figure 5 (default: 15)
- `--repetitions INT` Number of repetitions for Figure 5 (default: 5)

### Output Directory

Each run creates a timestamped folder in `results/`:

```
results/run_YYYYMMDD-HHMMSS/
├── config_snapshot.json          # Configuration used
├── run.log                       # Execution log
├── figure_5_circles.png          # Main figure (for Figure 5)
├── figure_5_results.json         # Numerical results
├── architecture_grid_1_circles.png  # Grid plots (for design benchmark)
└── tau_alpha_grid_2_moons.png   # Parameter grids (for tau/alpha benchmark)
```

## Configuration

Configuration files are stored in `configs/` directory:

**Available Configs:**
- `figure_5_circles.json` - Figure 5 reproduction on circles dataset
- `figure_5_moons.json` - Figure 5 reproduction on moons dataset
- `design_benchmark_circles.json` - 9-design architecture benchmark
- `tau_alpha_benchmark_moons.json` - Tau/alpha parameter grid search
- `quick_test.json` - Fast test with reduced parameters
- `example.json` - Template configuration

**Configuration Structure:**
```json
{
  "seed": 42,
  "outdir": "results",
  "device": "cpu",
  "dataset": {
    "name": "circles",
    "batch_size": 400
  },
  "experiment": {
    "type": "figure_5",
    "dataset": "circles",
    "alpha": 0.314159,  // π/10 phase scaling
    "tau": 1.0,         // Fisher loss temperature
    "max_layers": 15,   // For Figure 5
    "repetitions": 5    // Statistical repetitions
  },
  "training": {
    "epochs": 10000,
    "lr": 0.001,
    "patience": 1000
  }
}
```

**Key Parameters:**
- `alpha`: Phase scaling factor (π/10 in paper)
- `tau`: Fisher loss temperature parameter
- `max_layers`: Maximum number of reuploading layers
- `repetitions`: Number of statistical repetitions
- `batch_size`: 400 (full batch) as used in paper

## Results and Analysis

- Where results are stored, how to reproduce key figures/tables
- Any divergence from reported metrics and possible causes
- Post-hoc analyses (ablation, sensitivity, robustness)



## Extensions and Next Steps

- Potential model variations to explore
- Additional datasets or tasks
- Improved training strategies or evaluation metrics

**Additional Benchmarks Included:**
1. **Architecture Design Grid**: Tests 9 different circuit designs (AA, AB, AC, BA, BB, BC, CA, CB, CC) across multiple depths
2. **Hyperparameter Grid**: Explores tau (Fisher loss temperature) and alpha (phase scaling) parameter space

**Architecture Designs:**
- A: PS(φ₁) → BS → PS(φ₂) → BS
- B: PS(φ₁) → BS(φ₂)
- C: PS(φ₁) → PS(φ₂) → BS

First letter = data encoding block, Second letter = trainable block

**Next Steps:**
- Extend to more complex datasets (tetromino, overhead MNIST)
- Compare with classical baseline models
- Deploy on actual quantum photonic hardware
- Optimize for hardware-specific noise models

## Testing

Run tests from inside the `data_reuploading/` directory:

```bash
cd data_reuploading
pytest -q
```

Notes:
- Tests are scoped to this template folder and expect the current working directory to be `data_reuploading/`.
- If `pytest` is not installed: `pip install pytest`.
