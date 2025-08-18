# Algorithm 1: Linear quantum photonic circuits as variational quantum classifiers

This experiment demonstrates the use of linear quantum photonic circuits for binary classification tasks, comparing performance across different photon numbers and circuit architectures.

## Overview

The implementation includes:
- **Multiple VQC architectures**: beam splitter meshes, general interferometers, basic and spiral circuits
- **Three synthetic datasets**: linearly separable, circular, and moon-shaped data
- **Performance comparison**: VQC vs classical methods (MLP, SVM)
- **Decision boundary visualization**: Visual analysis of learned decision boundaries
- **Hyperparameter optimization**: Random search

## Key Results

The experiments validate that **increasing the number of photons increases circuit expressivity**, though higher expressivity can lead to both better and worse results depending on the dataset complexity and training conditions.

## Files Structure

- `VQC.py` - Core VQC implementations and circuit architectures
- `data.py` - Dataset generation and visualization utilities  
- `training.py` - Training loops, evaluation, and visualization functions
- `classical_models.py` - Classical baseline models for comparison
- `run_vqc.py` - Main execution script for experiments
- `run_vqc_hp_search.py` - Hyperparameter search script
- `VQC_classification.ipynb` - Interactive notebook with detailed analysis

## Usage
You should edit the following python files according to your usage:
- run_vqc.py
- run_vqc_hp_search.py
### Quick Start

Run the main experiment:
```bash
python run_vqc.py
```

### Hyperparameter Search

Random search for hyperparameter optimization:
```bash
python run_vqc_hp_search.py
```

### Interactive Analysis

Open the Jupyter notebook for detailed exploration:
```bash
jupyter notebook VQC_classification.ipynb
```

## Results

The experiment generates several visualization outputs:
- **Decision boundaries**: Comparison of VQC and classical model boundaries
- **Performance metrics**: Accuracy comparison across methods and datasets
- **Circuit diagrams**: Visual representation of quantum circuits used

Key findings show that VQCs with more photons have increased expressivity, which can lead to:
- Better performance on some datasets but more challenging optimization on others
- More flexible decision boundaries
- Variable performance compared to classical methods, as increased expressivity complexifies the optimization space

## Dependencies

- Python 3.8+
- PyTorch
- Perceval (quantum photonic simulation)
- MerLin (quantum machine learning framework)
- scikit-learn (classical ML baselines)
- matplotlib (visualization)
- wandb (experiment tracking)

## Hyperparameters

### Core Model Parameters
- **`m`** (3): Number of quantum modes in the system. Set to 3 to reproduce results from the reference paper.
- **`input_size`** (2): Dimension of input data features
- **`initial_state`**: Starting quantum state configuration. Use [1,0,0] (1 photon in first mode) to reproduce reference paper results.

### Training Parameters
- **`num_runs`** (10 or 3): Number of experimental repetitions for statistical reliability
- **`n_epochs`** (150, range: 75-125): Number of complete passes through the training data
- **`batch_size`** (30, range: 15-60): Number of samples processed together in each training step
- **`lr`** (0.02, range: 0.002-0.1): Learning rate - controls optimization step size

### Optimizer Settings
- **`alpha`** (0.0002-0.2): Regularization strength to prevent overfitting
- **`betas`** ((0.8, 0.999)): Adam optimizer momentum parameters for gradient smoothing
  - First value: exponential decay rate for first moment estimates
  - Second value: exponential decay rate for second moment estimates

### Quantum Circuit Configuration
- **`circuit`** ("bs_mesh", "general", "bs_basic", "spiral"): Type of quantum circuit architecture
  - "bs_mesh": Beam splitter mesh interferometer
  - "general": General linear optical circuit
  - "bs_basic": Basic beam splitter configuration
  - "spiral": Spiral circuit architecture
- **`activation`** ("none", "sigmoid", "softmax"): Activation function applied to circuit outputs
- **`scale_type`** ("/pi", "learned", "2pi", "pi", "1", "/2pi", "0.1", "0.5"): Parameter scaling method
  - "learned": Parameters learned during training
  - "/pi", "/2pi": Fixed scaling by π divisions
  - Numerical values: Fixed scaling factors
- **`regu_on`** ("linear", "all"): Which circuit parameters to apply regularization to
- **`no_bunching`** (False): Whether to prevent multiple photons in the same mode

### Logging
- **`log_wandb`** (True): Whether to log experiments to Weights & Biases

## Configuration

The main parameters can be adjusted in the execution scripts (`run_vqc.py`, `run_vqc_hp_search.py`). The hyperparameter search script performs random search across the parameter ranges listed above.