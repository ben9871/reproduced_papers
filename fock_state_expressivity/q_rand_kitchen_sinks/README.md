# Algorithm 3: Quantum-enhanced random kitchen sinks

This experiment implements quantum-enhanced random kitchen sinks for approximating Gaussian kernels, combining classical random Fourier features with quantum photonic circuits to achieve kernel approximations.

## Overview

The implementation explores quantum kernel approximation by:
- **Classical baseline**: Standard random kitchen sinks using cosine random features
- **Quantum enhancement**: Using photonic circuits to compute quantum random features
- **Circuit architectures**: MZI and general interferometer configurations
- **Performance comparison**: Quantum vs classical random features for kernel approximation
- **Photon number analysis**: Testing different photon numbers to study quantum behavior

## Key Results

The experiments demonstrate the behavior of quantum-enhanced random kitchen sinks:
- **Parameter-dependent performance**: The quantum method tends to outperform the classical method when using low gamma values (high standard deviation for Gaussian kernels) while the opposite happens with high gamma values
- **Photon number effects**: Using more photons leads to more extreme decision boundaries with more variance
- **Architecture sensitivity**: Different quantum circuits show varying performance characteristics
- **Parameter optimization**: Proper hyperparameter tuning is crucial for optimal performance

## Files Structure

- `approx_kernel.py` - Core implementation of quantum and classical random kitchen sinks
- `data.py` - Dataset generation and preprocessing utilities
- `training.py` - Training loops and optimization algorithms
- `hyperparameters.py` - Configuration management for experiments
- `run.py` - Main execution script for experiments
- `utils.py` - Utility functions for data processing and visualization
- `q_random_kitchen_sinks.ipynb` - Interactive notebook with detailed analysis

## Usage
You should edit the following python files according to your usage:
- run_rks.py
### Quick Start

Run the main experiment:
```bash
python run_rks.py
```

### Interactive Analysis

Open the Jupyter notebook for detailed exploration:
```bash
jupyter notebook q_random_kitchen_sinks.ipynb
```

## Results

The experiment generates comprehensive analysis outputs:
- **Kernel approximation plots**: Visual comparison of quantum vs classical kernel approximations
- **Performance metrics**: Accuracy comparison across different configurations
- **Parameter sweeps**: Analysis of hyperparameter effects on performance
- **Circuit diagrams**: Visual representation of quantum architectures used

Key findings include:
- Quantum circuits can approximate Gaussian kernels with competitive or superior accuracy
- The quantum method tends to outperform the classical method with low gamma values while classical methods excel with high gamma values
- Different photon numbers and circuit architectures provide trade-offs between expressivity and optimization difficulty

## Dependencies

- Python 3.8+
- PyTorch
- Perceval (quantum photonic simulation)
- MerLin (quantum machine learning framework)
- scikit-learn (classical ML baselines and kernel methods)
- matplotlib (visualization)
- wandb (experiment tracking)

## Hyperparameters

### Data Parameters
- **`n_samples`** (200): Number of training samples
- **`noise`** (0.2): Amount of noise added to data
- **`random_state`** (42): Random seed for reproducibility
- **`scaling`** ("MinMax", "Standard"): Data normalization method
- **`test_prop`** (0.4): Fraction of data used for testing

### Training Parameters
- **`batch_size`** (30): Number of samples processed together in each training step
- **`optimizer`** ("adam", "sgd", "adagrad"): Optimization algorithm choice
- **`learning_rate`** (0.01): Learning rate - controls optimization step size
- **`betas`** ((0.99, 0.9999)): Adam optimizer momentum parameters
- **`weight_decay`** (0.0002): L2 regularization strength
- **`num_epochs`** (200): Number of training iterations

### Quantum Circuit Parameters
- **`num_photon`** (10): Number of photons in the quantum system
- **`output_mapping_strategy`** ("LINEAR"): How to map quantum outputs to classical features
  - "NONE": No mapping applied
  - "LINEAR": Linear mapping transformation
  - "GROUPING": Group-based mapping (not working in this context)
- **`no_bunching`** (False): Whether to prevent multiple photons in the same mode
- **`circuit`** ("general", "mzi"): Quantum circuit architecture
  - "mzi": Mach-Zehnder interferometer
  - "general": General linear optical circuit

### Algorithm-Specific Parameters
- **`C`** (5): SVM regularization parameter (for classification phase)
- **`r`** (1, tested: [1,10,100]): Dimensionality of random Fourier features
- **`gamma`** (1, tested: [1-10]): Kernel bandwidth parameter (σ = 1/γ)
- **`train_hybrid_model`** (True): **Controls whether to train the hybrid quantum-classical model on the function fitting task.** This is separate from SVM classification and determines if quantum circuit parameters get optimized to approximate target functions.
- **`pre_encoding_scaling`** (1.0/π): Input scaling factor applied before quantum encoding
- **`z_q_matrix_scaling`** (10): Output matrix scaling factor
- **`hybrid_model_data`** ("Generated"): **Data source for training the hybrid model on fitting tasks** (separate from SVM classification)
  - "Generated": Uses synthetically generated data for fitting
  - "Default": Uses data from the moon dataset for fitting

## Configuration

Key parameters can be adjusted in `run_rks.py`. The script includes functions to run experiments with single parameter combinations or sweep across multiple `r` and `gamma` values for comprehensive analysis.