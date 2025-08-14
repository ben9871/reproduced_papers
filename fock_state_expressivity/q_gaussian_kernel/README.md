# Algorithm 2: Quantum Gaussian kernel sampler

This experiment implements quantum photonic circuits to learn and approximate Gaussian kernels for machine learning applications, comparing different circuit architectures and photon configurations.

## Overview

The implementation explores quantum kernel methods by:
- **Multiple circuit architectures**: MZI, general interferometers, spiral, phase shifter-based, and beam splitter-based circuits
- **Photon number studies**: Testing different photon numbers (2, 4, 6, 8, 10) to analyze expressivity
- **Kernel learning**: Training quantum circuits to approximate target Gaussian kernel functions
- **Performance evaluation**: Comparing learned kernels with classical Gaussian kernels using SVM

## Key Results

The experiments demonstrate that quantum circuits can learn to approximate Gaussian kernels, with performance varying based on:
- **Circuit architecture**: Different interferometer designs show varying approximation capabilities
- **Photon number**: Circuits with more photons have a tendency for better fits on Gaussians with smaller standard deviation
- **Training dynamics**: Quantum kernel learning presents unique optimization challenges compared to classical methods
- **Approximation accuracy**: Our Gaussian kernel fits are less accurate than those presented in the reference paper

## Files Structure

- `model.py` - Core quantum circuit architectures and quantum layer implementations
- `data.py` - Dataset generation and preprocessing utilities
- `training.py` - Training loops and kernel learning algorithms
- `hyperparameters.py` - Configuration management for experiments
- `run_gaussian_sampler.py` - Main execution script for kernel learning
- `use_q_gauss_kernel.py` - Utilities for applying learned quantum kernels
- `q_gaussian_kernel.ipynb` - Interactive notebook with detailed analysis

## Usage
You should edit the following python files according to your usage:
- run_gaussian_sampler.py
- use_q_gauss_kernel.py
### Quick Start

Run the main kernel learning experiment:
```bash
python run_gaussian_sampler.py
```
**Afterward**, you can use the Gaussian kernel samplers that were saved in ./models on 3 classification tasks by using:
```bash
python use_q_gauss_kernel.py
```
### Interactive Analysis

Open the Jupyter notebook for detailed exploration:
```bash
jupyter notebook q_gaussian_kernel.ipynb
```

## Results

The experiment generates several outputs:
- **Kernel approximation plots**: Visual comparison of learned vs target kernels
- **SVM performance**: Classification accuracy using quantum vs classical kernels
- **Circuit diagrams**: Visual representation of different quantum architectures
- **Training dynamics**: Loss curves and convergence analysis

Key findings include:
- Quantum circuits can successfully approximate Gaussian kernel functions
- Circuits with more photons have a tendency for better fits on Gaussians with smaller standard deviation
- Our implementation could not fit the Gaussians as accurately as seen in the reference paper, requiring further investigation

## Dependencies

- Python 3.8+
- PyTorch
- Perceval (quantum photonic simulation)
- MerLin (quantum machine learning framework)
- scikit-learn (classical ML baselines and kernel methods)
- matplotlib (visualization)
- wandb (experiment tracking)

## Hyperparameters

### Training Parameters
- **`num_runs`** (5): Number of experimental repetitions for statistical reliability
- **`num_epochs`** (200): Number of training iterations
- **`batch_size`** (32): Number of samples processed together in each training step
- **`lr`** (0.02, range: 0.002-0.2): Learning rate - controls optimization step size

### Optimizer Settings
- **`betas`** ([0.7, 0.9], options: [0.7,0.9], [0.9,0.999], [0.95,0.9999]): Adam optimizer momentum parameters
- **`weight_decay`** (0.0, options: 0.0, 0.02): L2 regularization strength
- **`optimizer`** ("adam", "adagrad"): Optimization algorithm choice
- **`shuffle_train`** (True): Whether to randomize training data order

### Quantum Circuit Configuration
- **`num_photons`** ([2,4,6,8,10]): Different photon numbers to test circuit expressivity
- **`train_circuit`** (True): Whether to optimize quantum circuit parameters during training
- **`scale_type`** ("learned"): Parameter scaling method
  - "learned": Parameters optimized during training
  - "1", "pi", "2pi", "/pi", "/2pi", "0.1": Fixed scaling factors
- **`circuit`** ("general"): Quantum circuit architecture
  - "mzi": Mach-Zehnder interferometer
  - "general": General linear optical circuit
  - "spiral": Spiral circuit architecture
  - "general_all_angles": General circuit with all angle parameters
  - "ps_based": Phase shifter-based circuit
  - "bs_based": Beam splitter-based circuit
- **`no_bunching`** (False): Whether to prevent multiple photons in the same mode

## Configuration

Key parameters can be adjusted in `run_gaussian_sampler.py`. The hyperparameter search function `run_q_gaussian_sampler_hp_search()` performs grid search across optimizer configurations.

