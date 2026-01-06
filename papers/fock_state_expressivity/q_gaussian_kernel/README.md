# Quantum Gaussian Kernel

Reproduction of Algorithm 2 (“Linear quantum photonic circuits as Gaussian kernel samplers”) from the parent paper (see the root README for citation details). Implementation recreated from the manuscript.

## Overview
- Trains photonic quantum circuits to approximate Gaussian kernels of varying standard deviations by minimizing MSE between learned kernels and analytical targets. This reproduces Figure 5 from the reference paper.
- Exports the best-performing kernel models (for photon counts `{2,4,6,8,10}`) and reuses them to build precomputed kernels for SVM classification on circular, moon, and blob datasets.
- Compares quantum kernel SVMs against classical RBF-SVM baselines.
- CLI task system (`implementation.py --task {sampler,classify}`) orchestrates both phases with JSON-configured hyperparameters.

The experiments demonstrate that quantum circuits can learn to approximate Gaussian kernels, with performance varying based on:
- **Circuit architecture**: Different interferometer designs show varying approximation capabilities
- **Photon number**: Circuits with more photons have a tendency for better fits on Gaussians with smaller standard deviation
- **Training dynamics**: Quantum kernel learning presents unique optimization challenges compared to classical methods
- **Approximation accuracy**: Our Gaussian kernel fits are less accurate than those presented in the reference paper

## How to Run

### CLI workflow
1. **Train Gaussian kernel samplers** (stores checkpoints + learned function plots):
   ```bash
   python implementation.py --task sampler
   ```
2. **Evaluate saved kernels on classification datasets** (uses manifest from step 1 by default):
   ```bash
   python implementation.py --task classify
   ```

Common flags:
- `--config PATH` select another JSON config (default `configs/defaults.json`).
- `--seed INT`, `--outdir DIR`, `--device STR` override reproducibility and device (default outdir `results/`).
- `--num-runs INT` override the sampler’s `training.num_runs` value (number of restarts per `(σ, n)` pair); defaults to 5 in the base config.
- `--manifest PATH` manually point to a checkpoint manifest for classification runs.

Each execution creates `results/run_YYYYMMDD-HHMMSS/` (or `<outdir>/...` if overridden) containing:
```
summary.txt               # Human-readable metrics
metrics.json              # Raw loss/accuracy traces
config_snapshot.json      # Resolved config (after CLI overrides)
figures/                  # Learned kernels, dataset previews, accuracy charts
models/                   # Kernel checkpoints (sampler task)
models_manifest.json      # Metadata for checkpoints (sampler task)
```

## Configuration
Configs follow the template structure (`configs/`):
- `experiment.task`: `"sampler"` or `"classify"` (can be overridden via CLI).
- `model`: circuit architecture (`general`, `mzi`, etc.), scaling layer type, trainable interferometer flag, bunching toggle.
- `training`: optimizer settings, number of restarts, epochs, batch size, shuffle toggle.
- `sampler`: photon counts, Gaussian grid definition (span/step/σ list), optional `export_dir` for checkpoint artifacts.
- `classification`: dataset generation parameters (noise, scaling) and manifest path reused for evaluation.
- `outputs`: toggle learned-function plots, dataset scatter plots, and accuracy bar charts.

Modify/duplicate `configs/defaults.json` (sampler) or `configs/classify.json` (classification) to explore other circuits, photon sets, or datasets.

## Results and Analysis
- **Sampler task**: `summary.txt` reports mean/min/max training MSE for each `(σ, n)` pair; `figures/learned_vs_target.png` overlays learned kernels with analytical Gaussians, showing better fits as photon number increases.
- **Classification task**: `figures/svm_accuracy.png` compares quantum precomputed kernels against RBF-SVMs. For small σ, high-photon circuits close the gap with classical baselines but optimization becomes harder, mirroring the paper’s finding.

The experiment generates several outputs:
- **Kernel approximation plots**: Visual comparison of learned vs target kernels
- **SVM performance**: Classification accuracy using quantum vs classical kernels
- **Circuit diagrams**: Visual representation of different quantum architectures
- **Training dynamics**: Loss curves and convergence analysis

The paper's results are the follow:

![q_gauss_kernel_to_reproduce](./results/Gaussian_kernel-kernel_methods.png)

Ours:

![q_gauss_kernel_reproduced](./results/learned_gauss_kernels_best_hps.png)

Key findings include:
- Quantum circuits can successfully approximate Gaussian kernel functions
- Circuits with more photons have a tendency for better fits on Gaussians with smaller standard deviation
- Our implementation could not fit the Gaussians as accurately as seen in the reference paper, requiring further investigation

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

## Reproducibility Notes
- Seeds propagate to `random`, `numpy`, and `torch`; dataset caches under `data/cache/` ensure deterministic reuse unless `force_regenerate` is set.
- Checkpoints are saved both inside each run directory and (optionally) under `models/gaussian_kernels/` for cross-run reuse.
- GPU execution depends on Perceval/MerLin CUDA builds; defaults run on CPU.

## Testing
```bash
pytest -q
```
Current tests cover Gaussian grid and dataset preparation. Extend with regression tests (e.g., MSE/accuracy thresholds) as needed.

## Additional Material
- `notebook.ipynb` preserves the exploratory workflow prior to the scripted refactor.
- `results/` houses the static figures reproduced from the paper for documentation.
