# VQC Classification

Structured reproduction of Algorithm 1 (“Linear quantum photonic circuits as variational quantum classifiers”) from *Fock state-enhanced expressivity of quantum machine learning models* (2022) by Gan, Leykam, and Angelakis.

## Reference and Attribution
- Paper: *Fock state-enhanced expressivity of quantum machine learning models* (2022)
- Authors: Beng Yee Gan, Daniel Leykam, Dimitris G. Angelakis
- DOI/ArXiv: https://arxiv.org/abs/2107.05224
- Original repository: not released; reproduction based on the manuscript and Quandela demos
- License & attribution: cite both the paper and this repository when using the code or figures

## Overview
The implementation includes:
- **Multiple VQC architectures**: beam splitter meshes, general interferometers, basic and spiral circuits
- **Three synthetic datasets**: linearly separable, circular, and moon-shaped data
- **Performance comparison**: VQC vs classical methods (MLP, SVM)
- **Decision boundary visualization**: Visual analysis of learned decision boundaries (reproduction of Figure 4 from the reference paper)

Key result:
- The experiments validate that **increasing the number of photons increases circuit expressivity**, though higher expressivity can lead to both better and worse results depending on the dataset complexity and training conditions.

## How to Run

### CLI entry point
```bash
python implementation.py --help
```

Key options:
- `--config PATH` JSON config (default `configs/defaults.json`).
- `--model-type {vqc,vqc_100,vqc_111,mlp_wide,mlp_deep,svm_lin,svm_rbf}` switches between photonic and classical baselines, with `vqc_100`/`vqc_111` selecting the input Fock state `[1,0,0]` or `[1,1,1]`.
- `--seed INT`, `--outdir DIR`, `--device STR` override reproducibility knobs and output location (default outdir `results/`).
- `--visualize-data` force dataset scatter plots; `--skip-boundaries` skips per-dataset decision boundaries.
- `--log-wandb` enables Weights & Biases logging (requires `wandb login`).

Examples:
```bash
# Default photonic VQC reproduction
python implementation.py

# Explicit photonic states
python implementation.py --model-type vqc_100   # prepares |1,0,0>
python implementation.py --model-type vqc_111   # prepares |1,1,1>

# Classical baseline on GPU with custom outdir
python implementation.py --model-type mlp_wide --device cuda:0 --outdir results/vqc_classif
```

Each run creates `results/run_YYYYMMDD-HHMMSS/` (or `<outdir>/...` if overridden):
```
summary.txt                 # Hyperparameters + dataset-wise stats
metrics.json                # Serialized accuracy traces
config_snapshot.json        # Resolved config (after CLI overrides)
figures/
  training_metrics.png
  datasets/<dataset>_scatter.png      # optional
  decision_boundaries/<dataset>_<model>.png
  circuits/circuit_<type>.png         # first VQC run visual
```

## Configuration
Files live in `configs/`. `defaults.json` reproduces the paper (3 modes, `[1,1,1]` photons, bs_mesh circuit, 10 runs × 150 epochs, lr 0.02). Ready-made variants include:
- `vqc_100.json` / `vqc_111.json` — identical to the default except for the preset model type and initial Fock state.
- `mlp_wide.json`, `mlp_deep.json`, `svm_lin.json`, `svm_rbf.json` — baseline configs with the appropriate `experiment.model_type`.

Use `--config configs/<name>.json` to run these presets, or create additional JSON files to explore other settings.

Important blocks:
- `model`: photonic circuit hyperparameters (modes, activation, scale layer, regularization target).
- `training`: optimizer settings, number of repeated runs, Adam betas, weight decay (`alpha`), logging toggles.
- `data`: dataset generation specs (samples, noise/class separation, subsampling ratio, cache dir).
- `outputs`: enable/disable training curves and decision-boundary plots.
- `experiment.model_type`: default model family (can be overridden with `--model-type`).

## Results and Analysis
The experiment generates several visualization outputs:
- **Decision boundaries**: Comparison of VQC and classical model boundaries
- **Performance metrics**: Accuracy comparison across methods and datasets
- **Circuit diagrams**: Visual representation of quantum circuits used

The main figure that was reproduced is Figure 4.

From the paper:

![VQC_classif_to_reproduce](./results/variational_classification.png)

Ours:

![VQC_classif_reproduced](./results/combined_decision_boundaries.png)

Key findings show that VQCs with more photons have increased expressivity, which can lead to:
- Better performance on some datasets but more challenging optimization on others
- More flexible decision boundaries
- Variable performance compared to classical methods, as increased expressivity complexifies the optimization space

## Reproducibility Notes
- The seed controls `random`, `numpy`, `torch`, and dataset generation; cached `.pt` files under `data/cache/` allow repeatable reruns.
- MerLin/Perceval layers currently run on CPU in this setup; CUDA builds are required for GPU backends.
- WandB logging is optional and only used when `log_wandb` is true (either in config or via `--log-wandb`).

## Testing
```bash
pytest -q
```
Current tests sanity-check dataset preparation; extend with regression tests (e.g., target accuracy thresholds) as needed.

## Additional Material
- `notebook.ipynb` allows for a more interactive and user-friendly exploration of the presented algorithms.
- `results/` stores curated reproduction figures (kept static for documentation).
