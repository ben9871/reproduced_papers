# Theory validation experiment: Expressivity of a VQC on a Fourier series fitting task

This experiment validates the theoretical claim that increasing the number of photons in a variational quantum circuit (VQC) increases its expressivity, demonstrated through Fourier series fitting tasks.

## Overview

The implementation demonstrates:
- **VQC expressivity analysis**: Testing circuits with different photon numbers ([1,0,0], [1,1,0], [1,1,1])
- **Fourier series fitting**: Using degree-3 Fourier series as target functions to assess learning capability
- **Multiple training runs**: Statistical validation through repeated experiments
- **Performance comparison**: Direct comparison of expressivity across photon configurations

## Key Results

The experiments validate the paper's main theoretical claim:
- **Increased expressivity**: Circuits with more photons demonstrate higher expressivity
- **Parameter scaling**: Higher photon numbers lead to more trainable parameters (16 → 19 → 23)
- **Fitting performance**: The 3-photon VQC ([1,1,1]) achieves near-perfect fitting (MSE ≈ 0), while single photon shows limited expressivity
- **Consistent behavior**: Results are reproducible across multiple training runs

## Files Structure

- `VQC_fourier_series.ipynb` - Complete interactive analysis with model definitions, training, and visualization
- `results/` - Generated plots comparing target vs learned functions

## Usage

### Interactive Analysis

Open the Jupyter notebook for complete exploration:
```bash
jupyter notebook VQC_fourier_series.ipynb
```

The notebook provides:
- Step-by-step implementation of VQC architectures
- Training loops with progress visualization  
- Comparative analysis of different photon configurations
- Statistical summary of results across multiple runs

## Results

The experiment generates visualization outputs:
- **Target function**: Degree-3 Fourier series visualization
- **Training curves**: Loss progression for different photon configurations
- **Learned functions**: Comparison of fitted vs target functions
- **Statistical summary**: Performance metrics across multiple training runs

Key findings confirm that VQCs with more photons can fit more complex functions, validating the theoretical expressivity claims.

## Dependencies

- Python 3.8+
- PyTorch
- Perceval (quantum photonic simulation)
- MerLin (quantum machine learning framework)
- matplotlib (visualization)
- numpy, scikit-learn (numerical computation and metrics)

## Configuration

The experiment uses fixed configurations optimized for demonstration:
- **Photon configurations**: [1,0,0], [1,1,0], [1,1,1] across 3 modes
- **Training parameters**: 120 epochs, batch size 32, Adam optimizer
- **Target function**: Degree-3 Fourier series with fixed coefficients
- **Multiple runs**: 3 runs per configuration for statistical validation
