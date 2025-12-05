# Fock State-Enhanced Expressivity of Quantum Machine Learning Models

## Reference and Attribution
- Paper: *Fock state-enhanced expressivity of quantum machine learning models* (2022)
- Authors: Beng Yee Gan, Daniel Leykam, Dimitris G. Angelakis
- DOI/ArXiv: https://arxiv.org/abs/2107.05224
- Original repository: not released; experiment reconstructed from the manuscript
- License & attribution: research reproduction for educational purposes. Cite both the paper and this repo when reusing the code.

Each subsection below is a self-contained template-compliant project dedicated to one experiment from the paper.

| Folder | Experiment | Description | Entry Point |
| --- | --- | --- | --- |
| `VQC_fourier_series` | Theory validation | Photonic VQC fits Fourier-series targets with increasing photon numbers to verify expressivity claims. | `python implementation.py` |
| `VQC_classif` | Algorithm 1 | Linear photonic classifiers vs classical baselines on linear/circular/moon datasets. | `python implementation.py` |
| `q_gaussian_kernel` | Algorithm 2 | Photonic circuits learn Gaussian kernels and feed them to SVMs. Supports sampler/classify tasks. | `python implementation.py --task {sampler,classify}` |
| `q_random_kitchen_sinks` | Algorithm 3 | Quantum-enhanced random kitchen sinks compared to classical RKS across `R`/`γ` grids. | `python implementation.py` |

## Install Dependencies (once)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Use This Hub
1. Activate the virtual environment you created above.
2. `cd` into the desired experiment folder and follow its README for configuration, CLI options, outputs, and testing.
3. All runs write artifacts to `results/run_YYYYMMDD-HHMMSS/` to keep workflows isolated and reproducible.

## Reference
- Paper: *Fock state-enhanced expressivity of quantum machine learning models* (PRX Quantum, 2022)
- Authors: Beng Yee Gan, Daniel Leykam, Dimitris G. Angelakis
- DOI: [10.1103/PRXQuantum.3.030341](https://arxiv.org/abs/2107.05224)

Please cite both the PRX Quantum article and the relevant subfolder when reusing these reproductions.
