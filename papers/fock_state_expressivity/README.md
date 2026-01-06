# Fock State-Enhanced Expressivity of Quantum Machine Learning Models

## Reference and Attribution
- Paper: *Fock state-enhanced expressivity of quantum machine learning models* (CLEO 2021)
- Authors: Beng Yee Gan, Daniel Leykam, Dimitris G. Angelakis
- DOI: https://doi.org/10.1364/CLEO_SI.2021.JW1A.73
- Original repository: not released; experiment reconstructed from the manuscript
- License & attribution: research reproduction for educational purposes. Cite both the paper and this repo when reusing the code.

This paper bundles four distinct reproductions; each lives in its own subfolder with separate configs, code, tests, and figures. Subfolders keep their own notebooks for interactive exploration.

Each subsection below is a self-contained template-compliant project dedicated to one experiment from the paper.

| Folder | Experiment | Description | Entry Point |
| --- | --- | --- | --- |
| `VQC_fourier_series` | Theory validation | Photonic VQC fits Fourier-series targets with increasing photon numbers to verify expressivity claims. | `python implementation.py --project fock_state_expressivity/VQC_fourier_series` |
| `VQC_classif` | Algorithm 1 | Linear photonic classifiers vs classical baselines on linear/circular/moon datasets. | `python implementation.py --project fock_state_expressivity/VQC_classif` |
| `q_gaussian_kernel` | Algorithm 2 | Photonic circuits learn Gaussian kernels and feed them to SVMs. Supports sampler/classify tasks. | `python implementation.py --project fock_state_expressivity/q_gaussian_kernel --task {sampler,classify}` |
| `q_random_kitchen_sinks` | Algorithm 3 | Quantum-enhanced random kitchen sinks compared to classical RKS across `R`/`γ` grids. | `python implementation.py --project fock_state_expressivity/q_random_kitchen_sinks` |

## Run the projects

Install dependencies once (from this folder):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch each experiment via the repo-level runner from the repository root (preferred):

| Experiment | Command from repo root |
| --- | --- |
| VQC Fourier Series | `python implementation.py --paper fock_state_expressivity/VQC_fourier_series --config configs/defaults.json` |
| VQC Classification | `python implementation.py --paper fock_state_expressivity/VQC_classif --config configs/defaults.json` |
| Quantum Gaussian Kernel | `python implementation.py --paper fock_state_expressivity/q_gaussian_kernel --config configs/defaults.json --task {sampler,classify}` |
| Quantum Random Kitchen Sinks | `python implementation.py --paper fock_state_expressivity/q_random_kitchen_sinks --config configs/defaults.json` |

Alternatively, `cd` into a subfolder and run `python ../../implementation.py ...` with the same flags. Each run writes to `results/run_YYYYMMDD-HHMMSS/` (or a custom `--outdir`).

## Notebooks

Interactive notebooks live with each project:
- [VQC_fourier_series/notebook.ipynb](papers/fock_state_expressivity/VQC_fourier_series/notebook.ipynb)
- [VQC_classif/notebook.ipynb](papers/fock_state_expressivity/VQC_classif/notebook.ipynb)
- [q_gaussian_kernel/notebook.ipynb](papers/fock_state_expressivity/q_gaussian_kernel/notebook.ipynb)
- [q_random_kitchen_sinks/notebook.ipynb](papers/fock_state_expressivity/q_random_kitchen_sinks/notebook.ipynb)

## Reference
- Paper: *Fock state-enhanced expressivity of quantum machine learning models* (CLEO 2021)
- Authors: Beng Yee Gan, Daniel Leykam, Dimitris G. Angelakis
- DOI: [10.1364/CLEO_SI.2021.JW1A.73](https://doi.org/10.1364/CLEO_SI.2021.JW1A.73)

Please cite the CLEO paper and the relevant subfolder when reusing these reproductions.
