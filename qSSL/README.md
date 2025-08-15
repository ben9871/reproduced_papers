# Quantum Self-Supervised Learning (qSSL)

## Overview

This project reproduces the results from ["Quantum Self-Supervised Learning"](https://arxiv.org/abs/2103.14653) by Jaderberg et al. (2022). The implementation compares classical and quantum self-supervised learning methods using a contrastive learning approach.

## Goal

The script implements a quantum self-supervised learning framework with the following components:
- **Backbone**: ResNet18 with compression layer (output width: 8)
- **Representation Network**: Either classical or quantum (photonic circuit)
- **Loss Function**: InfoNCE contrastive loss (NT-Xent)
- **Dataset**: CIFAR-10 subset (first 2-5 classes)

## How to Run

### Basic Usage

```bash
python3 main.py [OPTIONS]
```

### Key Arguments

#### Dataset Configuration
- `-d, --datadir`: Path to dataset directory (default: `./data`)
- `-cl, --classes`: Number of classes to use (default: 2)

#### Training Parameters
- `-e, --epochs`: Number of training epochs (default: 10)
- `-bs, --batch_size`: Batch size (default: 128, recommended: 256)

#### Model Configuration
- `-quant, --quantum`: Enable quantum SSL (default: False, uses classical)
- `-w, --width`: Feature dimension (default: 8)
- `-m, --modes`: Number of quantum modes (default: 10)
- `-bunch, --no_bunching`: Disable bunching mode (default: False)

#### Loss Parameters
- `-ld, --loss_dim`: Loss space dimension (default: 128)
- `-tau, --temperature`: InfoNCE temperature (default: 0.07)

### Examples

**Classical SSL training:**
```bash
python3 main.py --epochs 50 --batch_size 256 --classes 5
```

**Quantum SSL training with MerLin:**
```bash
python3 main.py --merlin --epochs 50 --batch_size 256 --modes 10 --classes 5
```

**Quantum SSL training with Qiskit (from [Jaderberg et al](https://github.com/bjader/QSSL/tree/main)):**
```bash
python3 main.py --qiskit --epochs 2 --batch_size 256
```

## Output

The script generates:
- Training progress logs
- JSON results file (`quantum_results.json` or `classical_results.json`)
- SSL training loss and fine-tuning metrics
- Final validation accuracy


## Training Process

1. **Self-supervised pre-training**: Uses contrastive learning on augmented image pairs
2. **Feature extraction**: Freezes learned representations
3. **Fine-tuning**: Trains linear classifier on frozen features
4. **Evaluation**: Reports validation accuracy and saves results

## Project Structure

### Core Files

- **`main.py`**: Main training script that orchestrates the complete SSL pipeline
- **`model.py`**: Model definitions including QSSL class with quantum/classical options
- **`data_utils.py`**: Data loading utilities with CIFAR-10 transformations and augmentations
- **`training_utils.py`**: Training utilities including InfoNCE loss and evaluation functions

### Quantum Backends

- **`qnn/`**: Qiskit-based quantum neural network implementation
- **MerLin**: Photonic quantum computing integration via Perceval
- **Classical**: Standard PyTorch neural networks for baseline comparison

### Results and Data

- **`results/`**: Training metrics and model checkpoints organized by backend
- **`data/`**: CIFAR-10 dataset storage
- **`*.json`**: Experiment results and configuration files