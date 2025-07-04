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
python qSSL_train.py [OPTIONS]
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
python qSSL_train.py --epochs 50 --batch_size 256 --classes 5
```

**Quantum SSL training:**
```bash
python qSSL_train.py --quantum --epochs 50 --batch_size 256 --modes 10 --classes 5
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