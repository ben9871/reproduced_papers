"""
Data loading and preprocessing utilities for qLLM experiments.
"""

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset


def load_data_kfold(args):
    """Load and prepare the dataset for k-fold cross validation with repeated sampling"""
    if args.verbose:
        print(f"\nLoading dataset '{args.dataset}' for k-fold cross validation...")

    dataset = load_dataset("SetFit/sst2")

    # Combine all splits (train, validation, test) for maximum data
    full_dataset = concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    )

    # Separate by label
    label_0_samples = [example for example in full_dataset if example["label"] == 0]
    label_1_samples = [example for example in full_dataset if example["label"] == 1]

    if args.verbose:
        print(
            f"Total available samples: Label 0: {len(label_0_samples)}, Label 1: {len(label_1_samples)}"
        )

    # Use 256 samples per class for train/val as per paper
    trainval_samples_per_label = 256

    # Calculate test set size (all remaining samples)
    test_samples_per_label = (
        min(len(label_0_samples), len(label_1_samples)) - trainval_samples_per_label
    )
    total_test_samples = test_samples_per_label * 2

    if args.verbose:
        print(
            f"Each fold will use {trainval_samples_per_label} samples per label for train/val ({2 * trainval_samples_per_label} total)"
        )
        print(
            f"Each fold will have {test_samples_per_label} samples per label for test set ({total_test_samples} total)"
        )

    return label_0_samples, label_1_samples, trainval_samples_per_label


def create_fold_data_splits(
    label_0_samples, label_1_samples, trainval_samples_per_label, fold_idx, args
):
    """Create a single fold's train/val/test split by sampling fresh data"""
    np.random.seed(args.seed + fold_idx)  # Different seed for each fold

    # Sample 256 samples per label for train/val
    selected_label_0 = np.random.choice(
        len(label_0_samples), trainval_samples_per_label, replace=False
    )
    selected_label_1 = np.random.choice(
        len(label_1_samples), trainval_samples_per_label, replace=False
    )

    # Create train/val data from selected samples
    trainval_data = []
    for i in selected_label_0:
        trainval_data.append(label_0_samples[i])
    for i in selected_label_1:
        trainval_data.append(label_1_samples[i])

    # Shuffle the combined train/val data
    np.random.shuffle(trainval_data)

    # Split into 85% train, 15% validation
    total_samples = len(trainval_data)  # 512 samples
    train_size = int(0.85 * total_samples)  # ~435 samples

    train_data = trainval_data[:train_size]
    val_data = trainval_data[train_size:]

    # Create train and validation datasets
    train_dataset = {
        "sentence": [ex["text"] for ex in train_data],
        "label": [ex["label"] for ex in train_data],
    }
    eval_dataset = {
        "sentence": [ex["text"] for ex in val_data],
        "label": [ex["label"] for ex in val_data],
    }

    # Create test set from ALL remaining samples (not used in train/val)
    used_indices_0 = set(selected_label_0)
    used_indices_1 = set(selected_label_1)

    test_data = []
    test_labels = []

    # Add unused label 0 samples to test
    for i in range(len(label_0_samples)):
        if i not in used_indices_0:
            test_data.append(label_0_samples[i]["text"])
            test_labels.append(0)

    # Add unused label 1 samples to test
    for i in range(len(label_1_samples)):
        if i not in used_indices_1:
            test_data.append(label_1_samples[i]["text"])
            test_labels.append(1)

    test_dataset = {
        "sentence": test_data,
        "label": test_labels,
    }

    # Extract features for contrastive learning
    texts = train_dataset["sentence"]
    features = [texts]
    labels = torch.tensor(train_dataset["label"])

    if args.verbose:
        print(
            f"Fold {fold_idx + 1}: Train={len(train_dataset['sentence'])}, "
            f"Val={len(eval_dataset['sentence'])}, Test={len(test_dataset['sentence'])}"
        )

    return train_dataset, eval_dataset, test_dataset, features, labels


def load_data(args):
    """Load and prepare the dataset according to paper: 256 samples per label, 85% train, 15% val"""
    if args.verbose:
        print(f"\nLoading dataset '{args.dataset}' with paper sampling strategy...")

    dataset = load_dataset("SetFit/sst2")

    # Combine all splits (train, validation, test) for maximum data
    full_train_dataset = concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    )

    # Paper approach: 256 samples from each label {+, -}

    # Separate by label
    label_0_samples = [
        example for example in full_train_dataset if example["label"] == 0
    ]
    label_1_samples = [
        example for example in full_train_dataset if example["label"] == 1
    ]

    # Sample 256 from each label
    np.random.seed(args.seed)
    samples_per_label = 256

    if len(label_0_samples) < samples_per_label:
        raise ValueError(
            f"Not enough samples for label 0: {len(label_0_samples)} < {samples_per_label}"
        )
    if len(label_1_samples) < samples_per_label:
        raise ValueError(
            f"Not enough samples for label 1: {len(label_1_samples)} < {samples_per_label}"
        )

    selected_label_0 = np.random.choice(
        len(label_0_samples), samples_per_label, replace=False
    )
    selected_label_1 = np.random.choice(
        len(label_1_samples), samples_per_label, replace=False
    )

    sampled_data = []
    sampled_data.extend([label_0_samples[i] for i in selected_label_0])
    sampled_data.extend([label_1_samples[i] for i in selected_label_1])

    # Shuffle the combined data
    np.random.shuffle(sampled_data)

    # Split into 85% training and 15% validation
    total_samples = len(sampled_data)  # 512 samples
    train_size = int(0.85 * total_samples)  # 435 samples

    train_data = sampled_data[:train_size]
    val_data = sampled_data[train_size:]

    # Create datasets in the expected format
    train_dataset = {
        "sentence": [ex["text"] for ex in train_data],
        "label": [ex["label"] for ex in train_data],
    }
    eval_dataset = {
        "sentence": [ex["text"] for ex in val_data],
        "label": [ex["label"] for ex in val_data],
    }

    # Use remaining original training data as test set (N - 2×256 samples)
    # Track which original indices were used for sampling
    used_label_0_indices = set()
    used_label_1_indices = set()

    # Map selected indices back to original dataset indices
    label_0_idx = 0
    label_1_idx = 0
    for i, example in enumerate(full_train_dataset):
        if example["label"] == 0:
            if label_0_idx in selected_label_0:
                used_label_0_indices.add(i)
            label_0_idx += 1
        else:  # label == 1
            if label_1_idx in selected_label_1:
                used_label_1_indices.add(i)
            label_1_idx += 1

    used_indices = used_label_0_indices | used_label_1_indices
    remaining_indices = [
        i for i in range(len(full_train_dataset)) if i not in used_indices
    ]
    test_dataset = {
        "sentence": [full_train_dataset[i]["text"] for i in remaining_indices],
        "label": [full_train_dataset[i]["label"] for i in remaining_indices],
    }

    # Extract texts and labels for contrastive learning
    texts = train_dataset["sentence"]
    features = [texts]
    labels = torch.tensor(train_dataset["label"])

    if args.verbose:
        print("Paper sampling strategy applied:")
        print(f"- Sampled {samples_per_label} samples from each label")
        print(
            f"- Training: {len(train_dataset['sentence'])} samples ({len(train_dataset['sentence']) / total_samples * 100:.1f}%)"
        )
        print(
            f"- Validation: {len(eval_dataset['sentence'])} samples ({len(eval_dataset['sentence']) / total_samples * 100:.1f}%)"
        )
        print(f"- Test: {len(test_dataset['sentence'])} samples")
        print(f"- Train label distribution: {np.bincount(train_dataset['label'])}")
        print(f"- Val label distribution: {np.bincount(eval_dataset['label'])}")

    return train_dataset, eval_dataset, test_dataset, features, labels
