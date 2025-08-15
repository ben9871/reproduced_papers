import random
import numpy as np
import torch
import torchvision
from PIL import ImageFilter
from torch.utils.data import Subset
from torchvision import transforms


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [0.1, 2.0]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_first_n_classes(dataset, n_classes):
    """Get indices of samples belonging to first n classes"""
    targets = np.array(dataset.targets)
    indices = np.where(targets < n_classes)[0]
    return Subset(dataset, indices)


# load data for the SSL training
# apply transformation to generate pairs of augmented images


def load_transformed_data(args):
    # Normalization for CIFAR
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )

    augmentation = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ]

    full_dataset = torchvision.datasets.CIFAR10(
        root=args.datadir,
        train=True,
        download=True,
        transform=TwoCropsTransform(transforms.Compose(augmentation)),
    )
    train_dataset = get_first_n_classes(full_dataset, args.classes)
    # Print actual labels used after filtering
    unique_labels = {train_dataset[i][1] for i in range(len(train_dataset))}
    print(f"SSL training dataset - Actual labels used: {sorted(unique_labels)}")
    return train_dataset


# load training + validation data for the fine-tuning of the model
def load_finetuning_data(args):
    # Normalization for CIFAR (same as pre-training)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )

    # Simple augmentations for fine-tuning
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),  # Keep this basic augmentation
            transforms.RandomCrop(32, padding=4),  # Gentle spatial augmentation
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Minimal or no augmentation for validation
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.datadir,
        train=True,
        download=False,
        transform=train_transforms,  # Note: Single transform, not TwoCropsTransform
    )
    train_dataset = get_first_n_classes(train_dataset, args.classes)
    # Print actual labels used after filtering
    unique_labels = {train_dataset[i][1] for i in range(len(train_dataset))}
    print(f"Fine-tuning training dataset - Actual labels used: {sorted(unique_labels)}")

    val_dataset = torchvision.datasets.CIFAR10(
        root=args.datadir, train=False, download=False, transform=val_transforms
    )
    val_dataset = get_first_n_classes(val_dataset, args.classes)
    # Print actual labels used after filtering
    unique_labels = {val_dataset[i][1] for i in range(len(val_dataset))}
    print(
        f"Fine-tuning validation dataset - Actual labels used: {sorted(unique_labels)}"
    )

    return train_dataset, val_dataset
