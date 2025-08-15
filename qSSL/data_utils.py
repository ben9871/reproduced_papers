import random

import numpy as np
import perceval as pcvl
import torch
import torchvision
from PIL import ImageFilter
from torch.utils.data import Subset
from torchvision import transforms

#####################
### data handling ###
#####################


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


####################
### InfoNCE Loss ###
####################


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Args:
            features: Tensor of shape (2 * batch_size, feature_dim).
                      The first half are augmented views of instances in the batch.
                      The second half are corresponding positive pairs.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        batch_size = features.shape[0] // 2
        # create pseudo labels
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (
            (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)
        )
        # sim(z_i,z_j)/tau
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # mask to remove self-comparisons
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))
        # print(f"\n Similarity matrix:\n{similarity_matrix}")
        # exp(sim(z_i,z_j)/tau)
        sim_exp = torch.exp(similarity_matrix)
        # exp(sim(z_i,z_j)/tau) - same indices
        sim_exp_sum = sim_exp.sum(dim=1, keepdim=True) - torch.exp(
            similarity_matrix.diagonal().view(-1, 1)
        )
        # - log( # exp(sim(z_i,z_j)/tau) / sum )
        log_prob = similarity_matrix - torch.log(sim_exp_sum + 1e-8)

        pos_indices = torch.arange(batch_size).to(features.device)
        pos_pairs = torch.cat([pos_indices + batch_size, pos_indices]).to(
            features.device
        )
        # Apply softmax to get probabilities
        loss = -log_prob[torch.arange(2 * batch_size), pos_pairs].mean()
        """loss = -torch.log(
            F.softmax(similarity_matrix, dim=1) * labels
        ).sum(dim=1) / labels.sum(dim=1)"""
        # print(f"Loss: {loss}")
        return loss.mean()


##########################
### Quantum SSL models ###
##########################


def create_quantum_circuit(modes=10, feature_size=10):
    # first trainable circuit
    pre_circuit = pcvl.GenericInterferometer(
        modes,
        lambda i: (
            pcvl.BS()  # theta=pcvl.P(f"bs_1_{i}")
            .add(0, pcvl.PS(pcvl.P(f"phase_train_1_{i}")))
            .add(0, pcvl.BS())  # theta=pcvl.P(f"bs_1_{i}")
            .add(0, pcvl.PS(pcvl.P(f"phase_train_2_{i}")))
        ),
    )
    # data encoding in phase shifters (sandwhich)
    var = pcvl.Circuit(modes)
    for k in range(0, feature_size):
        var.add(k % modes, pcvl.PS(pcvl.P(f"feature-{k}")))

    # second trainable circuit
    post_circuit = pcvl.GenericInterferometer(
        modes,
        lambda i: (
            pcvl.BS()  # theta=pcvl.P(f"bs_1_{i}")
            .add(0, pcvl.PS(pcvl.P(f"phase_train_3_{i}")))
            .add(0, pcvl.BS())  # theta=pcvl.P(f"bs_1_{i}")
            .add(0, pcvl.PS(pcvl.P(f"phase_train_4_{i}")))
        ),
    )

    circuit = pcvl.Circuit(modes)

    circuit.add(0, pre_circuit, merge=True)
    circuit.add(0, var, merge=True)
    circuit.add(0, post_circuit, merge=True)

    return circuit
