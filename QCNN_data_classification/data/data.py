"""Data preparation utilities for the QCNN reproduction."""

from __future__ import annotations

import torch
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

_DATASET_MAP = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
}


def make_pca(k: int, dataset: str = "mnist"):
    dataset_key = dataset.replace("-", "").replace("_", "").lower()
    try:
        print(f"With dataset: {dataset_key}")
        dataset_cls = _DATASET_MAP[dataset_key]
    except KeyError as exc:
        valid = ", ".join(sorted(_DATASET_MAP))
        raise ValueError(
            f"Unknown dataset '{dataset}'. Valid options: {valid}."
        ) from exc

    to_t = transforms.Compose([transforms.ToTensor()])
    base_tr = dataset_cls("./data", train=True, download=True, transform=to_t)
    base_te = dataset_cls("./data", train=False, download=True, transform=to_t)

    def filt(base):
        Xs, Ys = [], []
        for img, lab in base:
            if int(lab) in (0, 1):
                Xs.append(img.view(-1).float())
                Ys.append(0 if int(lab) == 0 else 1)
        return torch.stack(Xs, 0), torch.tensor(Ys, dtype=torch.long)

    Xtr, ytr = filt(base_tr)
    Xte, yte = filt(base_te)
    pca = PCA(n_components=k, svd_solver="full", whiten=False, random_state=0)
    Ztr_raw = torch.from_numpy(pca.fit_transform(Xtr.numpy())).float()
    Zte_raw = torch.from_numpy(pca.transform(Xte.numpy())).float()
    mins = Ztr_raw.min(0, keepdim=True).values
    maxs = Ztr_raw.max(0, keepdim=True).values
    Ztr = torch.clamp((Ztr_raw - mins) / (maxs - mins + 1e-8), 0.0, 1.0)
    Zte = torch.clamp((Zte_raw - mins) / (maxs - mins + 1e-8), 0.0, 1.0)
    return (Ztr, ytr), (Zte, yte)
