from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from lib.quantum_kernel import build_quantum_kernel


@dataclass
class TrainingResult:
    losses: list[float]
    train_mses: list[float]


def _configure_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_name = cfg.get("optimizer", "adam").lower()
    lr = float(cfg.get("learning_rate", 0.02))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    betas = cfg.get("betas", [0.9, 0.999])
    if opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
    if opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
    if opt_name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = float(cfg.get("momentum", 0.0))
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer: {opt_name}")


def train_kernel_model(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict,
    device: torch.device,
    desc: str | None = None,
) -> TrainingResult:
    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 200))
    shuffle = bool(cfg.get("shuffle_train", True))

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    optimizer = _configure_optimizer(model, cfg)
    criterion = nn.MSELoss()

    losses: list[float] = []
    mses: list[float] = []

    model.to(device)
    progress_desc = desc or "Training kernel"
    for _epoch in tqdm(range(epochs), desc=progress_desc, leave=True):
        model.train()
        running = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds.squeeze(), batch_y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg_loss = running / len(loader)
        losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            preds = model(inputs.to(device))
            mse = mean_squared_error(targets.cpu().numpy(), preds.cpu().numpy())
            mses.append(mse)
    return TrainingResult(losses=losses, train_mses=mses)


def train_sampler(
    model_cfg: dict,
    training_cfg: dict,
    photon_counts: list[int],
    grid,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    inputs = torch.tensor(grid.delta, dtype=torch.float32).unsqueeze(-1)
    results: dict[str, dict] = {}
    best_models: list[dict] = []

    num_runs = int(training_cfg.get("num_runs", 3))

    for sigma_label, sigma_value, target in zip(
        grid.sigma_labels, grid.sigma_values, grid.targets
    ):
        sigma_results = {}
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        tqdm.write(f"\n[Sampler] σ={sigma_value:.2f} ({sigma_label})")

        for photons in photon_counts:
            run_histories: list[TrainingResult] = []
            best_mse = float("inf")
            best_state = None
            best_prediction = None
            tqdm.write(f"  • Photons n={photons}")

            for run_idx in range(num_runs):
                run_desc = (
                    f"σ={sigma_value:.2f} | n={photons} | run {run_idx + 1}/{num_runs}"
                )
                tqdm.write(f"    - {run_desc} (training hybrid model)")
                model = build_quantum_kernel(photons, model_cfg)
                history = train_kernel_model(
                    model,
                    inputs,
                    target_tensor,
                    training_cfg,
                    device=device,
                    desc=run_desc,
                )
                run_histories.append(history)

                final_mse = history.train_mses[-1]
                if final_mse < best_mse:
                    best_mse = final_mse
                    model.eval()
                    with torch.no_grad():
                        preds = model(inputs.to(device)).squeeze().cpu().numpy()
                    best_state = model.cpu().state_dict()
                    best_prediction = preds

            sigma_results[photons] = {
                "runs": [history.__dict__ for history in run_histories],
                "best_mse": best_mse,
            }
            tqdm.write(f"    ↳ Best MSE for n={photons}: {best_mse:.5f}")
            best_models.append(
                {
                    "sigma_label": sigma_label,
                    "sigma_value": sigma_value,
                    "photons": photons,
                    "state_dict": best_state,
                    "prediction": best_prediction,
                }
            )

        results[sigma_label] = sigma_results

    return results, best_models


def summarize_sampler(results: dict) -> str:
    lines = ["Quantum Gaussian kernel training summary", ""]
    for sigma_label, data in results.items():
        lines.append(f"{sigma_label}:")
        for photons, entry in data.items():
            finals = [run["train_mses"][-1] for run in entry["runs"]]
            avg = sum(finals) / len(finals)
            std = (sum((m - avg) ** 2 for m in finals) / len(finals)) ** 0.5
            lines.append(
                f"  n={photons}: {avg:.5f} ± {std:.5f} (best {entry['best_mse']:.5f})"
            )
        lines.append("")
    return "\n".join(lines)


def calculate_delta(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    diff = x1[:, None, :] - x2[None, :, :]
    delta = torch.sum(diff**2, dim=2)
    return delta


def apply_kernel(
    model: nn.Module, delta: torch.Tensor, device: torch.device
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        preds = model(delta.reshape(-1, 1).to(device))
    return preds.view(delta.shape).cpu()


def evaluate_quantum_classifiers(
    checkpoints: list[dict],
    datasets: dict[str, dict[str, torch.Tensor]],
    model_cfg: dict,
    device: torch.device,
) -> list[dict]:
    entries: list[dict] = []
    for entry in checkpoints:
        model = build_quantum_kernel(entry["photons"], model_cfg)
        model.load_state_dict(entry["state_dict"])
        model.to(device)

        metrics = {"sigma_label": entry["sigma_label"], "n_photons": entry["photons"]}
        for name in ["circular", "moon", "blob"]:
            delta_train = calculate_delta(
                datasets[name]["x_train"], datasets[name]["x_train"]
            )
            delta_test = calculate_delta(
                datasets[name]["x_test"], datasets[name]["x_train"]
            )
            k_train = apply_kernel(model, delta_train, device)
            k_test = apply_kernel(model, delta_test, device)

            clf = SVC(kernel="precomputed")
            clf.fit(k_train.numpy(), datasets[name]["y_train"].numpy())
            preds = clf.predict(k_test.numpy())
            acc = accuracy_score(datasets[name]["y_test"].numpy(), preds)
            metrics[f"{name}_acc"] = acc
        entries.append(metrics)
    return entries


def evaluate_classical_rbf(
    datasets: dict[str, dict[str, torch.Tensor]], sigmas: list[float]
) -> list[dict]:
    entries: list[dict] = []
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma**2)
        metrics = {"sigma_label": f"sigma={sigma:.2f}"}
        for name in ["circular", "moon", "blob"]:
            clf = SVC(kernel="rbf", gamma=gamma)
            clf.fit(
                datasets[name]["x_train"].numpy(), datasets[name]["y_train"].numpy()
            )
            preds = clf.predict(datasets[name]["x_test"].numpy())
            acc = accuracy_score(datasets[name]["y_test"].numpy(), preds)
            metrics[f"{name}_acc"] = acc
        entries.append(metrics)
    return entries
