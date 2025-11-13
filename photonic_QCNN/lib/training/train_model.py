import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

DEFAULT_TRAINING_PARAMS: dict[str, float] = {
    "lr": 0.1,
    "weight_decay": 0.001,
    "gamma": 0.9,
    "epochs": 20,
}


def _resolve_training_params(
    overrides: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    params = copy.deepcopy(DEFAULT_TRAINING_PARAMS)
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def train_model(
    model,
    train_loader,
    x_train,
    x_test,
    y_train,
    y_test,
    training_params: Optional[dict[str, float]] = None,
):
    """Train a single model and return training history"""
    params = _resolve_training_params(training_params)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["gamma"])
    loss_fn = nn.CrossEntropyLoss()

    device = next(model.parameters()).device

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    # Initial accuracy
    with torch.no_grad():
        output_train = model(x_train)
        pred_train = torch.argmax(output_train, dim=1)
        train_acc = (pred_train == y_train).float().mean().item()

        output_test = model(x_test)
        pred_test = torch.argmax(output_test, dim=1)
        test_acc = (pred_test == y_test).float().mean().item()

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

    # Training loop
    for _epoch in trange(params["epochs"], desc="Training epochs"):
        for _batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            train_loss_history.append(loss.item())

        # Evaluate accuracy
        with torch.no_grad():
            output_train = model(x_train)
            pred_train = torch.argmax(output_train, dim=1)
            train_acc = (pred_train == y_train).float().mean().item()

            output_test = model(x_test)
            test_loss = loss_fn(output_test, y_test)
            pred_test = torch.argmax(output_test, dim=1)
            test_acc = (pred_test == y_test).float().mean().item()

            test_loss_history.append(test_loss.item())
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
        scheduler.step()
    return {
        "loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
        "train_acc_history": train_acc_history,
        "test_acc_history": test_acc_history,
        "final_train_acc": train_acc,
        "final_test_acc": test_acc,
    }


def train_model_return_preds(
    model,
    train_loader,
    x_train,
    x_test,
    y_train,
    y_test,
    training_params: Optional[dict[str, float]] = None,
):
    """Train a single model and return training history"""
    params = _resolve_training_params(training_params)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["gamma"])
    loss_fn = nn.CrossEntropyLoss()

    device = next(model.parameters()).device

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    # Initial accuracy
    with torch.no_grad():
        output_train = model(x_train)
        pred_train = torch.argmax(output_train, dim=1)
        train_acc = (pred_train == y_train).float().mean().item()

        output_test = model(x_test)
        pred_test = torch.argmax(output_test, dim=1)
        test_acc = (pred_test == y_test).float().mean().item()

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

    # Training loop
    for _epoch in trange(params["epochs"], desc="Training epochs"):
        for _batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            train_loss_history.append(loss.item())

        # Evaluate accuracy
        with torch.no_grad():
            output_train = model(x_train)
            pred_train = torch.argmax(output_train, dim=1)
            train_acc = (pred_train == y_train).float().mean().item()

            output_test = model(x_test)
            test_loss = loss_fn(output_test, y_test)
            pred_test = torch.argmax(output_test, dim=1)
            test_acc = (pred_test == y_test).float().mean().item()

            test_loss_history.append(test_loss.item())
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
        scheduler.step()
    return (
        {
            "loss_history": train_loss_history,
            "test_loss_history": test_loss_history,
            "train_acc_history": train_acc_history,
            "test_acc_history": test_acc_history,
            "final_train_acc": train_acc,
            "final_test_acc": test_acc,
        },
        pred_test.cpu(),
        y_test.cpu(),
    )
