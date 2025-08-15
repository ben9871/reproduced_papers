import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


def train_model(model, train_loader, x_train, x_test, y_train, y_test):
    """Train a single model and return training history"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss()

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
    for epoch in trange(20, desc="Training epochs"):
        for batch_idx, (images, labels) in enumerate(train_loader):
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
        'loss_history': train_loss_history,
        'test_loss_history': test_loss_history,
        'train_acc_history': train_acc_history,
        'test_acc_history': test_acc_history,
        'final_train_acc': train_acc,
        'final_test_acc': test_acc
    }


def train_model_return_preds(model, train_loader, x_train, x_test, y_train, y_test):
    """Train a single model and return training history"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss()

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
    for epoch in trange(20, desc="Training epochs"):
        for batch_idx, (images, labels) in enumerate(train_loader):
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
        'loss_history': train_loss_history,
        'test_loss_history': test_loss_history,
        'train_acc_history': train_acc_history,
        'test_acc_history': test_acc_history,
        'final_train_acc': train_acc,
        'final_test_acc': test_acc
    }, pred_test, y_test