# here, we want to reproduce the results of the "Quantum Self-Supervised Learning", Jaderberg et al. (2022)
# they implement a qSSL method:
### the backbone is classical with representations of width W = 8 (ResNet18 + compressor layer to 8)
### the representation network is either classical or quantum
### the loss is the Contrastive Loss (NT-Xent loss)
### dataset: preliminary -> first 2 classes of CIFAR10 [10,000 32x32 images: aeroplane and automobile]
### dataset: training -> first 5 classes of CIFAR10 [25,000 32x32 images: aeroplane, automobile, bird, cat, deer]
import argparse
import datetime
import json
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from merlin import OutputMappingStrategy, QuantumLayer
from qSSL_utils import (
    InfoNCELoss,
    create_quantum_circuit,
    load_finetuning_data,
    load_transformed_data,
)
from tqdm import tqdm

# parser
parser = argparse.ArgumentParser(description="PyTorch Quantum self-sup training")
# dataset
parser.add_argument(
    "-d", "--datadir", metavar="DIR", default="./data", help="path to dataset"
)
parser.add_argument("-cl", "--classes", type=int, default=2, help="Number of classes")
# training
parser.add_argument(
    "-e", "--epochs", type=int, default=10, help="Number of epochs for training"
)
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="Batch size")
# the SSL model
parser.add_argument(
    "-bn",
    "--batch_norm",
    action="store_true",
    default=False,
    help="Set if we use BatchNorm after compression of the encoder",
)
# Contrastive Loss
parser.add_argument(
    "-ld", "--loss_dim", type=int, default=128, help="Dimension of the loss space"
)
parser.add_argument(
    "-tau",
    "--temperature",
    type=float,
    default=0.07,
    help="Temperature of the InfoNCELoss",
)
# quantum SSL
parser.add_argument(
    "-w",
    "--width",
    type=int,
    default=8,
    help="Dimension of the features encoded in the QNN",
)
parser.add_argument(
    "-quant",
    "--quantum",
    action="store_true",
    default=False,
    help="Set if we use Quantum SSL",
)
parser.add_argument("-m", "--modes", type=int, default=10, help="Number of modes")
parser.add_argument(
    "-bunch",
    "--no_bunching",
    action="store_true",
    default=False,
    help="No bunching mode",
)


#####################
### the SSL model ###
#####################


class QSSL(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        print(f"\n Defining the SSL model with \n - quantum : {args.quantum} \n - ")
        # backbone
        self.width = args.width
        # backbone with FC = Identity
        self.backbone = torchvision.models.resnet18(pretrained=False)
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        # compressing layer to map to self.width dimension
        self.comp = nn.Linear(backbone_features, self.width)

        # building the representation network
        self.quantum = args.quantum
        self.batch_norm = args.batch_norm
        self.bn = nn.BatchNorm2d(self.width)

        # photonic circuit
        if self.quantum:
            print("\n -> Building the quantum representation network ")
            self.modes = args.modes
            self.no_bunching = args.no_bunching
            self.circuit = create_quantum_circuit(
                modes=self.modes, feature_size=self.width
            )

            input_state = [(i + 1) % 2 for i in range(args.modes)]
            print(f"input state: {input_state} and no bunching: {self.no_bunching}")

            self.representation_network = QuantumLayer(
                input_size=self.width,
                output_size=None,  # math.comb(args.modes+photon_count-1,photon_count), # but we do not use it
                circuit=self.circuit,
                trainable_parameters=[
                    p.name
                    for p in self.circuit.get_parameters()
                    if not p.name.startswith("feature")
                ],
                input_parameters=["feature"],
                input_state=input_state,
                no_bunching=self.no_bunching,
                output_mapping_strategy=OutputMappingStrategy.NONE,
            )

            self.rep_net_output_size = self.representation_network.output_size

        else:
            print("\n -> Building the classical representation network ")
            # we want to create a classical representation network with similar # of parameters to the QLayer with 10 modes, 5 photons
            # compute the number of parameters in a quantum network given 10 modes and 5 photons
            circuit = create_quantum_circuit(modes=10, feature_size=8)
            nb_trainable_params = len(
                [
                    p.name
                    for p in circuit.get_parameters()
                    if not p.name.startswith("feature")
                ]
            )
            output_size = (
                math.comb(10, 5) if args.no_bunching else math.comb(10 + 5 - 1, 5)
            )
            total_parameters = (
                nb_trainable_params + output_size * self.width
            )  # circuit + first layer of the proj
            # number of parameters in classical repnet
            classical_params = (
                self.width * self.width + self.width
            ) * 3  # rep + first layer of the proj
            # difference
            diff = total_parameters - classical_params
            print(
                f"--> Difference would be: {diff} (for {total_parameters} parameters for QNN)"
            )

            layers = []
            for _i in range(2):
                layers.append(nn.Linear(self.width, self.width, bias=True))
                layers.append(nn.LeakyReLU())
            # add another layer + activation to increase MLP size (TODO: have a more regular MLP)
            catching_output_size = int(diff / self.width - 1)
            layers.append(nn.Linear(self.width, catching_output_size, bias=True))
            layers.append(nn.LeakyReLU())

            self.representation_network = nn.Sequential(*layers)
            print(
                f" Now in repnet = {sum(p.numel() for p in self.representation_network.parameters())}"
            )
            self.rep_net_output_size = catching_output_size

        # self.fc = nn.Linear(self.rep_net_output_size, args.classes)

        self.loss_dim = args.loss_dim
        # projector to the loss space
        self.proj = nn.Sequential(
            nn.Linear(self.rep_net_output_size, self.width),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.loss_dim),
        )

        self.normalize = nn.Sigmoid()
        self.temperature = args.temperature
        # contrastive loss
        self.criterion = InfoNCELoss(temperature=self.temperature)

    def forward(self, y1, y2):
        # Encoder and MLP layer for compression
        x1 = self.comp(self.backbone(y1))
        x2 = self.comp(self.backbone(y2))
        # print(f"\n After encoder x1 = {x1}")
        # BatchNorm if needed
        if self.batch_norm:
            x1 = self.bn(x1)
            x2 = self.bn(x2)
            # print(f"\n After Batch Norm x1 = {x1}")
        # Sigmoid before Representation Network
        x1 = torch.sigmoid(x1)
        x2 = torch.sigmoid(x2)
        # print(f"\n After sigmoid Norm x1 = {x1}")
        # in the original code they use x = x * np.pi
        z1 = self.representation_network(x1)
        z2 = self.representation_network(x2)
        # print(f"\n After representation network z1 = {z1}")
        # projection to loss space
        z1 = self.proj(z1)
        z2 = self.proj(z2)
        # print(f"\n After projection z1 = {z1}")

        # L2 normalize features before contrastive loss
        z1 = nn.functional.normalize(z1, p=2, dim=1)
        z2 = nn.functional.normalize(z2, p=2, dim=1)

        # Contrastive loss on the concatenated features (along batch dimension)
        z = torch.cat((z1, z2), dim=0)
        loss = self.criterion(z)
        # print(f"\n --- Loss = {loss} --- \n")
        return loss


##########################
### training functions ###
##########################


def training_step(model, train_loader, optimizer):
    pbar = tqdm(train_loader)
    total_loss = 0.0
    for (x1, x2), _target in pbar:
        loss = model(x1, x2)

        # Check for NaN/inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected: {loss}")
            continue

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


def train(model, train_loader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    training_losses = []
    for epoch in range(args.epochs):
        loss = training_step(model, train_loader, optimizer)
        print(f"epoch: {epoch+1}/{args.epochs}, training loss: {loss}")
        training_losses.append(loss)
    return model, training_losses


def fine_tune(model, train_loader, val_loader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(args.epochs):
        # training
        model.train()
        pbar = tqdm(train_loader)
        train_acc = 0
        train_loss_total = 0
        for img, target in pbar:
            output = model(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == target).sum().item()
            train_acc += accuracy
            train_loss_total += loss.item()
            pbar.set_postfix(
                {
                    "Training Loss": f"{loss.item():.4f} - Training Accuracy: {accuracy:.4f}"
                }
            )

        # validation
        model.eval()
        pbar = tqdm(val_loader)
        val_acc = 0
        val_loss_total = 0
        with torch.no_grad():
            for img, target in pbar:
                output = model(img)
                loss = criterion(output, target)
                _, predicted = torch.max(output.data, 1)
                accuracy = (predicted == target).sum().item()
                val_acc += accuracy
                val_loss_total += loss.item()
                pbar.set_postfix(
                    {
                        "Validation Loss": f"{loss.item():.4f} - Validation Accuracy: {accuracy:.4f}"
                    }
                )

        avg_train_acc = train_acc / len(train_loader.dataset)
        avg_val_acc = val_acc / len(val_loader.dataset)
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        print(
            f"Epoch {epoch+1}/{args.epochs}: Train Acc = {avg_train_acc:.4f}, Val Acc = {avg_val_acc:.4f}"
        )

    return model, train_losses, val_losses, train_accs, val_accs


def plot_training_loss(training_losses, args):
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, args.epochs + 1),
        training_losses,
        "b-",
        linewidth=2,
        label="Training Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f'SSL Training Loss ({"Quantum" if args.quantum else "Classical"} Network)'
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f'ssl_training_loss_{"quantum" if args.quantum else "classical"}.png',
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_evaluation_metrics(train_losses, val_losses, train_accs, val_accs, args):
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot losses
    epochs = range(1, args.epochs + 1)
    ax1.plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss")
    ax1.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(
        f'Fine-tuning Losses ({"Quantum" if args.quantum else "Classical"} Network)'
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot accuracies
    ax2.plot(epochs, train_accs, "b-", linewidth=2, label="Training Accuracy")
    ax2.plot(epochs, val_accs, "r-", linewidth=2, label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(
        f'Fine-tuning Accuracies ({"Quantum" if args.quantum else "Classical"} Network)'
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        f'evaluation_metrics_{"quantum" if args.quantum else "classical"}.png',
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def save_results_to_json(
    args,
    ssl_training_losses,
    ft_train_losses,
    ft_val_losses,
    ft_train_accs,
    ft_val_accs,
):
    # Determine filename based on quantum mode
    filename = f'{"quantum" if args.quantum else "classical"}_results.json'

    # Create experiment entry
    experiment = {
        "timestamp": datetime.datetime.now().isoformat(),
        "arguments": {
            "quantum": args.quantum,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "classes": args.classes,
            "modes": getattr(args, "modes", None),
            "no_bunching": getattr(args, "no_bunching", None),
            "datadir": args.datadir,
        },
        "ssl_training_losses": ssl_training_losses,
        "fine_tuning": {
            "train_losses": ft_train_losses,
            "val_losses": ft_val_losses,
            "train_accuracies": ft_train_accs,
            "val_accuracies": ft_val_accs,
            "final_val_accuracy": ft_val_accs[-1] if ft_val_accs else 0.0,
            "best_val_accuracy": max(ft_val_accs) if ft_val_accs else 0.0,
        },
    }

    # Load existing results or create new list
    try:
        with open(filename) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    # Append new experiment
    results.append(experiment)

    # Save updated results
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {filename}")
    print(
        f"Final validation accuracy: {experiment['fine_tuning']['final_val_accuracy']:.4f}"
    )
    print(
        f"Best validation accuracy: {experiment['fine_tuning']['best_val_accuracy']:.4f}"
    )


if __name__ == "__main__":
    args = parser.parse_args()
    # load data
    train_dataset = load_transformed_data(args)
    print(f"\n Loaded a train dataset of shape {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    # define model
    model = QSSL(args)
    print(
        f"Number of parameters in model = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    # train model
    model, ssl_training_losses = train(model, train_loader, args)

    # plot SSL training loss
    # plot_training_loss(ssl_training_losses, args)

    # build model for evaluation
    frozen_model = nn.Sequential(
        model.backbone,
        model.comp,
        model.representation_network,
        nn.Linear(model.rep_net_output_size, args.classes),
    )
    print(
        f"Number of parameters in model = {sum(p.numel() for p in frozen_model[2].parameters() if p.requires_grad) + sum(p.numel() for p in frozen_model[-1].parameters() if p.requires_grad)}"
    )
    frozen_model.requires_grad_(False)
    frozen_model[-1].requires_grad_(True)
    # evaluate the model
    train_dataset, eval_dataset = load_finetuning_data(args)
    print(f"\n Loaded a train dataset of shape {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=True
    )
    model, ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs = fine_tune(
        frozen_model, train_loader, val_loader, args
    )

    # plot evaluation metrics
    # plot_evaluation_metrics(ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs, args)

    # save results to JSON
    save_results_to_json(
        args,
        ssl_training_losses,
        ft_train_losses,
        ft_val_losses,
        ft_train_accs,
        ft_val_accs,
    )
