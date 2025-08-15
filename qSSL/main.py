import argparse

import torch
import torch.nn as nn
from data_utils import load_finetuning_data, load_transformed_data
from model import QSSL
from torchsummary import summary
from training_utils import linear_evaluation, save_results_to_json, train

parser = argparse.ArgumentParser(description="PyTorch Quantum self-sup training")
# dataset
parser.add_argument(
    "-d", "--datadir", metavar="DIR", default="./data", help="path to dataset"
)
parser.add_argument("-cl", "--classes", type=int, default=2, help="Number of classes")
# training
parser.add_argument(
    "-e", "--epochs", type=int, default=2, help="Number of epochs for training"
)
parser.add_argument(
    "-le", "--le-epochs", type=int, default=100, help="Number of epochs for training"
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
    "--merlin",
    action="store_true",
    default=False,
    help="Set if we use Quantum SSL with MerLin",
)
parser.add_argument("-m", "--modes", type=int, default=10, help="Number of modes")
parser.add_argument(
    "-bunch",
    "--no_bunching",
    action="store_true",
    default=False,
    help="No bunching mode",
)

# Qiskit SSL (from https://github.com/bjader/QSSL)
parser.add_argument(
    "--qiskit",
    action="store_true",
    default=False,
    help="Set if we use Quantum SSL with Qiskit",
)
parser.add_argument(
    "--layers",
    type=int,
    default=2,
    help="Number of layers in the test network (default: 2).",
)
parser.add_argument(
    "--q_backend",
    type=str,
    default="qasm_simulator",
    help="Type of backend simulator to run quantum circuits on (default: qasm_simulator)",
)

parser.add_argument(
    "--encoding",
    type=str,
    default="vector",
    help="Data encoding method (default: vector)",
)
parser.add_argument(
    "--q_ansatz",
    type=str,
    default="sim_circ_14_half",
    help="Variational ansatz method (default: sim_circ_14_half)",
)
parser.add_argument("--q_sweeps", type=int, default=1, help="Number of ansatz sweeeps.")
parser.add_argument(
    "--activation",
    type=str,
    default="null",
    help="Quantum layer activation function type (default: null)",
)
parser.add_argument(
    "--shots",
    type=int,
    default=100,
    help="Number of shots for quantum circuit evaluations.",
)
parser.add_argument(
    "--save-dhs",
    action="store_true",
    help="If enabled, compute the Hilbert-Schmidt distance of the quantum statevectors belonging to"
    " each class. Only works for -q and --classes 2.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load SSL training data
    train_dataset = load_transformed_data(args)
    print(f"\n Loaded a train dataset of shape {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Define model
    model = QSSL(args)
    summary(model, [(3, 32, 32), (3, 32, 32)])
    print(
        f"Number of parameters in model = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Train SSL model
    model, ssl_training_losses, results_dir = train(model, train_loader, args)

    # Build model for linear evaluation
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

    # Load linear evaluation data
    train_dataset, eval_dataset = load_finetuning_data(args)
    print(f"\n Loaded a train dataset of shape {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Perform linear evaluation
    model, ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs = (
        linear_evaluation(frozen_model, train_loader, val_loader, args, results_dir)
    )

    # Save results to JSON
    save_results_to_json(
        args,
        ssl_training_losses,
        ft_train_losses,
        ft_val_losses,
        ft_train_accs,
        ft_val_accs,
        results_dir,
    )
