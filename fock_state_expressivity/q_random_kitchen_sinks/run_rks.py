import os

import numpy as np
from approx_kernel import get_random_w_b
from hyperparameters import Hyperparameters
from training import (
    classical_rand_kitchen_sinks,
    get_save_data,
    q_rand_kitchen_sinks,
    train_svm,
)

os.environ["WANDB_SILENT"] = "true"
import wandb
from utils import combine_saved_figures, create_experiment_dir, save_hyperparameters


def run_single_gamma_r(args):
    # Get data
    x_train, x_test, y_train, y_test = get_save_data(args)

    # Get random features w and b for both methods
    w, b = get_random_w_b(args.r, args.random_state)
    args.set_random(w, b)

    unique_id = wandb.util.generate_id()
    wandb.init(
        project="Q-Random Kitchen Sinks - Gan paper",
        group=f"Quantum, r={args.r}, gamma={args.gamma}",
        tags=[f"r={args.r}", f"gamma={args.gamma}", "quantum"],
        name=f"Q_r{args.r}_gamma{args.gamma}_{unique_id}",  # optional: clear naming
        config={
            "n_samples": args.n_samples,
            "noise": args.noise,
            "random_state": args.random_state,
            "scaling": args.scaling,
            "test_prop": args.test_prop,
            "num_photons": args.num_photon,
            "output_mapping_strategy": args.output_mapping_strategy,
            "no_bunching": args.no_bunching,
            "circuit": args.circuit,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "betas": args.betas,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "C": args.C,
            "r": args.r,
            "gamma": args.gamma,
            "w": args.w,
            "b": args.b,
        },
    )

    q_model_opti, q_kernel_matrix_train, q_kernel_matrix_test = q_rand_kitchen_sinks(
        x_train, x_test, y_train, args
    )
    q_acc = train_svm(
        q_kernel_matrix_train,
        q_kernel_matrix_test,
        q_model_opti,
        x_train,
        x_test,
        y_train,
        y_test,
        args,
    )
    print(f"q_rand_kitchen_sinks acc: {q_acc}")
    wandb.finish()

    wandb.init(
        project="Q-Random Kitchen Sinks - Gan paper",
        group=f"Classical, r={args.r}, gamma={args.gamma}",
        tags=[f"r={args.r}", f"gamma={args.gamma}", "classical"],
        name=f"Classical_r{args.r}_gamma{args.gamma}_{unique_id}",  # optional: clear naming
        config={
            "n_samples": args.n_samples,
            "noise": args.noise,
            "random_state": args.random_state,
            "scaling": args.scaling,
            "test_prop": args.test_prop,
            "num_photons": args.num_photon,
            "output_mapping_strategy": args.output_mapping_strategy,
            "no_bunching": args.no_bunching,
            "circuit": args.circuit,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "betas": args.betas,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "C": args.C,
            "r": args.r,
            "gamma": args.gamma,
            "w": args.w,
            "b": args.b,
        },
    )
    kernel_matrix_train, kernel_matrix_test = classical_rand_kitchen_sinks(
        x_train, x_test, y_train, args
    )
    acc = train_svm(
        kernel_matrix_train,
        kernel_matrix_test,
        None,
        x_train,
        x_test,
        y_train,
        y_test,
        args,
    )
    print(f"rand_kitchen_sinks acc: {acc}")
    wandb.finish()
    save_hyperparameters(args)
    return


def run_different_gamma_r(args):
    rs = [1, 10, 100]
    gammas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for r in rs:
        args.set_r(r)
        for gamma in gammas:
            args.set_gamma(gamma)
            print(f"For r={r}, gamma={gamma}")

            # Get data
            x_train, x_test, y_train, y_test = get_save_data(args)

            w, b = get_random_w_b(args.r, args.random_state)
            args.set_random(w, b)

            unique_id = wandb.util.generate_id()
            wandb.init(
                project="Q-Random Kitchen Sinks - Gan paper",
                group=f"Quantum, r={args.r}, gamma={args.gamma}",
                tags=[f"r={args.r}", f"gamma={args.gamma}", "quantum"],
                name=f"Q_r{args.r}_gamma{args.gamma}_{unique_id}",  # optional: clear naming
                config={
                    "n_samples": args.n_samples,
                    "noise": args.noise,
                    "random_state": args.random_state,
                    "scaling": args.scaling,
                    "test_prop": args.test_prop,
                    "num_photons": args.num_photon,
                    "output_mapping_strategy": args.output_mapping_strategy,
                    "no_bunching": args.no_bunching,
                    "circuit": args.circuit,
                    "batch_size": args.batch_size,
                    "optimizer": args.optimizer,
                    "learning_rate": args.learning_rate,
                    "betas": args.betas,
                    "weight_decay": args.weight_decay,
                    "num_epochs": args.num_epochs,
                    "C": args.C,
                    "r": args.r,
                    "gamma": args.gamma,
                    "w": args.w,
                    "b": args.b,
                },
            )
            q_model_opti, q_kernel_matrix_train, q_kernel_matrix_test = (
                q_rand_kitchen_sinks(x_train, x_test, y_train, args)
            )
            q_acc = train_svm(
                q_kernel_matrix_train,
                q_kernel_matrix_test,
                q_model_opti,
                x_train,
                x_test,
                y_train,
                y_test,
                args,
            )
            print(f"q_rand_kitchen_sinks acc: {q_acc}")
            wandb.finish()

            wandb.init(
                project="Q-Random Kitchen Sinks - Gan paper",
                group=f"Classical, r={args.r}, gamma={args.gamma}",
                tags=[f"r={args.r}", f"gamma={args.gamma}", "classical"],
                name=f"Classical_r{args.r}_gamma{args.gamma}_{unique_id}",  # optional: clear naming
                config={
                    "n_samples": args.n_samples,
                    "noise": args.noise,
                    "random_state": args.random_state,
                    "scaling": args.scaling,
                    "test_prop": args.test_prop,
                    "num_photons": args.num_photon,
                    "output_mapping_strategy": args.output_mapping_strategy,
                    "no_bunching": args.no_bunching,
                    "circuit": args.circuit,
                    "batch_size": args.batch_size,
                    "optimizer": args.optimizer,
                    "learning_rate": args.learning_rate,
                    "betas": args.betas,
                    "weight_decay": args.weight_decay,
                    "num_epochs": args.num_epochs,
                    "C": args.C,
                    "r": args.r,
                    "gamma": args.gamma,
                    "w": args.w,
                    "b": args.b,
                },
            )
            kernel_matrix_train, kernel_matrix_test = classical_rand_kitchen_sinks(
                x_train, x_test, y_train, args
            )
            acc = train_svm(
                kernel_matrix_train,
                kernel_matrix_test,
                None,
                x_train,
                x_test,
                y_train,
                y_test,
                args,
            )
            print(f"rand_kitchen_sinks acc: {acc}")
            wandb.finish()

    exp_dir = create_experiment_dir("./results/exps/")
    combine_saved_figures(True, path=exp_dir)
    combine_saved_figures(False, path=exp_dir)
    save_hyperparameters(args, filepath=exp_dir)

    return


# Hyperparams
n_samples = 200
noise = 0.2
random_state = 42
scaling = "MinMax"  # ["Standard", "MinMax"]
test_prop = 0.4
num_photon = 10
output_mapping_strategy = (
    "LINEAR"  # ['NONE', 'LINEAR', 'GROUPING'] GROUPING does not work in this context
)
no_bunching = False
circuit = "mzi"  # ['mzi', 'general']
batch_size = 30  # Problem if [num_samples * (1 - test_prop)] % batch_size != 0
optimizer = "adam"  # ['adam', 'sgd', 'adagrad']
learning_rate = 0.01
betas = (0.99, 0.9999)
weight_decay = 0.0002
num_epochs = 200
C = 8
r = 1  # dimensionality of the random Fourier features
gamma = 1  # [1, 2, ... , 10] determines the standard deviation of the Gaussian kernel: sigma = 1 / gamma
train_hybrid_model = False
pre_encoding_scaling = 1.0 / np.pi
z_q_matrix_scaling = "10"  # ['sqrt(R)', '1/sqrt(R)', 'sqrt(R) + 3', any constant]
hybrid_model_data = "Generated"  # ['Default', 'Generated']

args = Hyperparameters(
    n_samples=n_samples,
    noise=noise,
    random_state=random_state,
    scaling=scaling,
    test_prop=test_prop,
    num_photon=num_photon,
    output_mapping_strategy=output_mapping_strategy,
    no_bunching=no_bunching,
    circuit=circuit,
    batch_size=batch_size,
    optimizer=optimizer,
    learning_rate=learning_rate,
    betas=betas,
    weight_decay=weight_decay,
    num_epochs=num_epochs,
    C=C,
    r=r,
    gamma=gamma,
    train_hybrid_model=train_hybrid_model,
    pre_encoding_scaling=pre_encoding_scaling,
    z_q_matrix_scaling=z_q_matrix_scaling,
    hybrid_model_data=hybrid_model_data,
)

# To run the algorithm for a single value of R and gamma
"""run_single_gamma_r(args)"""
# To run the algorithm for 3 values of R and 10 values of gamma
run_different_gamma_r(args)
