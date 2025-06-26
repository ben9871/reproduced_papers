# here, we reproduce the HQNN experiments of the work: https://arxiv.org/abs/2412.04991
import math

import numpy as np
import torch
from merlin import OutputMappingStrategy, QuantumLayer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from utils import (
    ScaleLayer,
    count_parameters,
    create_quantum_circuit,
    load_spiral_dataset,
    save_experiment_results,
    train_model,
)


def generate_architectures() -> list[list[int]]:
    """Generate different QLayer architectures varying modes, number of photons and bunching strategy"""
    architectures = []
    # Hidden dimensions to try
    modes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    for mode in modes:
        nb_photons_max = mode // 2
        for photons_count in range(1, nb_photons_max + 1):
            architectures.append([mode, photons_count, True])
            architectures.append([mode, photons_count, False])

    return architectures


def sort_architecture(nb_features, out_features):
    architectures = generate_architectures()
    print(architectures)
    arch_with_params = []
    for mode, nb_photons, no_bunching in architectures:
        if no_bunching:
            output_size = math.comb(mode, nb_photons)
        else:
            output_size = math.comb(mode + nb_photons - 1, nb_photons)

        circuit = create_quantum_circuit(mode, size=nb_features)
        trainable_parameters_qlayer = len(
            [p.name for p in circuit.get_parameters() if not p.name.startswith("px")]
        )
        param_count = (
            nb_features + trainable_parameters_qlayer + output_size * out_features
        )

        # param_count = count_parameters(q_model)
        arch_with_params.append((param_count, mode, nb_photons, no_bunching))

    # Sort by parameter count (smallest to largest)
    arch_with_params.sort(key=lambda x: x[0])

    return arch_with_params


def main():
    # choice of device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # dataset parameters
    nb_samples = 1875
    nb_classes = 3
    features_to_try = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    # experiment settings
    accuracy_threshold = 90
    reproduction = 5
    # training
    bs = 64
    lr = 0.05

    for nb_features in features_to_try:
        print(f"\n ---- nb_features: {nb_features} ----")
        above_thresh = False

        print("Ordering the models")
        arch_with_params = sort_architecture(nb_features, nb_classes)
        print("Models ordered")

        for _i, (param_count, modes, photons_count, no_bunching) in enumerate(
            arch_with_params
        ):
            if not above_thresh:
                all_accs = []
                for _iter in range(reproduction):
                    ### load data ###
                    x_train, x_val, y_train, y_val, input_size, output_features = (
                        load_spiral_dataset(
                            nb_features=nb_features,
                            samples=nb_samples,
                            nb_classes=nb_classes,
                        )
                    )
                    train_dataset = TensorDataset(x_train, y_train)
                    train_loader = DataLoader(
                        train_dataset, batch_size=bs, shuffle=True
                    )
                    val_dataset = TensorDataset(x_val, y_val)
                    val_loader = DataLoader(val_dataset, batch_size=bs)

                    print(
                        f"-> moving on to modes {modes}, {photons_count} ({param_count}) parameters"
                    )

                    # define input state
                    input_state = [0] * modes
                    for k in range(photons_count):
                        input_state[2 * k] = 1
                    print(f"\n - input state: {input_state}")
                    # circuit
                    circuit = create_quantum_circuit(modes, size=input_size)
                    # build QuantumLayer from circuit
                    boson_layer = QuantumLayer(
                        input_size=nb_features,
                        output_size=None,  # but we do not use it
                        circuit=circuit,
                        trainable_parameters=[
                            p.name
                            for p in circuit.get_parameters()
                            if not p.name.startswith("px")
                        ],
                        input_parameters=["px"],
                        input_state=input_state,
                        no_bunching=no_bunching,
                        output_mapping_strategy=OutputMappingStrategy.NONE,
                        device=device,
                    )
                    # build classification layer and initialize the biases to 0
                    boson_output_size = boson_layer.output_size
                    classification_layer = nn.Linear(
                        in_features=boson_output_size,
                        out_features=output_features,
                        bias=True,
                    )
                    nn.init.constant_(classification_layer.bias, 0.0)
                    # learned ScaleLayer for the input features
                    input_layer = ScaleLayer(input_size, scale_type="learned")

                    # model
                    q_model = nn.Sequential(
                        input_layer, boson_layer, classification_layer
                    ).to(device)

                    ### TRAINING ###
                    (
                        q_train_losses,
                        q_val_losses,
                        best_q_acc,
                        q_train_accs,
                        q_val_accs,
                    ) = train_model(
                        q_model,
                        train_loader,
                        val_loader,
                        num_epochs=25,
                        lr=lr,
                        device=device,
                    )
                    print(f" -  best ACC found = {best_q_acc}")
                    all_accs.append(best_q_acc)

                mean_iter = np.mean(all_accs)
                std_iter = np.std(all_accs)
                if mean_iter > accuracy_threshold:
                    print(
                        f"Convergence obtained for this dataset using {modes} modes and {photons_count} photons."
                    )
                    above_thresh = True

                    dict = {
                        "dataset": "spiral",
                        "lr": lr,
                        "bs": bs,
                        "nb_samples": nb_samples,
                        "nb_features": nb_features,
                        "nb_classes": nb_classes,
                        "modes": modes,
                        "nb_photons": photons_count,
                        "no_bunching": no_bunching,
                        "input_state": input_state,
                        "binning": "linear",
                        "embedding": "learned",
                        "init": "none",
                        "BEST q ACC": mean_iter,
                        "BEST q ACC std": std_iter,
                        "q parameters": count_parameters(q_model),
                        "q_curves": {
                            "train ACC": q_train_accs,
                            "val ACC": q_val_accs,
                            "train loss": q_train_losses,
                            "val loss": q_val_losses,
                        },
                    }

                    save_experiment_results(
                        dict, f"MinMax_qNN_bs{bs}-lr{lr}-samples{nb_samples}.json"
                    )
                else:
                    print(f"\n ----- No satisfying results found on {modes} modes")
                    # no update of the json dictionary
            else:
                break


if __name__ == "__main__":
    main()
