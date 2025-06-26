# here, we reproduce the HQNN experiments of the work: https://arxiv.org/abs/2412.04991
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import math
import json
import pandas as pd
from typing import List, Tuple
import numpy as np
from merlin import OutputMappingStrategy, QuantumLayer
from utils import (create_quantum_circuit, train_model, count_parameters,
                            ScaleLayer, save_experiment_results, load_spiral_dataset)


def generate_architectures() -> List[List[int]]:
    """Generate different QLayer architectures varying modes, number of photons and bunching strategy"""
    architectures = []
    # Hidden dimensions to try
    modes = [2,4,6,8,10,12,14,16,18,20,22,24,26,28]
    for mode in modes:
        nb_photons_max = mode // 2
        for photons_count in range(1, nb_photons_max + 1):
            architectures.append([mode,photons_count,True])
            architectures.append([mode, photons_count, False])

    return architectures


def sort_architecture(NB_FEATURES, OUT_FEATURES):
    architectures = generate_architectures()
    print(architectures)
    arch_with_params = []
    for mode, nb_photons, no_bunching in architectures:
        if no_bunching:
            OUTPUT_SIZE = math.comb(mode, nb_photons)
        else:
            OUTPUT_SIZE = math.comb(mode + nb_photons - 1, nb_photons)

        circuit = create_quantum_circuit(mode, size=NB_FEATURES)
        trainable_parameters_qlayer = len([p.name for p in circuit.get_parameters() if
                                  not p.name.startswith("px")])
        param_count = NB_FEATURES + trainable_parameters_qlayer + OUTPUT_SIZE*OUT_FEATURES


        #param_count = count_parameters(q_model)
        arch_with_params.append((param_count, mode, nb_photons, no_bunching))

    # Sort by parameter count (smallest to largest)
    arch_with_params.sort(key=lambda x: x[0])

    return arch_with_params


def main():
    # choice of device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # dataset parameters
    NB_SAMPLES = 1875
    NB_CLASSES = 3
    features_to_try = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    # experiment settings
    THRESH = 90
    reproduction = 5
    # training
    BS = 64
    LR = 0.05

    for NB_FEATURES in features_to_try:
        print(f"\n ---- NB_FEATURES: {NB_FEATURES} ----")
        above_thresh = False

        print(f"Ordering the models")
        arch_with_params = sort_architecture(NB_FEATURES, NB_CLASSES)
        print(f"Models ordered")

        for i, (param_count, MODES, photons_count, no_bunching) in enumerate(arch_with_params):
            if not above_thresh:
                all_accs = []
                for iter in range(reproduction):

                    ### load data ###
                    X_train, X_val, y_train, y_val, INPUT_SIZE, OUTPUT_FEATURES = load_spiral_dataset(
                        nb_features=NB_FEATURES, samples=NB_SAMPLES, nb_classes=NB_CLASSES)
                    train_dataset = TensorDataset(X_train, y_train)
                    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
                    val_dataset = TensorDataset(X_val, y_val)
                    val_loader = DataLoader(val_dataset, batch_size=BS)


                    print(f"-> moving on to MODES {MODES}, {photons_count} ({param_count}) parameters")

                    # define input state
                    input_state = [0]*MODES
                    for k in range(photons_count):
                        input_state[2*k] = 1
                    print(f"\n - input state: {input_state}")
                    # circuit
                    circuit = create_quantum_circuit(MODES, size=INPUT_SIZE)
                    # build QuantumLayer from circuit
                    boson_layer = QuantumLayer(
                        input_size=NB_FEATURES,
                        output_size=None,  # but we do not use it
                        circuit=circuit,
                        trainable_parameters=[p.name for p in circuit.get_parameters() if
                                              not p.name.startswith("px")],
                        input_parameters=["px"],
                        input_state=input_state,
                        no_bunching = no_bunching,
                        output_mapping_strategy=OutputMappingStrategy.NONE,
                        device = device,
                    )
                    # build classification layer and initialize the biases to 0
                    boson_output_size = boson_layer.output_size
                    classification_layer = nn.Linear(
                        in_features=boson_output_size,
                        out_features=OUTPUT_FEATURES,
                        bias=True)
                    nn.init.constant_(classification_layer.bias, 0.0)
                    # learned ScaleLayer for the input features
                    input_layer = ScaleLayer(INPUT_SIZE, scale_type="learned")

                    # model
                    q_model = nn.Sequential(input_layer, boson_layer, classification_layer).to(device)


                    ### TRAINING ###
                    q_train_losses, q_val_losses, best_q_acc, q_train_accs, q_val_accs = train_model(q_model, train_loader,
                                                                                                     val_loader,
                                                                                                     num_epochs=25,
                                                                                                     lr=LR,
                                                                                                     device = device)
                    print(f" -  best ACC found = {best_q_acc}")
                    all_accs.append(best_q_acc)


                mean_iter = np.mean(all_accs)
                std_iter = np.std(all_accs)
                if mean_iter > THRESH:
                    print(f"Convergence obtained for this dataset using {MODES} modes and {photons_count} photons.")
                    above_thresh = True

                    dict = {"dataset": "spiral",
                            "lr": LR,
                            "bs": BS,
                            "nb_samples": NB_SAMPLES,
                            "nb_features": NB_FEATURES,
                            "nb_classes": NB_CLASSES,
                            "modes": MODES,
                            "nb_photons": photons_count,
                            "no_bunching": no_bunching,
                            "input_state": input_state,
                            "binning": "linear",
                            "embedding": "learned",
                            "init": "none",
                            "BEST q ACC": mean_iter,
                            "BEST q ACC std": std_iter,
                            "q parameters": count_parameters(q_model),
                            "q_curves": {"train ACC": q_train_accs, "val ACC": q_val_accs, "train loss": q_train_losses, "val loss": q_val_losses},
                            }

                    save_experiment_results(dict, f"MinMax_qNN_bs{BS}-lr{LR}-samples{NB_SAMPLES}.json")
                else:
                    print(f"\n ----- No satisfying results found on {MODES} modes")
                    # no update of the json dictionary
            else:
                break

if __name__ == "__main__":
    main()