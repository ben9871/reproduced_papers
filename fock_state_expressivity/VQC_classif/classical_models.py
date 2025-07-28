import torch.nn as nn

def get_mlp_deep(input_size, activation):
    classical_layer_1 = nn.Linear(input_size, 3)
    classical_layer_2 = nn.Linear(3, 2)
    if activation == "none":
        mlp = nn.Sequential(classical_layer_1, nn.ReLU(), classical_layer_2, nn.ReLU(), nn.Linear(2, 1))
    elif activation == "sigmoid":
        mlp = nn.Sequential(classical_layer_1, nn.ReLU(), classical_layer_2, nn.ReLU(), nn.Linear(2, 1),
                                     nn.Sigmoid())
    elif activation == "softmax":
        mlp = nn.Sequential(classical_layer_1, nn.ReLU(), classical_layer_2, nn.ReLU(), nn.Linear(2, 2),
                                     nn.Softmax(dim=1))
    else:
        raise ValueError(f"Activation function unknown or not implemented: '{activation}'")

    return mlp


def get_mlp_wide(input_size, activation):
    classical_layer_1 = nn.Linear(input_size, 4)
    if activation == "none":
        mlp = nn.Sequential(classical_layer_1, nn.ReLU(), nn.Linear(4, 1))
    elif activation == "sigmoid":
        mlp = nn.Sequential(classical_layer_1, nn.ReLU(), nn.Linear(4, 1),
                                     nn.Sigmoid())
    elif activation == "softmax":
        mlp = nn.Sequential(classical_layer_1, nn.ReLU(), nn.Linear(4, 2),
                                     nn.Softmax(dim=1))
    else:
        raise ValueError(f"Activation function unknown or not implemented: '{activation}'")

    return mlp


def count_svm_parameters(model, kernel):
    if kernel == "linear":
        # w (1 per feature) + b (bias)
        num_params = model.coef_.size + model.intercept_.size
    elif kernel == "rbf":
        # Each support vector contributes:
        # - one alpha (dual coef)
        # - d feature values (support vector itself)
        # Plus one bias term
        num_support_vectors = model.support_vectors_.shape[0]
        num_features = model.support_vectors_.shape[1]
        num_params = num_support_vectors * (1 + num_features) + model.intercept_.size
    else:
        raise ValueError(f"kernel type unknown: '{kernel}'")
    return num_params