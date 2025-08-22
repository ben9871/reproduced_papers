"""
Photonic model training utilities for qLLM experiments using MerLin.
"""

import copy
import json
import math
import os
from datetime import datetime

import merlin as ml  # Using our Merlin framework
import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from classical_utils import evaluate
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

##########################################
### Quantum classifier and its wrapper ###
##########################################


def create_quantum_circuit(m, size=400):
    """Create quantum circuit with specified number of modes and input size"""

    wl = pcvl.GenericInterferometer(
        m,
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"phase_1_{i}"))
        // pcvl.BS()
        // pcvl.PS(pcvl.P(f"phase_2_{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c = pcvl.Circuit(m)
    c.add(0, wl, merge=True)

    c_var = pcvl.Circuit(m)
    for i in range(size):
        px = pcvl.P(f"px-{i + 1}")
        c_var.add(i % m, pcvl.PS(px))
    print(c_var)
    c.add(0, c_var, merge=True)

    wr = pcvl.GenericInterferometer(
        m,
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"phase_3_{i}"))
        // pcvl.BS()
        // pcvl.PS(pcvl.P(f"phase_4_{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c.add(0, wr, merge=True)

    return c

class ScaleLayer(nn.Module):
    def __init__(self, dim, scale_type = "learned"):
        super(ScaleLayer, self).__init__()
        # Create a single learnable parameter (initialized to 1.0 by default)
        # Caution: MerLin already mutltiplies by pi
        if scale_type == "learned":
            self.scale = nn.Parameter(torch.rand(dim))
        elif scale_type == "2pi":
            self.scale = torch.full((dim,), 2 * torch.pi)
        elif scale_type == "pi":
            self.scale = torch.full((dim,), torch.pi)
        elif scale_type == "1":
            self.scale = torch.full((dim,), 1)
        #print(f"SELF.SCALE: {self.scale.shape}")

    def forward(self, x):
        # Element-wise multiplication of each input element by the learned scale
        return x * self.scale

class DivideByPi(nn.Module):
    """Normalizes phases to be within quantum-friendly range"""

    def forward(self, x):
        return x / torch.pi


class QuantumClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=100,
        modes=10,
        num_classes=2,
        input_state=None,
        device="cpu",
        no_bunching=False,
    ):
        super().__init__()
        print(
            f"Building a model with {modes} modes and {input_state} as an input state and no_bunching = {no_bunching} (input of shape {hidden_dim})"
        )
        # this layer downscale the inputs to fit in the QLayer
        hidden_dim = modes
        self.downscaling_layer = nn.Linear(input_dim, hidden_dim, device=device)
        self.scale = ScaleLayer(dim = hidden_dim, scale_type="learned")
        self.pi_scale = DivideByPi()
        # building the QLayer with MerLin
        circuit = create_quantum_circuit(modes, size=hidden_dim)
        # default input state
        if input_state is None:
            input_state = [(i + 1) % 2 for i in range(modes)]
        photons_count = sum(input_state)
        # PNR output size
        output_size_slos = (
            math.comb(modes + photons_count - 1, photons_count)
            if not no_bunching
            else math.comb(modes, photons_count)
        )

        # build the QLayer with a linear output as in the original paper
        # "The measurement output of the second module is then passed through a single Linear layer"
        self.sig = nn.Sigmoid()
        print("\n -> self.q_circuit")
        self.q_circuit = ml.QuantumLayer(
            input_size=hidden_dim,
            output_size=None,  # but we do not use it
            circuit=circuit,
            trainable_parameters=[
                p.name for p in circuit.get_parameters() if not p.name.startswith("px")
            ],
            input_parameters=["px"],
            input_state=input_state,
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
            device=device,
            no_bunching=no_bunching,
        )
        self.bn = nn.LayerNorm(output_size_slos).requires_grad_(False)  # works OK
        print(f"\n -- Building the Linear layer with output size = {num_classes} -- ")
        self.output_layer = nn.Linear(output_size_slos, num_classes, device=device)

    def forward(self, x):
        # forward pass
        #print(f"x of shape {x.shape}")
        out = self.downscaling_layer(x)
        #out = self.sig(x)
        out = self.pi_scale(out)
        # casts the input to [0,1] + use q_circuit
        out = self.q_circuit(out) #self.sig
        out_scaled = out  # self.bn(out) #(out - out.mean(dim = 1, keepdim=True)) / (out.std(dim = 1, keepdim=True)+1e-6)
        out = self.output_layer(out_scaled)
        return out

class QuantumClassifier_v2(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=100,
        modes=10,
        num_classes=2,
        input_state=None,
        device="cpu",
        no_bunching=False,
        E = 1,
    ):
        super().__init__()
        print(
            f"Building a model with {modes} modes and {input_state} as an input state and no_bunching = {no_bunching} (input of shape {hidden_dim})"
        )
        # this layer downscale the inputs to fit in the QLayer
        hidden_dim = modes
        self.downscaling_layer = nn.Linear(input_dim, hidden_dim, device=device)
        self.pi_scale = DivideByPi()
        self.E = E
        # building the QLayer with MerLin
        circuit_1 = create_quantum_circuit(modes, size=hidden_dim)
        circuit_2 = create_quantum_circuit(modes, size=hidden_dim)

        # default input state
        if input_state is None:
            input_state = [(i + 1) % 2 for i in range(modes)]
        photons_count = sum(input_state)

        self.q_circuit_1 = ml.QuantumLayer(
            input_size=hidden_dim,
            output_size=None,  # but we do not use it
            circuit=circuit_1,
            trainable_parameters=[
                p.name for p in circuit_1.get_parameters() if not p.name.startswith("px")
            ],
            input_parameters=["px"],
            input_state=input_state,
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
            device=device,
            no_bunching=True,
        )
        if self.E == 2:
            self.q_circuit_2 = ml.QuantumLayer(
                input_size=hidden_dim,
                output_size=None,  # but we do not use it
                circuit=circuit_2,
                trainable_parameters=[
                    p.name for p in circuit_2.get_parameters() if not p.name.startswith("px")
                ],
                input_parameters=["px"],
                input_state=input_state,
                output_mapping_strategy=ml.OutputMappingStrategy.NONE,
                device=device,
                no_bunching=True,
            )

        output_encoder = math.comb(modes, photons_count)

        # PNR output size
        output_size_slos = (
            math.comb(modes + photons_count - 1, photons_count)
            if not no_bunching
            else math.comb(modes, photons_count)
        )

        # build the QLayer with a linear output as in the original paper
        # "The measurement output of the second module is then passed through a single Linear layer"
        self.sig = nn.Sigmoid()
        print("\n -> self.q_circuit")

        circuit_final = create_quantum_circuit(modes, size=self.E*output_encoder)

        self.q_circuit_final = ml.QuantumLayer(
            input_size=self.E*output_encoder,
            output_size=None,  # but we do not use it
            circuit=circuit_final,
            trainable_parameters=[
                p.name for p in circuit_final.get_parameters() if not p.name.startswith("px")
            ],
            input_parameters=["px"],
            input_state=input_state,
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
            device=device,
            no_bunching=no_bunching,
        )
        self.bn = nn.LayerNorm(output_size_slos).requires_grad_(False)  # works OK
        print(f"\n -- Building the Linear layer with output size = {num_classes} -- ")
        self.output_layer = nn.Linear(output_size_slos, num_classes, device=device)

    def forward(self, x):
        # forward pass
        #print(f"x of shape {x.shape}")
        x = self.downscaling_layer(x)
        #out = self.sig(x)
        x = self.pi_scale(x)

        e1 = self.q_circuit_1(x)
        if self.E == 2:
            e2 = self.q_circuit_2(x)

            e_cat = torch.cat([e1,e2], dim=1)
        else:
            e_cat = e1
        # casts the input to [0,1] + use q_circuit
        out = self.q_circuit_final(e_cat) #self.sig
        out_scaled = out  # self.bn(out) #(out - out.mean(dim = 1, keepdim=True)) / (out.std(dim = 1, keepdim=True)+1e-6)
        out = self.output_layer(out_scaled)
        return out

def marginalize_photon_presence(keys, probs):
    """
    Marginalize Fock state probabilities to get per-mode occupation probabilities.

    Computes the probability that each mode contains at least one photon
    by summing over all Fock states where that mode is occupied.

    Args:
        keys (list): List of Fock state tuples, e.g., [(0,1,0,2), (1,0,1,0), ...]
        probs (torch.Tensor): Tensor of shape (N, num_keys) with probabilities
            for each Fock state, with requires_grad=True

    Returns:
        torch.Tensor: Shape (N, num_modes) with marginal probability that
            each mode has at least one photon
    """
    device = probs.device
    keys_tensor = torch.tensor(
        keys, dtype=torch.long, device=device
    )  # shape: (num_keys, num_modes)
    keys_tensor.shape[1]

    # Create mask of shape (num_modes, num_keys)
    # Each mask[i] is a binary vector indicating which Fock states have >=1 photon in mode i
    mask = (keys_tensor >= 1).T  # shape: (num_modes, num_keys)

    # Convert to float to allow matrix multiplication
    mask = mask.float()

    # Now do: (N, num_keys) @ (num_keys, num_modes) → (N, num_modes)
    marginalized = probs @ mask.T  # shape: (N, num_modes)
    return marginalized

class QuantumClassifier_amplitude(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=100,
        modes=10,
        num_classes=2,
        input_state=None,
        device="cpu",
        no_bunching=False,
        E = 1,
    ):
        super().__init__()
        print(
            f"Building a model with {modes} modes and {input_state} as an input state and no_bunching = {no_bunching} (input of shape {hidden_dim})"
        )
        # this layer downscale the inputs to fit in the QLayer
        hidden_dim = modes
        self.downscaling_layer = nn.Linear(input_dim, hidden_dim, device=device)
        self.pi_scale = DivideByPi()
        self.E = E
        # building the QLayer with MerLin
        circuit_1 = create_quantum_circuit(modes, size=hidden_dim)
        circuit_2 = create_quantum_circuit(modes, size=hidden_dim)

        # default input state
        if input_state is None:
            input_state = [(i + 1) % 2 for i in range(modes)]
        photons_count = sum(input_state)

        self.q_circuit_1 = ml.QuantumLayer(
            input_size=hidden_dim,
            output_size=None,  # but we do not use it
            circuit=circuit_1,
            trainable_parameters=[
                p.name for p in circuit_1.get_parameters() if not p.name.startswith("px")
            ],
            input_parameters=["px"],
            input_state=input_state,
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
            device=device,
            no_bunching=True,
        )
        if self.E == 2:
            self.q_circuit_2 = ml.QuantumLayer(
                input_size=hidden_dim,
                output_size=None,  # but we do not use it
                circuit=circuit_2,
                trainable_parameters=[
                    p.name for p in circuit_2.get_parameters() if not p.name.startswith("px")
                ],
                input_parameters=["px"],
                input_state=input_state,
                output_mapping_strategy=ml.OutputMappingStrategy.NONE,
                device=device,
                no_bunching=True,
            )

        # we generate the keys associated with the probs
        dummy_input = torch.zeros(1, hidden_dim, dtype=torch.float32)
        # Get MerLin probability distribution (no gradients to avoid sampling warnings)
        with torch.no_grad():
            merlin_params = self.q_circuit_1.prepare_parameters([dummy_input])
            unitary = self.q_circuit_1.computation_process.converter.to_tensor(
                *merlin_params
            )
            # perfect distribution (no sampling)
            self.keys, probs = (
                self.q_circuit_1.computation_process.simulation_graph.compute(
                    unitary, input_state
                )
            )

            probs_marginalized = marginalize_photon_presence(self.keys, probs)

            print(f"Probabilities marginalized with shape {probs_marginalized.shape}")
            output_encoder = probs_marginalized.shape[-1]
            print(f"Therefore the input of the encoder is of shape {output_encoder}")




        # build the QLayer with a linear output as in the original paper
        # "The measurement output of the second module is then passed through a single Linear layer"
        self.sig = nn.Sigmoid()
        print("\n -> self.q_circuit")

        circuit_final = create_quantum_circuit(modes, size=self.E*output_encoder)

        self.q_circuit_final = ml.QuantumLayer(
            input_size=self.E*output_encoder,
            output_size=None,  # but we do not use it
            circuit=circuit_final,
            trainable_parameters=[
                p.name for p in circuit_final.get_parameters() if not p.name.startswith("px")
            ],
            input_parameters=["px"],
            input_state=input_state,
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
            device=device,
            no_bunching=no_bunching,
        )

        # PNR output size
        output_size_slos = (
            math.comb(modes + photons_count - 1, photons_count)
            if not no_bunching
            else math.comb(modes, photons_count)
        )

        self.bn = nn.LayerNorm(output_size_slos).requires_grad_(False)  # works OK
        print(f"\n -- Building the Linear layer with output size = {num_classes} -- ")
        self.output_layer = nn.Linear(output_size_slos, num_classes, device=device)

    def forward(self, x):
        # forward pass
        #print(f"x of shape {x.shape}")
        x = self.downscaling_layer(x)
        #out = self.sig(x)
        x = self.pi_scale(x)

        e1 = self.q_circuit_1(x)
        e1_expectation = marginalize_photon_presence(self.keys, e1)
        e_cat = e1_expectation
        if self.E == 2:
            e2 = self.q_circuit_2(x)
            e2_expectation = marginalize_photon_presence(self.keys, e2)
            e_cat = torch.cat([e1_expectation,e2_expectation], dim=1)

        # casts the input to [0,1] + use q_circuit
        out = self.q_circuit_final(e_cat) #self.sig
        out_scaled = out  # self.bn(out) #(out - out.mean(dim = 1, keepdim=True)) / (out.std(dim = 1, keepdim=True)+1e-6)
        out = self.output_layer(out_scaled)
        return out


class QLayerTraining(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=100,
        modes=10,
        num_classes=2,
        dropout_rate=0.2,
        lr=0.001,
        weight_decay=1e-5,
        epochs=100,
        batch_size=32,
        device=None,
        input_state=None,
        no_bunching=False,
        expectation = True,
        vanilla = False,
        E = 1,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.modes = modes
        self.input_state = input_state
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize model
        self.model = None
        self.classes_ = None
        self.is_fitted_ = False
        # Training history
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        self.train_accuracies = []
        self.val_accuracies = []
        self.no_bunching = no_bunching
        self.expectation = expectation
        self.vanilla = vanilla
        self.E = E

    def _initialize_model(self):
        """Initialize or re-initialize the model."""

        if self.vanilla:
            print(f" \n -------------------- \n Vanilla Quantum Head \n --------------------")
            self.model = QuantumClassifier_amplitude(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                modes=self.modes,
                num_classes=len(self.classes_),
                input_state=self.input_state,
                device=self.device,
                no_bunching=self.no_bunching,
            )

        if self.expectation:
            print(f" \n -------------------- \n Expectation encoding for second PQC \n --------------------")
            self.model = QuantumClassifier_amplitude(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                modes=self.modes,
                num_classes=len(self.classes_),
                input_state=self.input_state,
                device=self.device,
                no_bunching=self.no_bunching,
                E=self.E,
            )
        else:
            print(f" \n -------------------- \n Angle encoding for second PQC \n --------------------")
            self.model = QuantumClassifier_v2(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                modes=self.modes,
                num_classes=len(self.classes_),
                input_state=self.input_state,
                device=self.device,
                no_bunching=self.no_bunching,
                E=self.E,
            )

        print(
            f"\n ---- Number of parameters in Quantum head: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}"
        )
        self.model = self.model.to(self.device)

    def _train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            # Forward pass
            outputs = self.model(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()

        train_accuracy = 100 * correct_train / total_train
        return epoch_loss / len(train_loader), train_accuracy

    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_accuracy = 100 * correct / total
        return epoch_loss / len(val_loader), val_accuracy

    def fit(self, x, y, val_x=None, val_y=None):
        """Train the QLayer with a manual training loop and optional validation data."""
        # Store classes
        self.classes_ = np.unique(y)
        print(f"\n -- Self.classes_ = {self.classes_} --")

        # Initialize model
        self._initialize_model()

        # Convert to PyTorch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Setup validation data if provided
        val_loader = None
        if val_x is not None and val_y is not None:
            val_x_tensor = torch.tensor(val_x, dtype=torch.float32)
            val_y_tensor = torch.tensor(val_y, dtype=torch.long)
            val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Add learning rate scheduler based on validation loss
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        print(
            f"\n Set up the optimizer with lr: {self.lr} and weight_decay = {self.weight_decay}"
        )
        best_val = 0
        self.best_model_state_dict = None

        # Training loop
        print("Entering in the training loop")
        for epoch in range(self.epochs):
            # Train for one epoch
            # print("\n -- train step ...")
            train_loss, train_accuracy = self._train_epoch(
                train_loader, criterion, optimizer
            )
            # print("\n --- train step done ---")
            self.history["train_loss"].append(train_loss)
            self.train_accuracies.append(train_accuracy)

            # Validation phase
            val_loss, val_accuracy = 0, 0
            if val_loader is not None:
                # print("\n -- val step ...")
                val_loss, val_accuracy = self._validate_epoch(val_loader, criterion)
                # print("\n --- val step done ---")
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_accuracy)
                self.val_accuracies.append(val_accuracy)
                if val_accuracy >= best_val:
                    best_val = val_accuracy
                    self.best_model_state_dict = copy.deepcopy(self.model.state_dict())

                # Step the scheduler with validation loss
                scheduler.step()

            # if (epoch + 1) % 50 == 0:
            if val_loader is not None:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% - Best Val Acc: {best_val:.2f} - [lr = {scheduler.get_last_lr()[0]}]"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%"
                )

        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            print(f"Best model loaded with validation accuracy: {best_val:.2f}%")
        self.is_fitted_ = True
        self.best_val = best_val
        return self

    def predict(self, x):
        """Predict class labels for samples in X."""
        self._check_is_fitted()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            _, predicted = torch.max(outputs, 1)
            print(
                f"Output and predicted in predict of shape and classes {outputs.shape} and {predicted.shape} and {self.classes_}"
            )

        return self.classes_[predicted.cpu().numpy()]

    def predict_proba(self, x):
        """Predict class probabilities for samples in X."""
        self._check_is_fitted()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

    def _check_is_fitted(self):
        """Check if model is fitted."""
        if not self.is_fitted_ or self.model is None:
            raise ValueError(
                "This model has not been fitted yet. Call 'fit' before using this method."
            )


def create_setfit_with_q_layer(
    model,
    input_dim=768,
    hidden_dim=100,
    modes=10,
    num_classes=2,
    epochs=100,
    lr=0.001,
    input_state=None,
    no_bunching=False,
    expectation = True,
    vanilla = False,
    E_dim = 1,
):
    """
    Replace the classification head of a SetFit model with a quantum layer.

    Args:
        model: SetFit model to modify
        input_dim: Dimension of input embeddings
        hidden_dim: Dimension after downscaling
        modes: Number of modes in the quantum circuit
        num_classes: Number of output classes
        epochs: Training epochs for the quantum head
        input_state: Photon distribution across modes

    Returns:
        Modified SetFit model with quantum classification head
    """
    # Get the device the model is on
    device = next(model.model_body.parameters()).device
    # Replace model head with QLayer
    model.model_head = QLayerTraining(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        modes=modes,
        num_classes=num_classes,
        epochs=epochs,
        input_state=input_state,
        device=device,
        no_bunching=no_bunching,
        lr=lr,
        expectation = expectation,
        vanilla = vanilla,
        E = E_dim,
    )
    print(f"\n -> Model with {modes} modes and input_state = {input_state} initialized")
    return model


### utility functions ###
def save_experiment_results(results, filename="ft-qllm_exp.json"):
    """
    Append experiment results to a JSON file.

    Args:
        results (dict): Dictionary containing experiment results
        filename (str): Path to the JSON file to store results
    """
    filename = os.path.join("./results", filename)

    # Create results directory if it doesn't exist
    os.makedirs("./results", exist_ok=True)

    # Check if file exists and load existing data
    if os.path.exists(filename):
        try:
            with open(filename) as file:
                all_results = json.load(file)
        except json.JSONDecodeError:
            all_results = []
    else:
        all_results = []

    # Append new results
    all_results.append(results)

    # Write updated data back to file
    with open(filename, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"Results saved. Total experiments: {len(all_results)}")
    return len(all_results)


def train_quantum_heads(
    model,
    sentence_transformer,
    train_embeddings,
    global_train_max,
    global_train_min,
    train_labels,
    eval_dataset,
    test_dataset,
    args,
    num_classes,
    device,
    results_folder,
    use_normalization,
):
    """Train quantum classification heads"""
    if args.verbose:
        print("\n4. Training Quantum Layer heads...")

    quantum_results = {}

    for mode in args.quantum_modes:
        photon_max = int(mode // 2) if args.photons <= 0 else args.photons
        if args.photons > int(mode // 2):
            photon_max = int(mode // 2)
        if args.photons_max:
            photon_max = 3
        for k in range(1, photon_max + 1):
            # Create input state with k photons
            input_state = [0] * mode
            for p in range(k):
                input_state[2 * p] = 1

            if args.verbose:
                print(f"\n   Training Quantum Head: {mode} modes, {k} photons")
                print(f"   Input state: {input_state}")

            # Create quantum model
            model = create_setfit_with_q_layer(
                model,
                input_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                modes=mode,
                num_classes=num_classes,
                epochs=args.head_epochs,
                input_state=input_state,
                no_bunching=args.no_bunching,
                lr=args.lr_q,
                expectation = args.expectation_encoding,
                vanilla = args.vanilla,
                E_dim=args.e_dim,
            )
            # Move model to device BEFORE training to avoid parameter issues
            model = model.to(device)
            print("\n -> Model sent to device")

            # Generate test embeddings for validation during training
            test_embeddings = []

            with torch.no_grad():
                num_batches = (
                    len(test_dataset["sentence"]) + args.batch_size - 1
                ) // args.batch_size
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * args.batch_size
                    end_idx = min(
                        start_idx + args.batch_size, len(test_dataset["sentence"])
                    )
                    batch_texts = test_dataset["sentence"][start_idx:end_idx]
                    batch_embeddings = sentence_transformer.encode(
                        batch_texts, convert_to_tensor=True
                    )
                    test_embeddings.extend(batch_embeddings.detach().cpu().numpy())
            # normalization
            test_embeddings = np.array(test_embeddings)
            if use_normalization:
                test_embeddings = (test_embeddings - global_train_min) / (
                    global_train_max - global_train_min
                )
                test_embeddings = np.clip(test_embeddings, 0, 1)
            else:
                test_embeddings = torch.tensor(test_embeddings)
            # Val dataset
            eval_embeddings = sentence_transformer.encode(eval_dataset["sentence"])
            if use_normalization:
                eval_embeddings = (eval_embeddings - global_train_min) / (
                    global_train_max - global_train_min
                )
                eval_embeddings = torch.tensor(np.clip(eval_embeddings, 0, 1))
            else:
                eval_embeddings = torch.tensor(eval_embeddings)
            print(f" -> test embeddings obtained: {test_embeddings.shape}")

            # Train the quantum head with test data as validation
            print("\n Training the quantum head")
            model.model_head.fit(
                train_embeddings, train_labels, eval_embeddings, eval_dataset["label"]
            )

            # Fallback to using the predict method
            q_val_predictions = model.model_head.predict(eval_embeddings.cpu().numpy())
            q_val_accuracy = accuracy_score(eval_dataset["label"], q_val_predictions)

            # Test evaluation using the same method as MLP
            test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32)
            q_test_accuracy, _ = evaluate(
                model, test_embeddings_tensor, test_dataset["label"]
            )
            best_val_decimal = (
                model.model_head.best_val / 100.0
            )  # Convert from percentage to decimal
            if args.verbose:
                print(
                    f"   Quantum {mode}-{k} - Val: {q_val_accuracy:.4f}, Test: {q_test_accuracy:.4f}, Best Val: {best_val_decimal:.4f}"
                )

            quantum_results[f"{mode}-qlayer-{k}"] = [
                q_val_accuracy,
                q_test_accuracy,
                best_val_decimal,
            ]

            # Save individual result immediately after each experiment
            save_individual_quantum_result(
                mode,
                k,
                [q_val_accuracy, q_test_accuracy, best_val_decimal],
                results_folder,
            )

    return quantum_results


def save_individual_quantum_result(mode, photons, result, results_folder):
    """Save individual quantum experiment result to prevent data loss"""
    individual_result = {
        "mode": mode,
        "photons": photons,
        "config": f"{mode}-qlayer-{photons}",
        "val_accuracy": result[0],
        "test_accuracy": result[1],
        "best_val": result[2],
        "timestamp": datetime.now().isoformat(),
    }

    # Save to individual result file
    individual_file = f"{results_folder}/quantum_individual_results.json"
    with open(individual_file, "a") as f:
        f.write(json.dumps(individual_result) + "\n")

    print(
        f"Saved individual result for {mode} modes, {photons} photons to {individual_file}"
    )
