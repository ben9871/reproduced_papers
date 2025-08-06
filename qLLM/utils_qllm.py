import copy
import json
import math
import os

import merlin as ml  # Using our Merlin framework
import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# the `ModelWrapper` class provides a unified interface for
## handling tokenization
## forward passes with sentence transformer models
# here, we can work with different model architectures seamlessly


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def tokenize(self, texts):
        """
        Delegates tokenization to the underlying model.

        Args:
            texts (List[str]): List of text strings to tokenize

        Returns:
            Dict or Tensor: Tokenized inputs in the format expected by the model
        """
        try:
            # Try to use the tokenize method of the underlying model
            return self.model.tokenize(texts)
        except AttributeError:
            # If the model doesn't have a tokenize method, try alternative approaches
            if hasattr(self.model, "tokenizer"):
                return self.model.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True
                )
            elif hasattr(self.model, "_first_module") and hasattr(
                self.model._first_module, "tokenizer"
            ):
                return self.model._first_module.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True
                )
            else:
                raise ValueError(
                    "Unable to tokenize texts with this model. Please provide a model that has a tokenize or tokenizer method."
                ) from None

    def forward(self, inputs):
        """
        Process inputs through the model to get embeddings.

        Args:
            inputs: Can be raw text strings or pre-tokenized inputs

        Returns:
            torch.Tensor: The sentence embeddings
        """
        try:
            # Handle different input formats
            if isinstance(inputs, dict) and all(
                isinstance(v, torch.Tensor) for v in inputs.values()
            ):
                outputs = self.model(inputs)
            elif isinstance(inputs, list) and all(isinstance(t, str) for t in inputs):
                tokenized = self.tokenize(inputs)
                device = next(self.model.parameters()).device
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                outputs = self.model(tokenized)
            else:
                outputs = self.model(inputs)

            # Extract embeddings from various output formats
            if isinstance(outputs, dict) and "sentence_embedding" in outputs:
                return outputs["sentence_embedding"]
            elif isinstance(outputs, dict) and "pooler_output" in outputs:
                return outputs["pooler_output"]
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                return outputs[0]
            else:
                return outputs
        except Exception as e:
            raise ValueError(f"Error during forward pass: {str(e)}") from e


## evaluation function ##


def evaluate(model, embeddings, labels):
    """
    Evaluate SetFit model on given texts and labels.

    Args:
        model: SetFit model with a trained classification head
        texts: List of text strings to classify
        labels: True labels for evaluation

    Returns:
        tuple: (accuracy, predictions)
    """
    batch_size = 16
    num_samples = embeddings.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    all_embeddings = []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # batch_texts = texts[start_idx:end_idx]

            # Get embeddings
            """batch_embeddings = model.model_body.encode(
                batch_texts, convert_to_tensor=True
            )"""
            batch_embeddings = embeddings[start_idx:end_idx]
            batch_embeddings_cpu = batch_embeddings  # .detach().cpu().numpy()

            all_embeddings.extend(batch_embeddings_cpu)

    # Use the classification head to predict
    predictions = model.model_head.predict(np.array(all_embeddings))
    accuracy = accuracy_score(labels, predictions)
    return accuracy, predictions


# MLPClassifier and its wrapper#


class MLPClassifier(nn.Module):
    """3-layer MLP classifier with dropout regularization"""

    def __init__(self, input_dim, hidden_dim=100, num_classes=2):
        super().__init__()
        self.layers = (
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )
            if hidden_dim > 0
            else nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for the MLP classifier"""

    def __init__(
        self,
        input_dim=768,
        hidden_dim=100,
        num_classes=2,
        lr=0.001,
        epochs=100,
        batch_size=32,
        device=None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.classes_ = None
        self.train_accuracies = []
        self.val_accuracies = []

    def fit(self, x, y, val_x=None, val_y=None):
        """Train the MLP classifier with optional validation data"""
        # Store unique classes
        self.classes_ = np.unique(y)

        # Convert numpy arrays to PyTorch tensors
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        if val_x is not None and val_y is not None:
            val_x = torch.tensor(val_x, dtype=torch.float32).to(self.device)
            val_y = torch.tensor(val_y, dtype=torch.long).to(self.device)

        # Initialize the model
        self.model = MLPClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.classes_),
        ).to(self.device)

        print(
            f"Number of parameters in MLP head: {sum([p.numel() for p in self.model.parameters()])}"
        )

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )

        # Add learning rate scheduler based on validation loss
        """scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )"""

        best_val = 0
        self.best_model_state_dict = None
        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            indices = torch.randperm(len(x))
            total_loss = 0
            correct_train = 0
            total_train = 0

            for i in range(0, len(x), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                batch_x = x[batch_indices]
                batch_y = y_tensor[batch_indices]

                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()

            train_accuracy = 100 * correct_train / total_train
            self.train_accuracies.append(train_accuracy)

            # Validation phase
            val_accuracy = 0
            # val_loss = 0
            if val_x is not None and val_y is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(val_x)
                    # val_loss = criterion(val_outputs, val_y).item()
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_accuracy = (
                        100 * (val_predicted == val_y).sum().item() / val_y.size(0)
                    )
                    self.val_accuracies.append(val_accuracy)
            if val_accuracy > best_val:
                best_val = val_accuracy
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())

            # Step the scheduler with validation loss (if validation data is provided)
            """if val_x is not None and val_y is not None:
                scheduler.step(val_loss)
            """
            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / (len(x) // self.batch_size + 1)
                if val_x is not None and val_y is not None:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}% (Best Val: {best_val:.2f}%)  [lr = {optimizer.param_groups[0]['lr']}]"
                    )
                else:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%"
                    )
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            print(f"Best model loaded with validation accuracy: {best_val:.2f}%")
        self.best_val = best_val
        return self

    def predict(self, x):
        """Predict classes for samples"""
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            _, predicted = torch.max(outputs, 1)
            return self.classes_[predicted.cpu().numpy()]

    def predict_proba(self, x):
        """Predict class probabilities"""
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            return probabilities


def replace_setfit_head_with_mlp(
    model, input_dim=768, hidden_dim=100, num_classes=2, epochs=100
):
    """Replace the classification head of a SetFitModel with an MLP."""
    # Get the device the model is on
    device = next(model.model_body.parameters()).device

    # Create new MLP head
    mlp_head = MLPClassifierWrapper(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        epochs=epochs,
        lr=0.001,
        device=device,
    )

    # Replace the model head
    model.model_head = mlp_head

    return model


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
            f"Building a model with {modes} modes and {input_state} as an input state and no_bunching = {no_bunching}"
        )
        # this layer downscale the inputs to fit in the QLayer
        self.downscaling_layer = nn.Linear(input_dim, hidden_dim, device=device)

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
        out = self.downscaling_layer(x)
        # casts the input to [0,1] + use q_circuit
        out = self.q_circuit(self.sig(out))
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

    def _initialize_model(self):
        """Initialize or re-initialize the model."""
        self.model = QuantumClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            modes=self.modes,
            num_classes=len(self.classes_),
            input_state=self.input_state,
            device=self.device,
            no_bunching=self.no_bunching,
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

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
                if val_accuracy > best_val:
                    best_val = val_accuracy
                    self.best_model_state_dict = copy.deepcopy(self.model.state_dict())

                # Step the scheduler with validation loss
                scheduler.step(val_loss)

            # if (epoch + 1) % 50 == 0:
            if val_loader is not None:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% - Best Val Acc: {best_val:.2f} - [lr = {optimizer.param_groups[0]['lr']}]"
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


## SupConLoss ##


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(
        self, model, temperature=0.07, contrast_mode="all", base_temperature=0.07
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, sentence_features, labels=None, mask=None):
        """Computes loss for model."""
        # Au lieu d'utiliser encode() qui peut détacher le graphe de calcul,
        # utilisons directement le modèle pour générer les embeddings
        # Tokenize the inputs
        tokenized_inputs = self.model.tokenize(sentence_features[0])

        # Si le modèle est sur un device particulier, déplacer les inputs sur ce device
        device = next(self.model.parameters()).device
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

        # Forward pass avec le modèle
        outputs = self.model(tokenized_inputs)

        # Récupérer les embeddings
        if isinstance(outputs, dict) and "sentence_embedding" in outputs:
            features = outputs["sentence_embedding"]
        else:
            # Si le modèle renvoie un format différent, adaptez ici
            features = outputs  # Ou une autre méthode pour extraire les embeddings

        # Normalize embeddings
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        # Add n_views dimension
        features = torch.unsqueeze(features, 1)
        device = features.device

        # Le reste du code reste inchangé
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
