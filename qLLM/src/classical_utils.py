"""
Classical model training utilities for qLLM experiments.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from setfit import SetFitModel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm

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
                nn.BatchNorm1d(hidden_dim),
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
        gamma=0.99,
        wd=1e-6,
        epochs=100,
        batch_size=32,
        device=None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.gamma = gamma
        self.weight_decay = wd
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
        print(f"\n --- \n Training MLP head with lr = {self.lr}, weight_decay = ")
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Add learning rate scheduler based on validation loss
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

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
            if val_x is not None and val_y is not None:
                scheduler.step()

            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / (len(x) // self.batch_size + 1)
                if val_x is not None and val_y is not None:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}% (Best Val: {best_val:.2f}%)  [lr = {scheduler.get_last_lr()[0]}]"
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
    model,
    input_dim=768,
    hidden_dim=100,
    num_classes=2,
    epochs=100,
    lr=0.001,
    gamma=0.99,
    wd=1e-6,
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
        lr=lr,
        gamma=gamma,
        wd=wd,
        device=device,
    )

    # Replace the model head
    model.model_head = mlp_head

    return model


def load_model(args, device):
    """Load the pre-trained model"""
    if args.verbose:
        print(f"\nLoading pre-trained model: {args.model_name}")

    model = SetFitModel.from_pretrained(args.model_name)
    sentence_transformer = model.model_body

    # Move model to device
    model = model.to(device)
    sentence_transformer = sentence_transformer.to(device)

    if args.verbose:
        print(f"Model loaded: {type(sentence_transformer).__name__}")
        print(f"Embedding dimension: {args.embedding_dim}")
        print(f"Model moved to device: {device}")

    return model, sentence_transformer


def train_body_with_contrastive_learning(
    sentence_transformer, features, labels, args, device
):
    """Train the model body with contrastive learning"""
    if args.verbose:
        print("\nTraining model body with contrastive learning...")

    model_wrapped = ModelWrapper(sentence_transformer)
    criterion = SupConLoss(model=model_wrapped)
    # Move labels to device
    labels = labels.to(device)

    # Enable gradients for fine-tuning
    for param in sentence_transformer.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model_wrapped.parameters(), lr=args.learning_rate)
    model_wrapped.train()

    # Training loop
    for iteration in tqdm(range(args.body_epochs), desc="Contrastive Learning"):
        optimizer.zero_grad()
        loss = criterion(features, labels)
        loss.backward()
        optimizer.step()

        if args.verbose and (iteration + 1) % 5 == 0:
            print(
                f"Iteration {iteration + 1}/{args.body_epochs}, Loss: {loss.item():.6f}"
            )

    if args.verbose:
        print("Model body fine-tuning completed!")

    return sentence_transformer


def generate_embeddings(sentence_transformer, train_dataset, args):
    """Generate embeddings from the fine-tuned model"""
    if args.verbose:
        print("\nGenerating embeddings for training data...")

    sentence_transformer.eval()
    train_embeddings = []
    train_labels = []

    with torch.no_grad():
        num_batches = (
            len(train_dataset["sentence"]) + args.batch_size - 1
        ) // args.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Encoding"):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(train_dataset["sentence"]))

            batch_texts = train_dataset["sentence"][start_idx:end_idx]
            batch_labels = train_dataset["label"][start_idx:end_idx]

            batch_embeddings = sentence_transformer.encode(
                batch_texts, convert_to_tensor=True
            )
            batch_embeddings_cpu = batch_embeddings.detach().cpu().numpy()

            for emb, lbl in zip(batch_embeddings_cpu, batch_labels):
                train_embeddings.append(emb)
                train_labels.append(lbl)

    train_embeddings = np.array(train_embeddings)
    train_labels = np.array(train_labels)

    if args.verbose:
        print(f"Embeddings shape: {train_embeddings.shape}")
        print(f"Labels shape: {train_labels.shape}")
    # scaling on the embeddings
    global_train_max = train_embeddings.max()
    global_train_min = train_embeddings.min()
    # min to 0 and max to 1
    train_embeddings_scaled = (train_embeddings - global_train_min) / (
        global_train_max - global_train_min
    )
    return (
        train_embeddings,
        train_embeddings_scaled,
        train_labels,
        global_train_max,
        global_train_min,
    )


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


def train_classical_heads(
    sentence_transformer,
    train_embeddings,
    train_labels,
    global_train_max,
    global_train_min,
    eval_dataset,
    test_dataset,
    args,
    device,
    use_normalization,
):
    """Train classical classification heads"""
    results = {}
    num_classes = len(set(train_labels))

    if args.verbose:
        print(
            f"\nTraining classical classification heads for {num_classes}-class classification..."
        )
        print("\n Extracting validation and test embeddings...")

    # eval embeddings
    eval_embeddings = sentence_transformer.encode(eval_dataset["sentence"])
    if use_normalization:
        eval_embeddings = (eval_embeddings - global_train_min) / (
            global_train_max - global_train_min
        )
        eval_embeddings = torch.tensor(np.clip(eval_embeddings, 0, 1))
    else:
        eval_embeddings = torch.tensor(eval_embeddings)
    # test embeddings
    test_embeddings = sentence_transformer.encode(test_dataset["sentence"])
    if use_normalization:
        test_embeddings = (test_embeddings - global_train_min) / (
            global_train_max - global_train_min
        )
        test_embeddings = torch.tensor(np.clip(test_embeddings, 0, 1))
    else:
        test_embeddings = torch.tensor(test_embeddings)

    model = SetFitModel.from_pretrained(args.model_name)
    model.model_body = sentence_transformer

    # Logistic Regression
    if args.verbose:
        print("\n1. Training Logistic Regression head...")

    model.model_head.fit(train_embeddings, train_labels)

    lg_val_accuracy, _ = evaluate(model, eval_embeddings, eval_dataset["label"])
    lg_test_accuracy, _ = evaluate(model, test_embeddings, test_dataset["label"])

    if args.verbose:
        print(
            f"Logistic Regression - Val: {lg_val_accuracy:.4f}, Test: {lg_test_accuracy:.4f}"
        )

    results["LogisticRegression"] = [lg_val_accuracy, lg_test_accuracy]

    # SVM with varying parameter counts (targeting 296 and 435 parameters)
    if args.verbose:
        print("\n2. Training SVM heads with varying parameter counts...")

    # Configuration 1: Target ~296 parameters (moderate regularization)
    if args.verbose:
        print("   2a. Training SVM targeting ~296 parameters...")

    model.model_head = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True)
    model.model_head.fit(train_embeddings, train_labels)

    svc_296_val_accuracy, _ = evaluate(model, eval_embeddings, eval_dataset["label"])
    svc_296_test_accuracy, _ = evaluate(model, test_embeddings, test_dataset["label"])

    n_support_vectors_296 = model.model_head.n_support_.sum()

    if args.verbose:
        print(
            f"   SVM (296 target) - Support vectors: {n_support_vectors_296}, Val: {svc_296_val_accuracy:.4f}, Test: {svc_296_test_accuracy:.4f}"
        )

    # Configuration 2: Target ~435 parameters (low regularization to use more support vectors)
    if args.verbose:
        print("   2b. Training SVM targeting ~435 parameters...")

    model.model_head = SVC(C=100.0, kernel="rbf", gamma="scale", probability=True)
    model.model_head.fit(train_embeddings, train_labels)

    svc_435_val_accuracy, _ = evaluate(model, eval_embeddings, eval_dataset["label"])
    svc_435_test_accuracy, _ = evaluate(model, test_embeddings, test_dataset["label"])

    n_support_vectors_435 = model.model_head.n_support_.sum()

    if args.verbose:
        print(
            f"   SVM (435 target) - Support vectors: {n_support_vectors_435}, Val: {svc_435_val_accuracy:.4f}, Test: {svc_435_test_accuracy:.4f}"
        )

    results["SVC_296"] = [
        svc_296_val_accuracy,
        svc_296_test_accuracy,
        int(n_support_vectors_296),
    ]
    results["SVC_435"] = [
        svc_435_val_accuracy,
        svc_435_test_accuracy,
        int(n_support_vectors_435),
    ]

    print("\n -> Generating test embeddings for MLP evaluation")
    # Generate test embeddings for validation during training
    test_embeddings = []
    eval_labels = eval_dataset["label"]

    with torch.no_grad():
        num_batches = (
            len(test_dataset["sentence"]) + args.batch_size - 1
        ) // args.batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(test_dataset["sentence"]))
            batch_texts = test_dataset["sentence"][start_idx:end_idx]
            batch_embeddings = sentence_transformer.encode(
                batch_texts, convert_to_tensor=True
            )
            test_embeddings.extend(batch_embeddings.detach().cpu().numpy())

    test_embeddings = np.array(test_embeddings)
    if use_normalization:
        test_embeddings = (test_embeddings - global_train_min) / (
            global_train_max - global_train_min
        )
        test_embeddings = torch.tensor(np.clip(test_embeddings, 0, 1))
    # Train with validation data (convert tensor back to numpy for consistent processing)
    eval_embeddings_numpy = (
        eval_embeddings.cpu().numpy()
        if isinstance(eval_embeddings, torch.Tensor)
        else eval_embeddings
    )
    print("\n ... Done !")
    # MLP
    if args.verbose:
        print("\n3. Training MLP head...")
    hidden_dims = [0, 48, 96, 144, 192]
    for hidden_dim in hidden_dims:
        print(f"\n3. Training MLP head with hidden dimension = {hidden_dim}...")
        model = replace_setfit_head_with_mlp(
            model,
            input_dim=args.embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            epochs=args.head_epochs,
            lr=args.lr_cl,
            gamma=args.gamma_cl,
            wd=args.wd_cl,
        )

        model.model_head.fit(
            train_embeddings, train_labels, eval_embeddings_numpy, eval_labels
        )

        # Fallback to using the predict method
        mlp_val_predictions = model.model_head.predict(eval_embeddings_numpy)
        mlp_val_accuracy = accuracy_score(eval_dataset["label"], mlp_val_predictions)
        mlp_test_accuracy, _ = evaluate(model, test_embeddings, test_dataset["label"])
        best_val = (
            model.model_head.best_val / 100.0
        )  # Convert from percentage to decimal
        if args.verbose:
            print(
                f"MLP - Val: {mlp_val_accuracy:.4f}, Test: {mlp_test_accuracy:.4f}, Best Val: {best_val:.4f}"
            )

        results[f"MLP-{hidden_dim}"] = [mlp_val_accuracy, mlp_test_accuracy, best_val]

    return results, model, num_classes
