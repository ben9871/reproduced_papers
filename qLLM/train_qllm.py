import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from setfit import SetFitModel
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm
from utils_qllm import (
    ModelWrapper,
    SupConLoss,
    create_setfit_with_q_layer,
    evaluate,
    replace_setfit_head_with_mlp,
)


def create_results_folder():
    """Create a timestamped results folder and return its path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"results/experiment_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)
    return results_folder


def save_arguments(args, results_folder):
    """Save experiment arguments to JSON file"""
    args_dict = vars(args)
    with open(f"{results_folder}/arguments.json", "w") as f:
        json.dump(args_dict, f, indent=2)


def save_results(results, results_folder):
    """Save experiment results to JSON file"""
    with open(f"{results_folder}/results.json", "w") as f:
        json.dump(results, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train qLLM with classical and quantum heads"
    )

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="sst2", help="Dataset name")
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=8,
        help="Number of samples per class (few-shot setting)",
    )
    parser.add_argument(
        "--eval-size", type=int, default=250, help="Validation set size"
    )

    # Model parameters
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-mpnet-base-v2",
        help="Pre-trained model name",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=768, help="Embedding dimension"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=100,
        help="Hidden dimension for MLP/Quantum layers",
    )

    # Training parameters
    parser.add_argument(
        "--body-epochs",
        type=int,
        default=20,
        help="Epochs for sentence transformer fine-tuning",
    )
    parser.add_argument(
        "--head-epochs",
        type=int,
        default=200,
        help="Epochs for classification head training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for body fine-tuning",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for evaluation"
    )

    # Quantum parameters
    parser.add_argument(
        "--quantum-modes",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8],
        help="List of quantum modes to test",
    )

    # Execution parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def setup_environment(args):
    """Set up random seeds and device"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.verbose:
        print(f"Using device: {device}")
        print("Configuration:")
        print(f"- Samples per class: {args.samples_per_class}")
        print(f"- Body training epochs: {args.body_epochs}")
        print(f"- Head training epochs: {args.head_epochs}")
        print(f"- Learning rate: {args.learning_rate}")

    return device


def load_data(args):
    """Load and prepare the dataset according to paper: 256 samples per label, 85% train, 15% val"""
    if args.verbose:
        print(f"\nLoading dataset '{args.dataset}' with paper sampling strategy...")

    dataset = load_dataset(args.dataset)

    # Paper approach: 256 samples from each label {+, -}
    full_train_dataset = dataset["train"]

    # Separate by label
    label_0_samples = [
        example for example in full_train_dataset if example["label"] == 0
    ]
    label_1_samples = [
        example for example in full_train_dataset if example["label"] == 1
    ]

    # Sample 256 from each label
    np.random.seed(args.seed)
    samples_per_label = 256

    if len(label_0_samples) < samples_per_label:
        raise ValueError(
            f"Not enough samples for label 0: {len(label_0_samples)} < {samples_per_label}"
        )
    if len(label_1_samples) < samples_per_label:
        raise ValueError(
            f"Not enough samples for label 1: {len(label_1_samples)} < {samples_per_label}"
        )

    selected_label_0 = np.random.choice(
        len(label_0_samples), samples_per_label, replace=False
    )
    selected_label_1 = np.random.choice(
        len(label_1_samples), samples_per_label, replace=False
    )

    sampled_data = []
    sampled_data.extend([label_0_samples[i] for i in selected_label_0])
    sampled_data.extend([label_1_samples[i] for i in selected_label_1])

    # Shuffle the combined data
    np.random.shuffle(sampled_data)

    # Split into 85% training and 15% validation
    total_samples = len(sampled_data)  # 512 samples
    train_size = int(0.85 * total_samples)  # 435 samples

    train_data = sampled_data[:train_size]
    val_data = sampled_data[train_size:]

    # Create datasets in the expected format
    train_dataset = {
        "sentence": [ex["sentence"] for ex in train_data],
        "label": [ex["label"] for ex in train_data],
    }
    eval_dataset = {
        "sentence": [ex["sentence"] for ex in val_data],
        "label": [ex["label"] for ex in val_data],
    }

    # Use remaining original training data as test set (N - 2×256 samples)
    # Track which original indices were used for sampling
    used_label_0_indices = set()
    used_label_1_indices = set()

    # Map selected indices back to original dataset indices
    label_0_idx = 0
    label_1_idx = 0
    for i, example in enumerate(full_train_dataset):
        if example["label"] == 0:
            if label_0_idx in selected_label_0:
                used_label_0_indices.add(i)
            label_0_idx += 1
        else:  # label == 1
            if label_1_idx in selected_label_1:
                used_label_1_indices.add(i)
            label_1_idx += 1

    used_indices = used_label_0_indices | used_label_1_indices
    remaining_indices = [
        i for i in range(len(full_train_dataset)) if i not in used_indices
    ]
    test_dataset = {
        "sentence": [full_train_dataset[i]["sentence"] for i in remaining_indices],
        "label": [full_train_dataset[i]["label"] for i in remaining_indices],
    }

    # Extract texts and labels for contrastive learning
    texts = train_dataset["sentence"]
    features = [texts]
    labels = torch.tensor(train_dataset["label"])

    if args.verbose:
        print("Paper sampling strategy applied:")
        print(f"- Sampled {samples_per_label} samples from each label")
        print(
            f"- Training: {len(train_dataset['sentence'])} samples ({len(train_dataset['sentence']) / total_samples * 100:.1f}%)"
        )
        print(
            f"- Validation: {len(eval_dataset['sentence'])} samples ({len(eval_dataset['sentence']) / total_samples * 100:.1f}%)"
        )
        print(f"- Test: {len(test_dataset['sentence'])} samples")
        print(f"- Train label distribution: {np.bincount(train_dataset['label'])}")
        print(f"- Val label distribution: {np.bincount(eval_dataset['label'])}")

    return train_dataset, eval_dataset, test_dataset, features, labels


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

    return train_embeddings, train_labels


def train_classical_heads(
    sentence_transformer,
    train_embeddings,
    train_labels,
    eval_dataset,
    test_dataset,
    args,
    device,
):
    """Train classical classification heads"""
    results = {}
    num_classes = len(set(train_labels))

    if args.verbose:
        print(
            f"\nTraining classical classification heads for {num_classes}-class classification..."
        )

    # Logistic Regression
    if args.verbose:
        print("\n1. Training Logistic Regression head...")

    model = SetFitModel.from_pretrained(args.model_name)
    model.model_body = sentence_transformer
    model.model_head.fit(train_embeddings, train_labels)

    lg_val_accuracy, _ = evaluate(
        model, eval_dataset["sentence"], eval_dataset["label"]
    )
    lg_test_accuracy, _ = evaluate(
        model, test_dataset["sentence"], test_dataset["label"]
    )

    if args.verbose:
        print(
            f"Logistic Regression - Val: {lg_val_accuracy:.4f}, Test: {lg_test_accuracy:.4f}"
        )

    results["LogisticRegression"] = [lg_val_accuracy, lg_test_accuracy]

    # SVM
    if args.verbose:
        print("\n2. Training SVM head...")

    model.model_head = SVC(C=1.0, kernel="linear", gamma="scale", probability=True)
    model.model_head.fit(train_embeddings, train_labels)

    svc_val_accuracy, _ = evaluate(
        model, eval_dataset["sentence"], eval_dataset["label"]
    )
    svc_test_accuracy, _ = evaluate(
        model, test_dataset["sentence"], test_dataset["label"]
    )

    if args.verbose:
        print(f"SVM - Val: {svc_val_accuracy:.4f}, Test: {svc_test_accuracy:.4f}")

    results["SVC"] = [svc_val_accuracy, svc_test_accuracy]

    # MLP
    if args.verbose:
        print("\n3. Training MLP head...")

    model = replace_setfit_head_with_mlp(
        model,
        input_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        epochs=args.head_epochs,
    )
    model.model_head.fit(train_embeddings, train_labels)

    mlp_val_accuracy, _ = evaluate(
        model, eval_dataset["sentence"], eval_dataset["label"]
    )
    mlp_test_accuracy, _ = evaluate(
        model, test_dataset["sentence"], test_dataset["label"]
    )

    if args.verbose:
        print(f"MLP - Val: {mlp_val_accuracy:.4f}, Test: {mlp_test_accuracy:.4f}")

    results["MLP"] = [mlp_val_accuracy, mlp_test_accuracy]

    return results, model, num_classes


def train_quantum_heads(
    model,
    sentence_transformer,
    train_embeddings,
    train_labels,
    eval_dataset,
    test_dataset,
    args,
    num_classes,
    device,
):
    """Train quantum classification heads"""
    if args.verbose:
        print("\n4. Training Quantum Layer heads...")

    quantum_results = {}

    for mode in args.quantum_modes:
        photon_max = int(mode // 2)

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
            )

            # Train the quantum head
            model.model_head.fit(train_embeddings, train_labels)

            # Evaluate
            q_val_predictions = model.model_head.predict(
                sentence_transformer.encode(
                    eval_dataset["sentence"], convert_to_tensor=True
                )
                .cpu()
                .numpy()
            )
            q_val_accuracy = accuracy_score(eval_dataset["label"], q_val_predictions)

            q_test_predictions = model.model_head.predict(
                sentence_transformer.encode(
                    test_dataset["sentence"], convert_to_tensor=True
                )
                .cpu()
                .numpy()
            )
            q_test_accuracy = accuracy_score(test_dataset["label"], q_test_predictions)

            if args.verbose:
                print(
                    f"   Quantum {mode}-{k} - Val: {q_val_accuracy:.4f}, Test: {q_test_accuracy:.4f}"
                )

            quantum_results[f"{mode}-qlayer-{k}"] = [q_val_accuracy, q_test_accuracy]

    return quantum_results


def plot_results_comparison(results, results_folder):
    """Plot comparison of classical and quantum results"""
    # Extract results for visualization
    classical_methods = ["LogisticRegression", "SVC", "MLP"]
    classical_val_accs = [results[method][0] for method in classical_methods]
    classical_test_accs = [results[method][1] for method in classical_methods]

    # Process quantum results
    quantum_configs = list(results["Qlayer"].keys())
    quantum_val_accs = [results["Qlayer"][config][0] for config in quantum_configs]
    quantum_test_accs = [results["Qlayer"][config][1] for config in quantum_configs]

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Validation accuracies
    x_classical = range(len(classical_methods))
    x_quantum = range(
        len(classical_methods), len(classical_methods) + len(quantum_configs)
    )
    x_labels_quantum = [
        f"{c.split('-qlayer-')[0]}-{c.split('-qlayer-')[1]}" for c in quantum_configs
    ]

    ax1.bar(x_classical, classical_val_accs, color="skyblue", label="Classical")
    ax1.bar(x_quantum, quantum_val_accs, color="lightcoral", label="Quantum")
    ax1.set_xticks(list(x_classical) + list(x_quantum))
    ax1.set_xticklabels(classical_methods + x_labels_quantum, rotation=45, ha="right")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_title("Validation Performance Comparison")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Test accuracies
    ax2.bar(x_classical, classical_test_accs, color="skyblue", label="Classical")
    ax2.bar(x_quantum, quantum_test_accs, color="lightcoral", label="Quantum")
    ax2.set_xticks(list(x_classical) + list(x_quantum))
    ax2.set_xticklabels(classical_methods + x_labels_quantum, rotation=45, ha="right")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Test Performance Comparison")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save the plot to the results folder
    plot_path = f"{results_folder}/comparison_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()


def print_results_summary(results):
    """Print summary of all results"""
    print("\n=== RESULTS SUMMARY ===")
    print("\nClassical Methods:")
    classical_methods = ["LogisticRegression", "SVC", "MLP"]
    classical_val_accs = [results[method][0] for method in classical_methods]
    classical_test_accs = [results[method][1] for method in classical_methods]

    for i, method in enumerate(classical_methods):
        print(
            f"{method:20s} - Val: {classical_val_accs[i]:.4f}, Test: {classical_test_accs[i]:.4f}"
        )

    print("\nQuantum Methods (best per mode count):")
    quantum_configs = list(results["Qlayer"].keys())
    modes_processed = set()
    for config in quantum_configs:
        mode_count = config.split("-")[0]
        if mode_count not in modes_processed:
            # Find best accuracy for this mode count
            mode_configs = [
                c for c in quantum_configs if c.startswith(mode_count + "-")
            ]
            best_val = max(results["Qlayer"][c][0] for c in mode_configs)
            best_test = max(results["Qlayer"][c][1] for c in mode_configs)
            print(
                f"{mode_count + ' modes':20s} - Val: {best_val:.4f}, Test: {best_test:.4f}"
            )
            modes_processed.add(mode_count)


def main():
    """Main training pipeline"""
    args = parse_args()
    device = setup_environment(args)

    # Create results folder for this experiment
    results_folder = create_results_folder()

    # Save experiment arguments
    save_arguments(args, results_folder)

    if args.verbose:
        print(f"Results will be saved to: {results_folder}")

    # Load data and model
    train_dataset, eval_dataset, test_dataset, features, labels = load_data(args)
    model, sentence_transformer = load_model(args, device)

    # Train body with contrastive learning
    sentence_transformer = train_body_with_contrastive_learning(
        sentence_transformer, features, labels, args, device
    )

    # Generate embeddings
    train_embeddings, train_labels = generate_embeddings(
        sentence_transformer, train_dataset, args
    )

    # Train classical heads
    results, model, num_classes = train_classical_heads(
        sentence_transformer,
        train_embeddings,
        train_labels,
        eval_dataset,
        test_dataset,
        args,
        device,
    )

    # Train quantum heads
    quantum_results = train_quantum_heads(
        model,
        sentence_transformer,
        train_embeddings,
        train_labels,
        eval_dataset,
        test_dataset,
        args,
        num_classes,
        device,
    )

    # Combine results
    results["Qlayer"] = quantum_results
    results.update(
        {
            "training_samples": args.samples_per_class,
            "epochs": args.body_epochs,
            "lr": args.learning_rate,
        }
    )

    # Save results to JSON
    save_results(results, results_folder)

    # Display results
    if not args.no_plot:
        plot_results_comparison(results, results_folder)

    print_results_summary(results)

    if args.verbose:
        print(f"\nExperiment completed! Results saved to: {results_folder}")


if __name__ == "__main__":
    main()
