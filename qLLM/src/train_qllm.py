import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from classical_utils import (
    generate_embeddings,
    load_model,
    train_body_with_contrastive_learning,
    train_classical_heads,
)

# Import utility modules
from data_utils import create_fold_data_splits, load_data, load_data_kfold
from merlin_llm_utils import train_quantum_heads


def create_results_folder():
    """Create a timestamped results folder and return its path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"results-classical/experiment_{timestamp}"
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

    parser.add_argument(
        "--no-bunching",
        action="store_true",
        help="No bunching parameter for the Quantum Layer",
    )

    parser.add_argument(
        "--photons",
        type=int,
        default=0,
        help="Number of photons max (0 stands for modes/2)",
    )

    parser.add_argument(
        "--photons-max",
        action="store_true",
        help="If True then max photons = 3",
    )

    parser.add_argument(
        "--lr-q",
        type=float,
        default=0.001,
        help="Learning rate for the quantum head",
    )

    parser.add_argument(
        "--lr-cl",
        type=float,
        default=0.001,
        help="Learning rate for the quantum head",
    )

    parser.add_argument(
        "--gamma-cl",
        type=float,
        default=0.99,
        help="Learning rate decay through ExponentialLR scheduler",
    )

    parser.add_argument(
        "--wd-cl",
        type=float,
        default=1e-6,
        help="Weight decay for classical NN",
    )
    # Execution parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # K-fold cross validation parameters
    parser.add_argument(
        "--k-folds", type=int, default=2, help="Number of folds for cross validation"
    )
    parser.add_argument(
        "--use-kfold", action="store_true", help="Enable k-fold cross validation"
    )

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


def plot_results_comparison(results, results_folder):
    """Plot comparison of classical and quantum results"""
    # Extract results for visualization
    classical_methods_1 = ["LogisticRegression", "SVC_296", "SVC_435"]
    classical_methods_2 = ["MLP-0", "MLP-48", "MLP-96", "MLP-144", "MLP-192"]
    classical_methods = classical_methods_1 + classical_methods_2
    classical_val_accs = [results[method][0] for method in classical_methods]
    classical_test_accs = [results[method][1] for method in classical_methods_1] + [
        results[method][1] for method in classical_methods_2
    ]

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


def aggregate_kfold_results(all_fold_results, args):
    """Aggregate results across k folds"""
    if args.verbose:
        print("\nAggregating results across folds...")

    # Initialize aggregated results structure
    aggregated = {}

    # Get all method names from first fold
    first_fold = all_fold_results[0]
    method_names = [key for key in first_fold.keys() if key != "fold"]

    for method_name in method_names:
        if method_name == "Qlayer":
            # Handle quantum results separately
            aggregated[method_name] = {}
            quantum_configs = list(first_fold[method_name].keys())

            for config in quantum_configs:
                val_scores = []
                test_scores = []
                best_val_scores = []

                for fold_result in all_fold_results:
                    fold_quantum_results = fold_result[method_name][config]
                    val_scores.append(fold_quantum_results[0])
                    test_scores.append(fold_quantum_results[1])
                    best_val_scores.append(fold_quantum_results[2])

                # Calculate mean and std
                aggregated[method_name][config] = {
                    "val_mean": np.mean(val_scores),
                    "val_std": np.std(val_scores),
                    "test_mean": np.mean(test_scores),
                    "test_std": np.std(test_scores),
                    "best_val_mean": np.mean(best_val_scores),
                    "best_val_std": np.std(best_val_scores),
                    "all_folds": {
                        "val": val_scores,
                        "test": test_scores,
                        "best_val": best_val_scores,
                    },
                }
        else:
            # Handle classical methods
            val_scores = []
            test_scores = []

            for fold_result in all_fold_results:
                fold_classical_result = fold_result[method_name]
                val_scores.append(fold_classical_result[0])
                test_scores.append(fold_classical_result[1])

            aggregated[method_name] = {
                "val_mean": np.mean(val_scores),
                "val_std": np.std(val_scores),
                "test_mean": np.mean(test_scores),
                "test_std": np.std(test_scores),
                "all_folds": {"val": val_scores, "test": test_scores},
            }

            # Handle support vectors for SVC methods
            if "SVC" in method_name:
                support_vectors = []
                for fold_result in all_fold_results:
                    support_vectors.append(fold_result[method_name][2])
                aggregated[method_name]["support_vectors_mean"] = np.mean(
                    support_vectors
                )
                aggregated[method_name]["support_vectors_std"] = np.std(support_vectors)
                aggregated[method_name]["all_folds"]["support_vectors"] = (
                    support_vectors
                )

            # Handle best_val for MLP methods
            elif "MLP" in method_name:
                best_val_scores = []
                for fold_result in all_fold_results:
                    best_val_scores.append(fold_result[method_name][2])
                aggregated[method_name]["best_val_mean"] = np.mean(best_val_scores)
                aggregated[method_name]["best_val_std"] = np.std(best_val_scores)
                aggregated[method_name]["all_folds"]["best_val"] = best_val_scores

    return aggregated


def print_kfold_summary(aggregated_results):
    """Print summary of k-fold cross validation results"""
    print(
        f"\n=== K-FOLD CROSS VALIDATION RESULTS (k={aggregated_results['k_folds']}) ==="
    )
    print("\nClassical Methods (Mean ± Std):")

    # Print LogisticRegression
    lr_results = aggregated_results["LogisticRegression"]
    print(
        f"{'LogisticRegression':20s} - Val: {lr_results['val_mean']:.4f}±{lr_results['val_std']:.4f}, "
        f"Test: {lr_results['test_mean']:.4f}±{lr_results['test_std']:.4f}"
    )

    # Print SVC variants
    for svc_method in ["SVC_296", "SVC_435"]:
        svc_results = aggregated_results[svc_method]
        print(
            f"{svc_method:20s} - Val: {svc_results['val_mean']:.4f}±{svc_results['val_std']:.4f}, "
            f"Test: {svc_results['test_mean']:.4f}±{svc_results['test_std']:.4f} "
            f"(SV: {svc_results['support_vectors_mean']:.1f}±{svc_results['support_vectors_std']:.1f})"
        )

    # Print MLP variants
    hidden_dims = [0, 48, 96, 144, 192]
    for hidden_dim in hidden_dims:
        mlp_method = f"MLP-{hidden_dim}"
        mlp_results = aggregated_results[mlp_method]
        print(
            f"{mlp_method:20s} - Val: {mlp_results['val_mean']:.4f}±{mlp_results['val_std']:.4f}, "
            f"Test: {mlp_results['test_mean']:.4f}±{mlp_results['test_std']:.4f}"
        )

    print("\nQuantum Methods (Mean ± Std, best per mode count):")
    quantum_results = aggregated_results["Qlayer"]
    quantum_configs = list(quantum_results.keys())
    modes_processed = set()

    for config in quantum_configs:
        mode_count = config.split("-")[0]
        if mode_count not in modes_processed:
            # Find best mean accuracy for this mode count
            mode_configs = [
                c for c in quantum_configs if c.startswith(mode_count + "-")
            ]
            best_config = max(
                mode_configs, key=lambda c: quantum_results[c]["val_mean"]
            )
            best_results = quantum_results[best_config]

            print(
                f"{mode_count + ' modes':20s} - Val: {best_results['val_mean']:.4f}±{best_results['val_std']:.4f}, "
                f"Test: {best_results['test_mean']:.4f}±{best_results['test_std']:.4f} "
                f"(Config: {best_config})"
            )
            modes_processed.add(mode_count)


def print_results_summary(results):
    """Print summary of all results"""
    print("\n=== RESULTS SUMMARY ===")
    print("\nClassical Methods:")

    # Print LogisticRegression
    print(
        f"{'LogisticRegression':20s} - Val: {results['LogisticRegression'][0]:.4f}, Test: {results['LogisticRegression'][1]:.4f}"
    )

    # Print SVC variants with parameter counts
    print(
        f"{'SVC_296':20s} - Val: {results['SVC_296'][0]:.4f}, Test: {results['SVC_296'][1]:.4f} (Support vectors: {results['SVC_296'][2]})"
    )
    print(
        f"{'SVC_435':20s} - Val: {results['SVC_435'][0]:.4f}, Test: {results['SVC_435'][1]:.4f} (Support vectors: {results['SVC_435'][2]})"
    )

    # Print MLP
    hidden_dims = [0, 48, 96, 144, 192]
    for hidden_dim in hidden_dims:
        print(
            f"{f'MLP-{hidden_dim}':20s} - Val: {results[f'MLP-{hidden_dim}'][0]:.4f}, Test: {results[f'MLP-{hidden_dim}'][1]:.4f}"
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
    trained_body = False
    trained_cl = True
    trained_q = True
    use_normalization_cl = False
    use_normalization_q = False
    # Create results folder for this experiment
    results_folder = create_results_folder()
    save_arguments(args, results_folder)

    if args.verbose:
        print(f"Results will be saved to: {results_folder}")

    if args.use_kfold:
        # K-fold cross validation mode
        if args.verbose:
            print(f"\nRunning {args.k_folds}-fold cross validation...")

        # Load data for k-fold
        label_0_samples, label_1_samples, trainval_samples_per_label = load_data_kfold(
            args
        )

        # Initialize results aggregation
        all_fold_results = []

        for fold_idx in range(args.k_folds):
            # Create fresh data split for this fold
            train_dataset, eval_dataset, test_dataset, features, labels = (
                create_fold_data_splits(
                    label_0_samples,
                    label_1_samples,
                    trainval_samples_per_label,
                    fold_idx,
                    args,
                )
            )
            if args.verbose:
                print(f"\n{'=' * 50}")
                print(f"FOLD {fold_idx + 1}/{args.k_folds}")
                print(f"{'=' * 50}")

            # Load model for this fold
            model, sentence_transformer = load_model(args, device)

            if trained_body:
                # Train body with contrastive learning
                sentence_transformer = train_body_with_contrastive_learning(
                    sentence_transformer, features, labels, args, device
                )

            # Generate embeddings for this fold
            (
                train_embeddings_not_scaled,
                train_embeddings,
                train_labels,
                global_train_max,
                global_train_min,
            ) = generate_embeddings(sentence_transformer, train_dataset, args)

            fold_results = {}
            num_classes = 2

            # Train classical heads for this fold
            if trained_cl:
                train_embeddings_cl = (
                    train_embeddings
                    if use_normalization_cl
                    else train_embeddings_not_scaled
                )
                fold_results, model, num_classes = train_classical_heads(
                    sentence_transformer,
                    train_embeddings_cl,
                    train_labels,
                    global_train_max,
                    global_train_min,
                    eval_dataset,
                    test_dataset,
                    args,
                    device,
                    use_normalization_cl,
                )
            if trained_q:
                # Train quantum heads for this fold
                train_embeddings_q = (
                    train_embeddings
                    if use_normalization_q
                    else train_embeddings_not_scaled
                )
                quantum_results = train_quantum_heads(
                    model,
                    sentence_transformer,
                    train_embeddings_q,
                    global_train_max,
                    global_train_min,
                    train_labels,
                    eval_dataset,
                    test_dataset,
                    args,
                    num_classes,
                    device,
                    results_folder,
                    use_normalization_q,
                )

                fold_results["Qlayer"] = quantum_results
            fold_results["fold"] = fold_idx + 1
            all_fold_results.append(fold_results)

            if args.verbose:
                print(f"\nFold {fold_idx + 1} completed!")

        # Aggregate results across folds
        aggregated_results = aggregate_kfold_results(all_fold_results, args)

        # Save aggregated results
        aggregated_results.update(
            {
                "k_folds": args.k_folds,
                "training_samples": 256,  # per label as per paper
                "epochs": args.body_epochs,
                "lr": args.learning_rate,
                "all_fold_results": all_fold_results,
            }
        )

        save_results(aggregated_results, results_folder)

        # Print k-fold summary
        print_kfold_summary(aggregated_results)

    else:
        # Original single-split mode
        # Load data and model
        train_dataset, eval_dataset, test_dataset, features, labels = load_data(args)
        model, sentence_transformer = load_model(args, device)

        if trained_body:
            # Train body with contrastive learning
            sentence_transformer = train_body_with_contrastive_learning(
                sentence_transformer, features, labels, args, device
            )

        # Generate embeddings
        (
            train_embeddings_not_scaled,
            train_embeddings,
            train_labels,
            global_train_max,
            global_train_min,
        ) = generate_embeddings(sentence_transformer, train_dataset, args)

        num_classes = 2
        results = {}  # Initialize results dictionary

        # Train classical heads
        if trained_cl:
            results, model, num_classes = train_classical_heads(
                sentence_transformer,
                train_embeddings,
                train_labels,
                global_train_max,
                global_train_min,
                eval_dataset,
                test_dataset,
                args,
                device,
            )

        # Train quantum heads
        if trained_q:
            quantum_results = train_quantum_heads(
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

        # print_results_summary(results)

    if args.verbose:
        print(f"\nExperiment completed! Results saved to: {results_folder}")


if __name__ == "__main__":
    main()
