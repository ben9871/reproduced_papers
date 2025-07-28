import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os

# Data from the table
models = [
    'VQC [1, 0, 0]',
    'VQC [1, 1, 1]',
    'MLP wide (4)',
    'MLP wide (5)',
    'MLP wide (6)',
    'MLP deep (2, 2)',
    'MLP deep (2, 3)',
    'MLP deep (3, 2)',
    'MLP deep (3, 3)',
    'SVM linear'
]

parameters = [16, 23, 17, 21, 25, 15, 19, 20, 25, 3]
accuracy_mean_lin = [0.875, 0.885, 0.890, 0.900, 0.892, 0.892, 0.890, 0.912, 0.892, 0.875]
accuracy_std_lin = [0.000, 0.017, 0.020, 0.022, 0.022, 0.019, 0.017, 0.020, 0.022, 0.000]
accuracy_mean_cir = [0.897, 0.827, 0.880, 0.915, 0.925, 0.622, 0.665, 0.722, 0.730, 0.375]
accuracy_std_cir = [0.096, 0.100, 0.103, 0.056, 0.052, 0.083, 0.087, 0.103, 0.090, 0.000]
accuracy_mean_moon = [0.802, 0.922, 0.982, 0.952, 0.982, 0.805, 0.805, 0.875, 0.897, 0.800]
accuracy_std_moon = [0.007, 0.036, 0.022, 0.054, 0.022, 0.029, 0.033, 0.067, 0.072, 0.000]

datasets_str = ['Linear', 'Circular', 'Moon']

def visu_accuracies_param(models, parameters, accuracy_mean, accuracy_std, dataset_str):
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Accuracy with error bars
    x_pos = np.arange(len(models))
    bars1 = ax1.bar(x_pos, accuracy_mean, yerr=accuracy_std, capsize=5,
                    alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_ylabel('Mean Test Accuracy', fontsize=12)
    ax1.set_title(f'Mean Final Test Accuracy (10 runs) on {dataset_str} Dataset', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0.2, 1)

    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars1, accuracy_mean, accuracy_std)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height - std_val - 0.1,
                 f'{mean_val:.2f}±{std_val:.2f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Number of parameters
    bars2 = ax2.bar(x_pos, parameters, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_ylabel('Number of Parameters', fontsize=12)
    ax2.set_title('Model Complexity (Number of Parameters)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, param_val in zip(bars2, parameters):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height - 1.2,
                 f'{param_val}', ha='center', va='bottom', fontsize=9)

    # Adjust layout and show
    plt.tight_layout()
    plt.savefig(f'./results/accuracies_comparison_{dataset_str}.png')

    # Optional: Create a combined plot showing both metrics
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create a scatter plot with accuracy vs parameters
    ax.scatter(parameters, accuracy_mean, s=100, alpha=0.7, c='purple')
    ax.errorbar(parameters, accuracy_mean, yerr=accuracy_std, fmt='none',
                capsize=5, color='purple', alpha=0.7)

    # Add labels for each point
    for i, model in enumerate(models):
        ax.annotate(model, (parameters[i], accuracy_mean[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, rotation=15, ha='left')

    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Mean Test Accuracy', fontsize=12)
    ax.set_title(f'Model Performance vs Complexity on {dataset_str} Dataset', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'./results/performance_vs_complexity_comparison_{dataset_str}.png')
    return

# visu_accuracies_param(models, parameters, accuracy_mean_lin, accuracy_std_lin, datasets_str[0])
# visu_accuracies_param(models, parameters, accuracy_mean_cir, accuracy_std_cir, datasets_str[1])
# visu_accuracies_param(models, parameters, accuracy_mean_moon, accuracy_std_moon, datasets_str[2])

def regroup_figures():
    """
    Regroup 6 decision boundary figures into a single figure with 2 columns and 3 rows.
    Columns represent different input Fock states, rows represent different datasets.
    """
    # Define the parameters
    input_fock_states = ['[1, 0, 0]', '[1, 1, 1]']
    datasets = ['linear', 'circular', 'moon']

    # Create figure and subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))

    # Set the main title
    fig.suptitle('Decision boundary of VQC in function of dataset and input Fock state',
                 fontsize=30, fontweight='bold', y=0.98)

    # Add column headers (input Fock states)
    for j, fock_state in enumerate(input_fock_states):
        axes[0, j].set_title(fock_state, fontsize=25, fontweight='bold', pad=20)

    # Load and display images
    for i, dataset in enumerate(datasets):
        for j, fock_state in enumerate(input_fock_states):
            # Construct the file path
            filename = f'decision_boundary_vqc_{fock_state}_{dataset}.png'
            filepath = os.path.join('./results', filename)

            # Check if file exists
            if os.path.exists(filepath):
                # Load and display the image
                img = mpimg.imread(filepath)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')  # Remove axes
            else:
                # Display error message if file doesn't exist
                axes[i, j].text(0.5, 0.5, f'File not found:\n{filename}',
                                ha='center', va='center', transform=axes[i, j].transAxes,
                                fontsize=10, color='red')
                axes[i, j].axis('off')

    # Adjust layout to make figures tightly placed
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title

    # Adjust spacing between subplots
    # Alternative: Even tighter spacing
    plt.subplots_adjust(hspace=-0.1, wspace=0.0, top=0.90, bottom=0.02, left=0.02, right=0.98)

    # Save the combined figure
    output_path = './results/combined_decision_boundaries.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    print(f"Combined figure saved as: {output_path}")
    return

regroup_figures()