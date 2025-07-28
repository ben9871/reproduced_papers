from training import train_vqc_multiple_runs, visualize_results, write_summary_statistics

# Hyperparameters and experimental set-up
m = 3
input_size = 2
initial_state = [1, 1, 1]

activation = "none"  # ["none", "sigmoid", "softmax"]
no_bunching = False
num_runs = 10
n_epochs = 150
batch_size = 30
lr = 0.02
alpha = 0.0002
betas = (0.8, 0.999)
circuit = "bs_mesh"  # ["bs_mesh", "general", "bs_basic", "spiral"]
scale_type = "/pi"  # ["learned", "2pi", "pi", "1", "/pi", "/2pi", "0.1", "0.5"]
regu_on = "linear"  # ["linear", "all"]

log_wandb = True

class Arguments:
    def __init__(self, m, input_size, initial_state, activation="none", no_bunching=False, num_runs=5, n_epochs=50,
                 batch_size=32, lr=0.02, alpha=0.2, betas=(0.9, 0.999), circuit="bs_mesh", scale_type="learned",
                 regu_on="linear", log_wandb=True):
        self.m = m
        self.input_size = input_size
        self.initial_state = initial_state
        self.activation = activation
        self.no_bunching = no_bunching
        self.num_runs = num_runs
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.betas = betas
        self.circuit = circuit
        self.scale_type = scale_type
        self.regu_on = regu_on
        self.dataset_name = ""
        self.model_type = ""
        self.log_wandb = log_wandb

    def set_dataset_name(self, dataset):
        self.dataset_name = dataset
        return

    def set_model_type(self, type):
        self.model_type = type
        return

args = Arguments(m, input_size, initial_state, activation, no_bunching, num_runs, n_epochs, batch_size, lr, alpha,
                 betas, circuit, scale_type, regu_on, log_wandb)

# Train VQC multiple times and show results
results = train_vqc_multiple_runs(args)
visualize_results(results)
write_summary_statistics(results, args)

# For comparison with classical models

# MLP
"""network_type = "wide"  # ["wide", "deep"]
results = train_mlp_multiple_runs(args, network_type=network_type)
visualize_results(results)
write_summary_statistics(results, args)"""

"""network_type = "deep"  # ["wide", "deep"]
results = train_mlp_multiple_runs(args, network_type=network_type)
visualize_results(results)
write_summary_statistics(results, args)"""

# SVM
"""kernel_type = "lin"  # ["lin", "rbf"]
results = train_svm_multiple_runs(args, kernel_type=kernel_type)
visualize_results(results)
write_summary_statistics(results, args)"""

