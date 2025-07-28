import random
import time

from training import train_vqc_multiple_runs


class Arguments:
    def __init__(
        self,
        m,
        input_size,
        initial_state,
        activation="none",
        no_bunching=False,
        num_runs=5,
        n_epochs=50,
        batch_size=32,
        lr=0.02,
        alpha=0.2,
        betas=(0.9, 0.999),
        circuit="bs_mesh",
        scale_type="learned",
        regu_on="linear",
    ):
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

    def set_dataset_name(self, dataset):
        self.dataset_name = dataset
        return

    def set_model_type(self, type):
        self.model_type = type
        return


# method: random search
num_hp_runs = 200

# metric:
#  name: Mean final test accuracy for all 3 datasets
#  goal: maximize

# HPs
learning_rates = [0.1, 0.02, 0.002]
batch_sizes = [15, 30, 60]
n_epochs = [75, 100, 125]
activations = ["none", "sigmoid", "softmax"]
alphas = [0.2, 0.02, 0.002, 0.0002]
betas = [
    (0.7, 0.9),
    (0.8, 0.99),
    (0.9, 0.999),
    (0.95, 0.9999),
    (0.8, 0.999),
    (0.9, 0.99),
]
circuits = ["bs_mesh", "general", "bs_basic", "spiral"]
scale_types = ["/pi", "/2pi", "0.1", "0.5", "learned"]
regus_on = ["linear", "all"]

# Other HPs
m = 3
input_size = 2
initial_state = [1, 1, 1]

no_bunching = False
num_runs = 3

n = 0
total_time = 0
while n < num_hp_runs:
    time1 = time.time()

    learning_rate = learning_rates[random.randint(0, len(learning_rates) - 1)]
    batch_size = batch_sizes[random.randint(0, len(batch_sizes) - 1)]
    n_epoch = n_epochs[random.randint(0, len(n_epochs) - 1)]
    activation = activations[random.randint(0, len(activations) - 1)]
    alpha = alphas[random.randint(0, len(alphas) - 1)]
    beta = betas[random.randint(0, len(betas) - 1)]
    circuit = circuits[random.randint(0, len(circuits) - 1)]
    scale = scale_types[random.randint(0, len(scale_types) - 1)]
    regu_on = regus_on[random.randint(0, len(regus_on) - 1)]

    args = Arguments(
        m,
        input_size,
        initial_state,
        activation,
        no_bunching,
        num_runs,
        n_epoch,
        batch_size,
        learning_rate,
        alpha,
        beta,
        circuit,
        scale,
        regu_on,
    )

    s = "----- Hyperparameters information for this run -----\n"
    s += f"m = {args.m}\ninput_size = {args.input_size}\ninitial_state = {args.initial_state}\n\n"
    s += (
        f"activation = {args.activation}\nno_bunching = {args.no_bunching}\nnum_runs = {args.num_runs}\nn_epochs = "
        f"{args.n_epochs}\nbatch_size = {args.batch_size}\nlr = {args.lr}\nalpha = {args.alpha}\nbetas = {args.betas}"
        f"\ncircuit = {args.circuit}\nscale_type = {args.scale_type}\nregu_on = {args.regu_on}\n"
    )
    print(s)

    # Train VQC multiple times and show results
    results = train_vqc_multiple_runs(args)
    n += 1

    time2 = time.time()
    run_time = time2 - time1
    total_time += run_time
    estimated_time_left = (total_time / n) * (num_hp_runs - n)
    print(
        f"\n\n\n{n} / {num_hp_runs} completed. Run time: {run_time}, Total time: {total_time}, Estimated time left: {estimated_time_left} ######################################################################\n\n\n"
    )
