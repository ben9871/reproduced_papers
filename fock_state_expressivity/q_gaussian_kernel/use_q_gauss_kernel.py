from data import get_data_x, target_function
from hyperparameters import Hyperparams
from training import train_classif, visualize_accuracies

# Training set-up
num_photons = [2, 4, 6, 8, 10]
colors = ["blue", "orange", "green", "red", "purple"]
std_names = ["std = 1.0", "std = 0.5", "std = 0.33", "std = 0.25"]
x, x_on_pi, delta = get_data_x()
ys = [target_function(delta, sigma=float(sigma[-4:])) for sigma in std_names]
# visu_target_functions(x_on_pi, ys)
ys_info = {"ys": ys, "names": std_names}

# Hyperparameters
num_runs = 5
num_epochs = 200
batch_size = 32
lr = 0.02
betas = [0.7, 0.9]
weight_decay = 0.0
train_circuit = True
scale_type = "learned"  # ["learned", "1", "pi", "2pi", "/pi", "/2pi", "0.1"]
circuit = "general"  # ["mzi", "general", "spiral", "general_all_angles", "ps_based", "bs_based"]
no_bunching = False
optimizer = "adam"  # ["adagrad", "adam", "adamw", "sgd"]
shuffle_train = True

args = Hyperparams(
    num_runs,
    num_epochs,
    batch_size,
    lr,
    betas,
    weight_decay,
    train_circuit,
    scale_type,
    circuit,
    no_bunching,
    optimizer,
    shuffle_train,
)
scaling_factor = 0.65

# q_time calculates 20 svc
q_accs, q_time = train_classif(
    args, scaling_factor, ys_info, num_photons, type="quantum"
)
# class_time calculates 4 svc
accs, class_time = train_classif(args, scaling_factor, ys_info, num_photons, type="rbf")

print(f"Time for calculating 20 quantum SVC: {q_time}")
print(f"Time for calculating 4 rbf SVC: {class_time}")

visualize_accuracies(q_accs, accs)
