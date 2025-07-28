from hyperparameters import Hyperparams
from training import train_models_multiple_runs, visualize_learned_function, save_models
from data import get_data_x, target_function
import time

intial_time = time.time()

# Training set-up
num_photons = [2, 4, 6, 8, 10]
colors = ["blue", "orange", "green", "red", "purple"]
std_names = ["std = 1.00", "std = 0.50", "std = 0.33", "std = 0.25"]
x, x_on_pi, delta = get_data_x()
ys = [target_function(delta, sigma=float(sigma[-4:])) for sigma in std_names]
#visu_target_functions(x_on_pi, ys)
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

def run_q_gaussian_sampler_hp_search():
      # Hyperparameter search
      optimizers = ["adam", "adagrad"]
      betas_s = [[0.7, 0.9], [0.9, 0.999], [0.95, 0.9999]]
      weight_decays = [0.0, 0.02]
      lrs = [0.2, 0.02, 0.002]

      for optimizer in optimizers:
          for betas in betas_s:
              for weight_decay in weight_decays:
                  for lr in lrs:
                      args = Hyperparams(num_runs, num_epochs, batch_size, lr, betas, weight_decay, train_circuit, scale_type,
                                         circuit, no_bunching, optimizer, shuffle_train)

                      # Train all models multiple runs
                      all_results, unique_id = train_models_multiple_runs(num_photons, colors, delta, ys_info, args)
                      visualize_learned_function(all_results, num_photons, x_on_pi, delta, ys_info, args, unique_id)
      return

args = Hyperparams(num_runs, num_epochs, batch_size, lr, betas, weight_decay, train_circuit, scale_type, circuit, no_bunching, optimizer, shuffle_train)

# Train all models multiple runs
all_results, unique_id = train_models_multiple_runs(num_photons, colors, delta, ys_info, args)
visualize_learned_function(all_results, num_photons, x_on_pi, delta, ys_info, args, unique_id)
save_models(all_results, num_photons, ys_info)

final_time = time.time()
print(f"Total time to train {args.num_runs * len(num_photons) * len(std_names)} models and save "
      f"{len(num_photons) * len(std_names)} models: {final_time - intial_time}")

# Run hyperparameter search
"""run_q_gaussian_sampler_hp_search()"""