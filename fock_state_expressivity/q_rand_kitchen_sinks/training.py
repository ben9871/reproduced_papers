import torch
import numpy as np
import matplotlib.pyplot as plt
from data import get_moon_dataset, split_train_test, scale_dataset, visualize_dataset, save_dataset, visualize_kernel, get_target_function
from approx_kernel import get_x_r_i_s, get_z_s_classically, get_approx_kernel_train, get_approx_kernel_predict, get_q_model
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import wandb

def get_optimizer(model, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas,
                                weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

def get_save_data(args):
    X, y = get_moon_dataset(args)
    X_train, X_test, y_train, y_test = split_train_test(X, y, args.test_prop, args.random_state)
    X_train, X_test = scale_dataset(X_train, X_test, args.scaling)
    visualize_dataset(X_train, X_test, y_train, y_test)
    save_dataset(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test

def training_q_model(X_train, X_test, args):
    # Transform data
    x_r_i_s_train_origin = get_x_r_i_s(X_train, args.w, args.b, args.r, args.gamma)
    x_r_i_s_test_origin = get_x_r_i_s(X_test, args.w, args.b, args.r, args.gamma)

    target_fit_train_origin = get_target_function(x_r_i_s_train_origin)
    target_fit_test_origin = get_target_function(x_r_i_s_test_origin)

    if args.hybrid_model_data == 'Default':
        x_r_i_s_train = x_r_i_s_train_origin
        x_r_i_s_test = x_r_i_s_test_origin
    elif args.hybrid_model_data == 'Generated':
        train_mins = x_r_i_s_train_origin.min(axis=0)  # shape (r,)
        train_maxs = x_r_i_s_train_origin.max(axis=0)  # shape (r,)
        test_mins = x_r_i_s_test_origin.min(axis=0)  # shape (r,)
        test_maxs = x_r_i_s_test_origin.max(axis=0)  # shape (r,)

        x_r_i_s_train = np.linspace(train_mins, train_maxs, 540, axis=0)
        x_r_i_s_test = np.linspace(test_mins, test_maxs, 100, axis=0)
    else:
        raise ValueError(f"Unknown hybrid_model_data: {args.hybrid_model_data}")

    target_fit_train = get_target_function(x_r_i_s_train)
    target_fit_test = get_target_function(x_r_i_s_test)

    assert x_r_i_s_train.shape == target_fit_train.shape, f'Target fit shape is wrong for x_r_i_s_train: {target_fit_train.shape}'
    assert x_r_i_s_train.shape == target_fit_train.shape, f'Target fit shape is wrong for x_r_i_s_test: {target_fit_test.shape}'

    q_model = get_q_model(args)
    print(q_model)
    # Count only trainable parameters
    trainable_params = sum(p.numel() for p in q_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    dataset = TensorDataset(torch.Tensor(x_r_i_s_train), torch.Tensor(target_fit_train))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    loss_f = torch.nn.MSELoss()
    optimizer = get_optimizer(q_model, args)
    epoch_bar = tqdm.tqdm(range(args.num_epochs), desc="Training Epochs")
    models = []
    losses = {'Train': [], 'Test': []}
    total_gradient = 0
    num_optimization_steps = 0

    for epoch in epoch_bar:
        q_model.train()
        total_loss = 0

        for x_batch, y_batch in dataloader:
            # Simpler to only consider batches of size batch_size
            if len(x_batch) < args.batch_size:
                continue

            num_optimization_steps += 1

            optimizer.zero_grad()
            # Reformat input
            x_batch = x_batch.view(args.batch_size * args.r, -1) * torch.tensor(args.pre_encoding_scaling)
            logits = q_model(x_batch)
            # Reformat output
            logits = logits.view(args.batch_size, args.r)
            loss = loss_f(logits, y_batch)
            loss.backward()

            for name, param in q_model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.norm().item()
                    wandb.log({f"grad_norm/{name}": grad})
                    total_gradient += grad

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses['Train'].append(avg_loss)
        epoch_bar.set_postfix({"Train Loss": avg_loss})

        #Eval
        q_model.eval()
        eval_input = torch.Tensor(x_r_i_s_test).view(len(x_r_i_s_test) * args.r, -1) * torch.tensor(args.pre_encoding_scaling)
        test_logits = q_model(eval_input)
        # Reformat
        test_logits = test_logits.view(len(x_r_i_s_test), args.r)
        test_loss = loss_f(test_logits, torch.Tensor(target_fit_test))
        epoch_bar.set_postfix({"Test Loss": test_loss})
        losses['Test'].append(test_loss.detach().numpy())

        models.append(q_model)

    best_test_mse = np.min(losses["Test"])
    best_test_mse_epoch = np.argmin(losses["Test"])
    print(f'Best test MSE: {best_test_mse:.3f} at epoch {best_test_mse_epoch}')
    wandb.log({"Best test MSE": best_test_mse})
    wandb.log({"Best test MSE epoch": best_test_mse_epoch})
    # We will keep and use the version of the q_model with the best test MSE
    q_model = models[best_test_mse_epoch]

    grad_norm_per_optimization_step = total_gradient / num_optimization_steps
    mean_grad_per_param = grad_norm_per_optimization_step / trainable_params
    print(f'Mean gradient per optimization step per parameter: {mean_grad_per_param:.6f}')

    return q_model, losses, x_r_i_s_train_origin, x_r_i_s_test_origin, target_fit_train_origin, target_fit_test_origin

def visualize_losses(losses):
    plt.figure(figsize=(7, 5))
    epochs = range(1, len(losses['Train']) + 1)

    plt.plot(epochs, losses['Train'], label='Train Loss', color='blue')
    plt.plot(epochs, losses['Test'], label='Test Loss', color='red')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('./results/loss_curve.png')
    wandb.log({"Losses": wandb.Image(plt.gcf())})
    plt.close()
    return

def no_train_q_model(X_train, X_test, args):
    # Transform data
    x_r_i_s_train = get_x_r_i_s(X_train, args.w, args.b, args.r, args.gamma)
    x_r_i_s_test = get_x_r_i_s(X_test, args.w, args.b, args.r, args.gamma)

    target_fit_train = get_target_function(x_r_i_s_train)
    target_fit_test = get_target_function(x_r_i_s_test)

    assert x_r_i_s_train.shape == target_fit_train.shape, f'Target fit shape is wrong for x_r_i_s_train: {target_fit_train.shape}'
    assert x_r_i_s_train.shape == target_fit_train.shape, f'Target fit shape is wrong for x_r_i_s_test: {target_fit_test.shape}'

    q_model = get_q_model(args)

    return q_model, x_r_i_s_train, x_r_i_s_test, target_fit_train, target_fit_test

def q_rand_kitchen_sinks(X_train, X_test, y_train, args):
    if args.train_hybrid_model:
        q_model_opti, losses, x_r_i_s_train, x_r_i_s_test, target_fit_train, target_fit_test = training_q_model(X_train, X_test, args)
        visualize_losses(losses)
    else:
        q_model_opti, x_r_i_s_train, x_r_i_s_test, target_fit_train, target_fit_test = no_train_q_model(X_train, X_test, args)

    q_model_opti.eval()
    train_input = torch.Tensor(x_r_i_s_train).view(len(x_r_i_s_train) * args.r, -1) * torch.tensor(args.pre_encoding_scaling)
    test_input = torch.Tensor(x_r_i_s_test).view(len(x_r_i_s_test) * args.r, -1) * torch.tensor(args.pre_encoding_scaling)
    z_s_train = q_model_opti(train_input)
    z_s_test = q_model_opti(test_input)

    z_s_train = z_s_train.view(len(x_r_i_s_train), args.r)
    z_s_test = z_s_test.view(len(x_r_i_s_test), args.r)

    # In the paper, their multiply by 1/sqrt(R)
    z_s_train = z_s_train * args.z_q_matrix_scaling_value
    z_s_test = z_s_test * args.z_q_matrix_scaling_value

    kernel_matrix_training = get_approx_kernel_train(z_s_train.detach().numpy())
    kernel_matrix_test = get_approx_kernel_predict(z_s_test.detach().numpy(), z_s_train.detach().numpy())

    visualize_kernel(kernel_matrix_training, y_train, args, True)

    return q_model_opti, kernel_matrix_training, kernel_matrix_test

def classical_rand_kitchen_sinks(X_train, X_test, y_train, args):
    # Transform data
    x_r_i_s_train = get_x_r_i_s(X_train, args.w, args.b, args.r, args.gamma)
    x_r_i_s_test = get_x_r_i_s(X_test, args.w, args.b, args.r, args.gamma)

    z_s_train = get_z_s_classically(x_r_i_s_train)
    z_s_test = get_z_s_classically(x_r_i_s_test)

    kernel_matrix_training = get_approx_kernel_train(z_s_train)
    kernel_matrix_test = get_approx_kernel_predict(z_s_test, z_s_train)

    visualize_kernel(kernel_matrix_training, y_train, args, False)

    return kernel_matrix_training, kernel_matrix_test

def save_decision_boundary(svc, q_model_opti, X_train, X_test, y_train,
                           y_test, acc, incorrect, args):
    # Combine train and test for full visualization
    X_all = np.vstack((X_train, X_test))

    # Build a meshgrid over the 2D input space
    h = 0.02  # mesh step size
    x_min, x_max = X_all[:, 0].min() - 0.2, X_all[:, 0].max() + 0.2
    y_min, y_max = X_all[:, 1].min() - 0.2, X_all[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Flatten grid to get (n_points, 2) shape
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if q_model_opti is None:  # Classically compute the random kitchen sinks
        grid_r_i_s = get_x_r_i_s(grid_points, args.w, args.b, args.r, args.gamma)
        x_r_i_s_train = get_x_r_i_s(X_train, args.w, args.b, args.r, args.gamma)

        grid_z_s = get_z_s_classically(grid_r_i_s)
        z_s_train = get_z_s_classically(x_r_i_s_train)

        K_grid = get_approx_kernel_predict(grid_z_s, z_s_train)

        figure_name = f'classical_rand_kitchen_sinks_R_{args.r}_sigma_{1.0/args.gamma}.png'
        figure_title = 'Decision boundary of SVC with classical Random Kitchen Sinks'


    else:  # Quantumly approximate the random kitchen sinks
        grid_r_i_s = get_x_r_i_s(grid_points, args.w, args.b, args.r, args.gamma)
        x_r_i_s_train = get_x_r_i_s(X_train, args.w, args.b, args.r, args.gamma)

        grid_input = torch.Tensor(grid_r_i_s).view(len(grid_r_i_s) * args.r, -1) * args.pre_encoding_scaling
        train_input = torch.Tensor(x_r_i_s_train).view(len(x_r_i_s_train) * args.r, -1) * args.pre_encoding_scaling

        grid_z_s = q_model_opti(grid_input)
        z_s_train = q_model_opti(train_input)

        grid_z_s = grid_z_s.view(len(grid_r_i_s), args.r)
        z_s_train = z_s_train.view(len(x_r_i_s_train), args.r)

        # In the paper, their multiply by 1/sqrt(R)
        grid_z_s = grid_z_s * args.z_q_matrix_scaling_value
        z_s_train = z_s_train * args.z_q_matrix_scaling_value

        K_grid = get_approx_kernel_predict(grid_z_s.detach().numpy(), z_s_train.detach().numpy())

        figure_name = f'q_rand_kitchen_sinks_R_{args.r}_sigma_{1.0 / args.gamma}.png'
        figure_title = 'Decision boundary of SVC with quantum approximated Random Kitchen Sinks'

    # Predict on the kernelized grid
    Z = svc.decision_function(K_grid)
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])

    # Decision boundary
    plt.contourf(xx, yy, Z > 0, cmap=cmap_light, alpha=0.6)

    # Plot data points
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label="Class 0 - Train", marker='o')
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='red', label="Class 0 - Test", marker='x')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label="Class 1 - Train", marker='o')
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='blue', label="Class 1 - Test", marker='x')

    plt.scatter(X_test[incorrect][:, 0], X_test[incorrect][:, 1], color='black', label="Incorrectly predicted", marker='o', s=10)

    plt.text(0.05, 0.95, f"{acc:.3}",
             transform=plt.gca().transAxes,
             fontsize=20, fontweight='bold',
             verticalalignment='top')

    if args.gamma == 1:
        s = f'R = {args.r}\n$\\sigma = 1$'
    else:
        s = f'R = {args.r}\n$\\sigma = 1 / {args.gamma}$'
    plt.text(0.05, 0.05, s,
             transform=plt.gca().transAxes,
             fontsize=20, verticalalignment='bottom')

    plt.title(figure_title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./results/{figure_name}")
    wandb.log({"Decision boundary": wandb.Image(plt.gcf())})
    plt.close()
    return

def train_svm(kernel_matrix_training, kernel_matrix_test, q_model_opti, X_train, X_test, y_train, y_test, args):
    svc = SVC(C=args.C, kernel='precomputed', random_state=args.random_state)
    svc.fit(kernel_matrix_training, y_train)
    preds = svc.predict(kernel_matrix_test)
    acc = accuracy_score(y_test, preds)
    wandb.log({"Test classif accuracy": acc})
    incorrect = (y_test != preds)

    save_decision_boundary(svc, q_model_opti, X_train, X_test, y_train, y_test, acc, incorrect, args)
    return acc
