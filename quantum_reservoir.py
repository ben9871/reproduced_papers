import numpy as np
import torch
import torch.nn as nn
from merlin import QuantumLayer, OutputMappingStrategy
import perceval as pcvl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from perceval.components import BS, PS
M = 20
N = 3

def create_circuit(seed, M):
    np.random.seed(seed)
    circuit = pcvl.Circuit(M)
    def layer_bs(circuit, k, M):
        for i in range(k, M-1, 2):
            theta = pcvl.P(f"theta_{i}")
            circuit.add(i, BS(theta=np.random.rand() * np.pi))

    layer_bs(circuit, 0, M)
    layer_bs(circuit, 1, M)
    layer_bs(circuit, 0, M)
    layer_bs(circuit, 1, M)
    layer_bs(circuit, 0, M)
    for i in range(M):
        phi = pcvl.P(f"phi_{i}")
        circuit.add(i, PS(phi))
    layer_bs(circuit, 0, M)
    layer_bs(circuit, 1, M)
    layer_bs(circuit, 0, M)
    layer_bs(circuit, 1, M)
    layer_bs(circuit, 0, M)
    return circuit

def standardize_to_pi_range(components, min_vals, max_vals):
    # Scale to [0, 1]
    scaled_components = (components - min_vals) / (max_vals - min_vals)
    # Scale to [0, π]
    standardized_components = scaled_components * torch.pi
    return standardized_components


batch_size = 100

num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 28*28

transform = transforms.ToTensor()
train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=60000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=20000, shuffle=False)
train_images, train_labels = next(iter(train_loader))  # one batch with all training images
x_train = train_images.view(train_images.size(0), -1).numpy()  # flatten images to 784-dim
y_train = train_labels.to(device).view(-1, batch_size)
print(x_train.shape)


test_images, test_labels = next(iter(test_loader))
x_test = test_images.view(test_images.size(0), -1).numpy()  # flatten images to 784-dim
y_test = test_labels.to(device).view(-1, batch_size)
print(x_test.shape)

# Apply PCA
pca = PCA(n_components=M)  # Reduce to 2D for visualization
x_train_pca = pca.fit_transform(x_train).reshape(-1, batch_size, M)
x_test_pca = pca.transform(x_test).reshape(-1, batch_size, M)
print(x_train_pca.shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_train = torch.tensor(x_train).view(-1, batch_size, 784).to(device)
x_test = torch.tensor(x_test).view(-1, batch_size, 784).to(device)

min_vals = torch.tensor(x_train_pca.min(axis=0)).to(device)
max_vals = torch.tensor(x_test_pca.max(axis=0)).to(device)

dropout_list = [0.02, 0.04, 0.06, 0.08, 0.1]

for dropout in dropout_list:
    seed = 41
    M = 20
    N = 3
    circuit = create_circuit(seed, M)
    a = (M - N)//2
    b = M - N - a
    pca = PCA(n_components=M)  # Reduce to 2D for visualization
    x_train_pca = pca.fit_transform(x_train.view(-1, 784).detach().cpu().numpy()).reshape(-1, batch_size, M)
    x_test_pca = pca.transform(x_test.view(-1, 784).detach().cpu().numpy()).reshape(-1, batch_size, M)
    print(x_train_pca.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_train = torch.tensor(x_train).view(-1, batch_size, 784).to(device)
    x_test = torch.tensor(x_test).view(-1, batch_size, 784).to(device)

    min_vals = torch.tensor(x_train_pca.min(axis=0)).to(device)
    max_vals = torch.tensor(x_test_pca.max(axis=0)).to(device)

    input_state = [1] + [0] * a + [1] + [0] * b +[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qlayer = QuantumLayer(
                input_size=M,
                output_size=None,
                circuit=circuit,
                n_photons=N,
                input_state=input_state,# Random Initial quantum state used only for initialization
                output_mapping_strategy=OutputMappingStrategy.NONE,
                input_parameters=["phi"],# Optional: Specify device
                shots=1000,  # Optional: Enable quantum measurement sampling
                no_bunching=True,
                sampling_method='multinomial', # Optional: Specify sampling method
            ).to(device)
    L = nn.Sequential(nn.BatchNorm1d(qlayer.output_size + img_size), nn.Dropout(p=dropout), nn.Linear(qlayer.output_size + img_size, 10)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(L.parameters(), lr=0.05)
    num_epochs = 100
    step = 1000
    def test():
        val_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for i in range(len(x_test)):
                output_bs = qlayer(standardize_to_pi_range(torch.tensor(x_test_pca[i]).to(device), min_vals, max_vals).to(torch.float32)).to(device)
                x = torch.cat((x_test[i], output_bs), dim=1).to(device)
                output = L(x.to(torch.float32))
                loss = criterion(output, y_test[i])
                val_loss += loss.item()
                predicted = torch.argmax(output, dim=1)
                total += batch_size
                correct += (predicted == y_test[i]).sum().item()

        print(f"Using seed: {seed}, Accuracy: {correct / total}")
    for epoch in range(num_epochs):
        for k in range(len(x_train_pca)):
            with torch.no_grad():
                output_bs = qlayer(standardize_to_pi_range(torch.tensor(x_train_pca[k]).to(device).to(torch.float32), min_vals, max_vals)).to(device)
            x = torch.cat((x_train[k], output_bs), dim=1).to(device)
            output = L(x.to(torch.float32))
            loss = criterion(output, y_train[k])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    L.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(len(x_test)):
            output_bs = qlayer(standardize_to_pi_range(torch.tensor(x_test_pca[i]).to(device), min_vals, max_vals).to(torch.float32)).to(device)
            x = torch.cat((x_test[i], output_bs), dim=1).to(device)
            output = L(x.to(torch.float32))
            loss = criterion(output, y_test[i])
            val_loss += loss.item()
            predicted = torch.argmax(output, dim=1)
            total += batch_size
            correct += (predicted == y_test[i]).sum().item()

    print(f"Using seed: {seed}, Accuracy: {correct / total}, total: {total}, {y_test.shape} ")

