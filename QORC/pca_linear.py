import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

batch_size = 100
k = 20
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def extract_flattened_data(loader):
    all_data, all_labels = [], []
    for x, y in loader:
        x = x.view(x.size(0), -1)  # Flatten images
        all_data.append(x)
        all_labels.append(y)
    return torch.cat(all_data), torch.cat(all_labels)


X_train, y_train = extract_flattened_data(train_loader)
X_test, y_test = extract_flattened_data(test_loader)

# Centrage
X_mean = X_train.mean(dim=0)
X_train_centered = X_train - X_mean
X_test_centered = X_test - X_mean  # centré avec la moyenne d'entraînement

# PCA
U, S, V = torch.pca_lowrank(X_train_centered, q=k)
X_train_pca = torch.matmul(X_train_centered, V[:, :k])
X_test_pca = torch.matmul(X_test_centered, V[:, :k])


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


model = LinearClassifier(k, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# Conversion des données PCA en DataLoader
train_tensor = torch.utils.data.TensorDataset(X_train_pca, y_train)
test_tensor = torch.utils.data.TensorDataset(X_test_pca, y_test)

train_pca_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
test_pca_loader = DataLoader(test_tensor, batch_size=batch_size)

# --- Training ---
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_pca_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# --- Test ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x_batch, y_batch in test_pca_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
