import torch
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader

from KrylovSubspaceLibrary.optimizers import KrylovCG
from torch.nn.functional import cross_entropy

# Simple MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
# Load MNIST (or FashionMNIST for variety)
transform - transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

model = MLP()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss_fn = cross_entropy

# Get a batch for Krylov optimizer (this version is full-batch)
inputs, targets = next(iter(train_loader))
inputs, targets = inputs.to(device), targets.to(device)

# Attach Krylov optimizer
optimizer = KrylovCG(model, loss_fn, inputs, targets, lr=0.5, tol = 1e-4, max_iter=10)

# Run training for a few epochs
for epoch in range(3):
    model.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item(): .4f}")