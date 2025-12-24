"""
Template script for training a PyTorch model with MLflow Manual Logging.

Source: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/#manual-logging
"""

# Imports
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Set the MLflow
mlflow.set_tracking_uri("sqlite:////statapp/mlflow/database.db")
mlflow.set_experiment("Pytorch Manual Logging")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):

        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


# Create synthetic data
X = torch.randn(1000, 784)  # 28 x 28 images flattened
y = torch.randint(0, 10, (1000,))
train_loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# Training parameters
params = {
    "epochs": 23,
    "learning_rate": 1e-3,
    "batch_size": 64,
}

# Training with MLflow manual logging
with mlflow.start_run():

    # Log parameters
    mlflow.log_params(params)

    # Initialize model and optimizer
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])

    # Training loop
    for epoch in range(params["epochs"]):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        # Log metrics per epoch
        avg_loss = train_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        mlflow.log_metrics(
            {"train_loss": avg_loss, "train_accuracy": accuracy}, step=epoch
        )

    # Log final model
    mlflow.pytorch.log_model(model, name="model")
