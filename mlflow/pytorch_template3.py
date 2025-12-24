"""
Template script for training a PyTorch model with MLflow Manual Logging and system
metrics enabled. Track hardware resource utilization during training to monitor GPU
usage, memory consumption, and system performance.

Source: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/#system-metrics-tracking
"""

# Initial imports
import mlflow
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

# Set the MLflow
mlflow.set_tracking_uri("sqlite:////statapp/mlflow/database.db")
mlflow.set_experiment("Pytorch System Metrics")

# Create data and model
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
train_loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Enable system metrics logging
mlflow.enable_system_metrics_logging()

with mlflow.start_run():
    mlflow.log_params({"learning_rate": 0.001, "batch_size": 32, "epochs": 10})

    # Training loop - system metrics logged automatically
    for epoch in range(10):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        mlflow.log_metric("loss", loss.item(), step=epoch)

    mlflow.pytorch.log_model(model, name="model")
