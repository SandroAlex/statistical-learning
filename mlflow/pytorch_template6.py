"""
Template for integrating PyTorch model training with MLflow and Optuna for
hyperparameter optimization. This example demonstrates how to set up an
Optuna study to optimize hyperparameters while logging experiments with MLflow.

Source: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/#hyperparameter-optimization
"""

# Imports
import mlflow
import optuna
import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

# Set the MLflow
mlflow.set_tracking_uri("sqlite:////statapp/mlflow/database.db")
mlflow.set_experiment("Pytorch Hyperparameter Optimization")
mlflow.enable_system_metrics_logging()


# Create synthetic dataset for demonstration
input_size = 784  # e.g., flattened 28x28 images
output_size = 10  # e.g., 10 classes

X_train = torch.randn(1000, input_size)
y_train = torch.randint(0, output_size, (1000,))
X_val = torch.randn(200, input_size)
y_val = torch.randint(0, output_size, (200,))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)


def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=5):
    """
    Simple training loop for demonstration.
    """

    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()

    return val_loss / len(val_loader)


def objective(trial):
    """
    Optuna objective for hyperparameter tuning.
    """

    with mlflow.start_run(nested=True):

        # Define hyperparameter search space
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "hidden_size": trial.suggest_int("hidden_size", 32, 512),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        }

        # Log parameters
        mlflow.log_params(params)

        # Create model
        model = nn.Sequential(
            nn.Linear(input_size, params["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["hidden_size"], output_size),
        )

        # Train model
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        val_loss = train_and_evaluate(model, optimizer, train_loader, val_loader)

        # Log validation loss
        mlflow.log_metric("val_loss", val_loss)

        return val_loss


# Run optimization
with mlflow.start_run(run_name="PyTorch HPO"):

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Log best parameters
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_val_loss", study.best_value)
