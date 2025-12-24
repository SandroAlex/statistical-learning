"""
PyTorch iris classifier with MLflow checkpointing and metric logging. Track model
checkpoints during training with MLflow 3's checkpoint versioning. Use the step
parameter to version checkpoints and link metrics to specific model versions.

Source: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/#checkpoint-tracking
"""

# Initial imports.
from typing import Tuple

import mlflow
import torch

import torch.nn as nn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Set the MLflow
mlflow.set_tracking_uri("sqlite:////statapp/mlflow/database.db")
mlflow.set_experiment("Pytorch Checkpoints")
mlflow.enable_system_metrics_logging()


# Helper function to prepare data
def prepare_data(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:

    X: torch.Tensor = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y: torch.Tensor = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    return X, y


# Helper function to compute accuracy
def compute_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:

    with torch.no_grad():

        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).sum().item() / y.size(0)

    return accuracy


# Define a basic PyTorch classifier
class IrisClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# Load Iris dataset and prepare the DataFrame
iris = load_iris()
iris_df: pd.DataFrame = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

# Split into training and testing datasets
train_df, test_df = train_test_split(iris_df, test_size=0.2, random_state=42)

# Prepare training data
train_dataset = mlflow.data.from_pandas(train_df, name="iris_train")
X_train, y_train = prepare_data(train_dataset.df)

# Initialize model
input_size = X_train.shape[1]
hidden_size = 16
output_size = len(iris.target_names)
model = IrisClassifier(input_size, hidden_size, output_size)

# Training configuration
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Run training with MLflow checkpointing
with mlflow.start_run() as run:

    # Log parameters once at the start
    mlflow.log_params(
        {
            "n_layers": 3,
            "activation": "ReLU",
            "criterion": "CrossEntropyLoss",
            "optimizer": "Adam",
            "learning_rate": 0.01,
        }
    )

    for epoch in range(101):

        # Training step
        out = model(X_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log a checkpoint every 10 epochs
        if epoch % 10 == 0:

            # Log model checkpoint with step parameter
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name=f"iris-checkpoint-{epoch}",
                step=epoch,
                input_example=X_train[:5].numpy(),
            )

            # Log metrics linked to this checkpoint and dataset
            accuracy = compute_accuracy(model, X_train, y_train)

            mlflow.log_metric(
                key="train_accuracy",
                value=accuracy,
                step=epoch,
                model_id=model_info.model_id,
                dataset=train_dataset,
            )

# Search and rank checkpoints by performance
ranked_checkpoints = mlflow.search_logged_models(
    filter_string=f"source_run_id='{run.info.run_id}'",
    order_by=[{"field_name": "metrics.train_accuracy", "ascending": False}],
    output_format="list",
)

best_checkpoint = ranked_checkpoints[0]

print(f"Best checkpoint: {best_checkpoint.name}")
print(f"Accuracy: {best_checkpoint.metrics[0].value}")

print("Loading the best checkpoint for inference ...")

# Load as PyTorch model
model_uri: str = best_checkpoint.model_uri
loaded_model = mlflow.pytorch.load_model(model_uri)

# Make predictions
input_tensor = torch.randn(5, 4)
predictions = loaded_model(input_tensor)

# Diagnostic output
print(f"   Model URI: {model_uri}")
print(f"   Model loaded successfully:")
print(f"{loaded_model}")
print(f"")
print(f"   Sample predictions:")
print(f"   Input shape: {input_tensor.shape}")
print(f"   Output shape: {predictions.shape}")
