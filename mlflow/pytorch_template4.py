"""
Template script for training a PyTorch model with MLflow Manual Logging and model
signatures. Ensure that the input and output schema of the model is logged for better
model management and deployment.

Source: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/#model-logging-with-signatures
"""

# Initial imports
import mlflow
import torch
import torch.nn as nn

from mlflow.models import infer_signature

# Set the MLflow
mlflow.set_tracking_uri("sqlite:////statapp/mlflow/database.db")
mlflow.set_experiment("Pytorch Signatures")

# Create a simple model
model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))

# Create sample input and output for signature
input_example = torch.randn(1, 10)
predictions = model(input_example)

# Infer signature from input / output
signature = infer_signature(input_example.numpy(), predictions.detach().numpy())

with mlflow.start_run():

    # Log model with signature and input example
    mlflow.pytorch.log_model(
        model,
        name="pytorch_model",
        signature=signature,
        input_example=input_example.numpy(),
    )
