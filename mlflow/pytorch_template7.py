"""
Template script to log a PyTorch model to MLflow Model Registry with tags and alias.
Register PyTorch models for version control and deployment.

Source: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/#model-registry-integration
"""

# Initial imports
import mlflow
import torch.nn as nn

from mlflow import MlflowClient

# Set the MLflow
mlflow.set_tracking_uri("sqlite:////statapp/mlflow/database.db")
mlflow.set_experiment("Pytorch Model Registry")
mlflow.enable_system_metrics_logging()
client = MlflowClient()

with mlflow.start_run():

    # Create a simple model for demonstration
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 15 * 15, 10),
    )

    # Log model to registry
    model_info = mlflow.pytorch.log_model(
        model, name="pytorch_model", registered_model_name="ImageClassifier"
    )

    # Tag for tracking
    mlflow.set_tags(
        {"model_type": "cnn", "dataset": "imagenet", "framework": "pytorch"}
    )

# Set alias for production deployment
client.set_registered_model_alias(
    name="ImageClassifier",
    alias="champion",
    version=model_info.registered_model_version,
)
