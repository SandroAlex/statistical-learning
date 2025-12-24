"""
Template script for training a scikit-learn model with MLflow Manual Logging.

Source: https://mlflow.org/docs/latest/ml/getting-started/quickstart/#step-5---log-a-model-and-metadata-manually
"""

# Initial imports
import mlflow

import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set the MLflow:
# (1) tracking URI;
# (2) experiment name;
mlflow.set_tracking_uri("sqlite:////statapp/mlflow/database.db")
mlflow.set_experiment("Sklearn Template Manual Logging Experiment")

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Start an MLflow run
with mlflow.start_run():

    # Log the hyperparameters
    mlflow.log_params(params)

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Log the model
    model_info = mlflow.sklearn.log_model(sk_model=lr, name="iris_model")

    # Predict on the test set, compute and log the loss metric
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Optional: Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")
