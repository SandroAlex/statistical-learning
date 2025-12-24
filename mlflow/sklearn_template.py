"""
Template script for training a scikit-learn model with MLflow Autologging enabled.

Source: https://mlflow.org/docs/latest/ml/getting-started/quickstart/#step-3---train-a-model-with-mlflow-autologging
"""

# Initial imports
import mlflow

import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Set the MLflow:
# (1) tracking URI;
# (2) experiment name;
# (3) enable autologging for sklearn
mlflow.set_tracking_uri("sqlite:////statapp/mlflow/database.db")
mlflow.set_experiment("Sklearn Autologging")
mlflow.sklearn.autolog()

# Prepare training data
# (1) Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# (2) Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# (3) Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train a model with MLflow Autologging
# (1) Just train the model normally
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
