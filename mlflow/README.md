# MLFlow

Configurations for executing machine learning experiments with mlflow.

## Contents

The scripts here are templates for Mlflow experiments:

- `pytorch_template.py`:
    - Template script for training a PyTorch model with MLflow Manual Logging.
- `pytorch_template2.py`:
    - Template script for training a PyTorch model with MLflow Autologging enabled.
- `pytorch_template3.py`:
    - Template script for training a PyTorch model with MLflow Manual Logging and system metrics enabled. Track hardware resource utilization during training to monitor GPU usage, memory consumption, and system performance.
- `pytorch_template4.py`: 
    - Template script for training a PyTorch model with MLflow Manual Logging and model signatures. Ensure that the input and output schema of the model is logged for better model management and deployment.
- `pytorch_template5.py`:
    - PyTorch iris classifier with MLflow checkpointing and metric logging. Track model checkpoints during training with MLflow 3's checkpoint versioning. Use the step parameter to version checkpoints and link metrics to specific model versions.
- `pytorch_template6.py`:
    - Template for integrating PyTorch model training with MLflow and Optuna for hyperparameter optimization. This example demonstrates how to set up anOptuna study to optimize hyperparameters while logging experiments with MLflow.
- `pytorch_template7.py`:
    - Template script to log a PyTorch model to MLflow Model Registry with tags and alias. Register PyTorch models for version control and deployment.
- `sklearn_template.py`
    - Template script for training a scikit-learn model with MLflow Autologging enabled.
- `sklearn_template2.py`
    - Template script for training a scikit-learn model with MLflow Manual Logging.