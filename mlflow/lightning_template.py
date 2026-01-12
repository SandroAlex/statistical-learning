"""
Tutorial on how to simplify deep learning model development with PyTorch
Lightning and MLflow integration for experiment tracking.

Source: https://www.datacamp.com/tutorial/pytorch-lightning-tutorial
"""

# Initial imports
import torch
import mlflow

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class CIFAR10CNN(L.LightningModule):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        # Log the loss at each training step and epoch, create a progress bar
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# Version info
print("\n>>> Main dependencies versions:")
print("* Lightning version:", L.__version__)
print("* Torch version:", torch.__version__)
print("* CUDA is available:", torch.cuda.is_available())
print("* MLflow version:", mlflow.__version__)

# Set the MLflow
mlflow.set_tracking_uri("sqlite:////statapp/mlflow/database.db")
mlflow.set_experiment("Pytorch Lightning")
mlflow.enable_system_metrics_logging()

# Set seed for reproducibility
L.seed_everything(1121218)

# Hyperparameters for experiment
num_epochs = 2
batch_size = 32
learning_rate = 0.001

# Data augmentation and normalization for training
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ],
)

# Normalization for testing
transform_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(
    root="./data/cifar10", train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root="./data/cifar10", train=False, download=True, transform=transform_test
)

# Data loaders
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

# Callbacks and loggers
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    monitor="val_loss",
    filename="cifar10-{epoch:02d}-{step:04d}-{val_loss:.2f}-{val_acc:.2f}",
    save_top_k=3,
    mode="min",
)
logger = TensorBoardLogger(save_dir="./lightning_logs", name="cifar10_cnn")
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, mode="min", verbose=False
)
