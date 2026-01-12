"""
Tutorial on how to simplify deep learning model development with PyTorch
Lightning and MLflow integration for experiment tracking.

Source: https://www.datacamp.com/tutorial/pytorch-lightning-tutorial
"""

# Initial imports
import torch
import mlflow
from typing import Optional

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


class CIFAR10DataModule(L.LightningDataModule):

    def __init__(
        self, data_dir: str = "./data", batch_size: int = 64, num_workers: int = 2
    ) -> None:

        super().__init__()

        # Define any custom user-defined parameters
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def prepare_data(self) -> None:

        # Notes:
        # Since prepare_data() runs on a single process, you shouldn't set the class
        # state here. In other words, don,t use the self keyword inside prepare_data()
        # because a variable is defined as self.x = 1 won't be available to different
        # processes.

        # Download the data
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit" or stage is None:

            self.cifar_train = datasets.CIFAR10(
                self.data_dir, train=True, transform=self.transform_train
            )
            self.cifar_val = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.transform_test
            )

        if stage == "test" or stage is None:

            self.cifar_test = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.transform_test
            )

    def train_dataloader(self):

        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):

        return DataLoader(
            self.cifar_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):

        return DataLoader(
            self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers
        )


# Version info
print("\n>>> Main dependencies versions:")
print("* Lightning version:", L.__version__)
print("* Torch version:", torch.__version__)
print("* CUDA is available:", torch.cuda.is_available())
print("* MLflow version:", mlflow.__version__)
print("")

# Set the MLflow
mlflow.set_tracking_uri("sqlite:////statapp/mlflow/database.db")
mlflow.set_experiment("Pytorch Lightning")
mlflow.enable_system_metrics_logging()
mlflow.pytorch.autolog()

# Set seed for reproducibility
L.seed_everything(1121218)

# Hyperparameters for experiment
num_epochs = 5
batch_size = 32
learning_rate = 0.001
num_workers = 8

# Callbacks and loggers
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    monitor="val_loss",
    filename="cifar10-{epoch:02d}-{step:04d}-{val_loss:.2f}-{val_acc:.2f}",
    save_top_k=1,
    mode="min",
)
logger = TensorBoardLogger(save_dir="./lightning_logs", name="cifar10_cnn")
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, mode="min", verbose=False
)

# Initialize the data module
data_module = CIFAR10DataModule(num_workers=num_workers, batch_size=batch_size)

# Initialize the model
model = CIFAR10CNN()

# Initialize the trainer
trainer = L.Trainer(
    fast_dev_run=False,
    max_epochs=5,
    callbacks=[checkpoint_callback, early_stopping],
    logger=logger,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices="auto",
)

# Start a new mlflow run and set the run name
run_name: str = "experiment-002"
with mlflow.start_run(run_name=run_name):

    # Register hyperparameters with MLflow
    mlflow.log_params(
        {
            "Author": "Alex Araujo",
            "email": "alex.fate2000@gmail.com",
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_workers": num_workers,
        }
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Test the model on the validation dataset
    trainer.test(model, datamodule=data_module)
