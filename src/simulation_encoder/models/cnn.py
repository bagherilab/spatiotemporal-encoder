from typing import Optional, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.simulation_encoder.logger import Logger

DEBUG = False


class BaseCNN(nn.Module):
    """
    Base convolutional neural network class.

    Parameters
    ----------
    logger : Logger, optional
        Logger object for logging, by default None
    """

    def __init__(self, logger: Optional[Logger] = None):
        super().__init__()
        self.logger = logger

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        General forward pass of the network
        """
        raise NotImplementedError("forward method should be implemented in the subclass")

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        val_loader: DataLoader = None,
    ) -> tuple[list[float], list[float]]:
        """
        Fits the netwrok over the training data for a number of epochs.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing training data
        epochs : int
            Number of epochs to train the network
        optimizer : torch.optim.Optimizer
            Optimizer to use for training
        loss_fn : torch.nn.Module
            Loss function to use for training
        val_loader: DataLoader, optional
            DataLoader containing validation data, by default None

        Returns
        -------
        list[float], list[float]
            List of training losses and validation losses
        """
        train_losses = []
        val_losses = []
        best_vloss = float("inf")
        for e in range(epochs):
            epoch_loss = self.train_one_epoch(train_loader, optimizer, loss_fn)
            train_losses.append(epoch_loss)

            if val_loader:
                val_loss = self.eval_one_epoch(val_loader, loss_fn)
                val_loss = min(val_loss, best_vloss)
                val_losses.append(val_loss)

            if (e + 1) % 5 == 0:
                if self.logger:
                    self.logger.log(
                        f"Epoch {e+1}/{epochs}: Train loss: {epoch_loss}, Val loss: {val_loss}"
                    )
                print(f"Epoch {e+1}/{epochs}: Train loss: {epoch_loss}, Val loss: {val_loss}")

        if self.logger and val_loader:
            self.logger.log(f"Best validation loss: {best_vloss}")

        return (train_losses, val_losses)

    def train_one_epoch(
        self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module
    ) -> float:
        """
        Trains the network for one epoch with batches of data.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing training data
        optimizer : torch.optim.Optimizer
            Optimizer to use for training
        loss_fn : torch.nn.Module
            Loss function to use for training

        Returns
        -------
        float
            Average loss for the epoch
        """

        # Sets dropout and batch normalization layers to training mode
        self.train()

        running_loss = 0.0
        avg_epoch_loss = 0.0
        for inputs in train_loader:
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = loss_fn(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_epoch_loss = running_loss / len(train_loader)

        return avg_epoch_loss

    def eval_one_epoch(self, val_loader: DataLoader, loss_fn: torch.nn.Module) -> float:
        """
        Validates the network during training with batches of data.

        Parameters
        ----------
        val_loader : DataLoader
            DataLoader containing validation data
        loss_fn : torch.nn.Module
            Loss function to use for validation

        Returns
        -------
        float
            Average validation loss for the epoch
        """

        # Sets dropout and batch normalization layers to evaluation mode
        self.eval()

        running_loss = 0.0
        avg_epoch_loss = 0.0
        with torch.no_grad():
            for inputs in val_loader:
                outputs = self(inputs)
                loss = loss_fn(outputs, inputs)
                running_loss += loss.item()

        avg_epoch_loss = running_loss / len(val_loader)

        return avg_epoch_loss


class ConvolutionalAutoencoder(BaseCNN):
    """
    Convolutional neural network class for autoencoding image data.

    Parameters
    ----------
    input_shape : tuple[int]
        Shape of the input data
    logger : Logger, optional
        Logger object for logging, by default None
    out_channels : int, optional
        Number of output channels for convolutional layers
    dim_z : int, optional
        Dimension of the latent space
    """

    def __init__(
        self,
        input_shape: tuple[int],
        out_channels: int = 16,
        dim_z: int = 2,
        logger: Optional[Logger] = None,
    ):
        super().__init__(logger=logger)

        self.input_shape = input_shape
        self.dim_z = dim_z
        self.in_channels = self.input_shape[0]
        self.out_channels = out_channels

        # Encoder
        self.enc_conv1 = nn.Conv2d(self.input_shape[0], self.out_channels, kernel_size=3, padding=1)
        self.enc_pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.enc_pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.enc_fc1 = nn.Linear(self.out_channels * 64 * 64, 32)
        self.enc_fc2 = nn.Linear(self.out_channels * 2, self.dim_z)

        # Decoder
        self.dec_fc1 = nn.Linear(self.dim_z, self.out_channels * 2)
        self.dec_fc2 = nn.Linear(32, self.out_channels * 64 * 64)
        self.dec_conv1 = nn.ConvTranspose2d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1
        )
        self.dec_upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = nn.ConvTranspose2d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1
        )
        self.dec_upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv3 = nn.ConvTranspose2d(
            self.out_channels, self.input_shape[0], kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs encoding and decoding to recreate original tensor
        """
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input data into a latent space representation.
        """
        debug("Input:", x.size())
        x = torch.relu(self.enc_conv1(x))
        debug("After enc_conv1:", x.size())
        x = self.enc_pool1(x)
        debug("After enc_pool1:", x.size())
        x = torch.relu(self.enc_conv2(x))
        debug("After enc_conv2:", x.size())
        x = self.enc_pool2(x)
        debug("After enc_pool2:", x.size())
        x = torch.relu(self.enc_conv3(x))
        debug("After enc_conv3:", x.size())
        x = x.view(-1, self.out_channels * 64 * 64)
        debug("After view:", x.size())
        x = torch.relu(self.enc_fc1(x))
        debug("After enc_fc1:", x.size())
        x = self.enc_fc2(x)
        debug("After enc_fc2:", x.size())
        debug("-----------------------")
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the input data from the latent space representation.
        """
        x = torch.relu(self.dec_fc1(x))
        debug("After dec_fc1:", x.size())
        x = self.dec_fc2(x)
        debug("After dec_fc2:", x.size())
        x = x.view(-1, self.out_channels, 64, 64)
        debug("After view:", x.size())
        x = torch.relu(self.dec_conv1(x))
        debug("After dec_conv1:", x.size())
        x = self.dec_upsample1(x)
        debug("After dec_upsample1:", x.size())
        x = torch.relu(self.dec_conv2(x))
        debug("After dec_conv2:", x.size())
        x = self.dec_upsample2(x)
        debug("After dec_upsample2:", x.size())
        x = self.dec_conv3(x)
        debug("After dec_conv3:", x.size())
        debug("-----------------------")
        return x


def debug(*args: Any, **kwargs: Any) -> None:
    """
    Prints debug information if DEBUG is set to True.
    """
    if DEBUG:
        print(*args, **kwargs)
