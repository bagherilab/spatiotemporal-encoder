from typing import Optional, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_encoder.logger import ExperimentLogger

DEBUG = False


class BaseCNN(nn.Module):
    """
    Base convolutional neural network class.

    Parameters
    ----------
    logger : Logger, optional
        Logger object for logging, by default None
    verbose : bool
        Boolean to control if training information is printed to the console
    """

    def __init__(self, logger: Optional[ExperimentLogger] = None, verbose: bool = True):
        super().__init__()
        self.logger = logger
        self.verbose = verbose

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
        val_loader: Optional[DataLoader] = None,
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

            if self.verbose:
                print(f"Epoch {e+1}/{epochs}: Train loss: {epoch_loss}, Val loss: {val_loss}")

            if (e + 1) % 5 == 0:
                if self.logger:
                    self.logger.log(
                        f"Epoch {e+1}/{epochs}: Train loss: {epoch_loss}, Val loss: {val_loss}"
                    )

        if val_loader:
            if self.logger:
                self.logger.log(f"Best validation loss: {best_vloss}")
            if self.verbose:
                print(f"Best validation loss: {best_vloss}")

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


class CAE(BaseCNN):
    """
    Convolutional neural network class for autoencoding image data.

    Parameters
    ----------
    input_shape : tuple[int]
        Shape of the input data
    conf : str
        Config file with model layers
    logger : Logger, optional
        Logger object for logging, by default None
    verbose : bool
        Controls if model training is output to console
    """

    def __init__(
        self,
        params: dict[str, Any],
        logger: Optional[ExperimentLogger] = None,
        verbose: bool = True,
    ):
        super().__init__(logger=logger, verbose=verbose)

        self.params = params

        encoder_layers = self._create_layers(self.params["architecture"]["encoder_layers"])
        decoder_layers = self._create_layers(self.params["architecture"]["decoder_layers"])

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs encoding and decoding to recreate original tensor
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _create_layers(self, layer_configs):
        layers = []
        for config in layer_configs:
            layer_type = getattr(nn, config["type"])
            if "in_channels" in config and "out_channels" in config:
                layer = layer_type(config["in_channels"], config["out_channels"])
            elif "in_features" in config and "out_features" in config:
                layer = layer_type(config["in_features"], config["out_features"])
            else:
                layer = layer_type(**config)
            layers.append(layer)
        return layers
