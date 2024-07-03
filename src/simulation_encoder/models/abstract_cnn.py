from abc import ABC, abstractmethod
from typing import Optional, Any
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader


class BaseCNN(ABC, nn.Module):
    """
    Abstract base class for Convolutional Autoencoder (CAE).
    """

    @abstractmethod
    def __init__(
        self,
        name: str = "",
        architecture: dict[str, list[dict[str, Any]]] = {},
        num_channels: int = 1,
        image_size: int = 256,
        num_epochs: int = 10,
        params: dict[str, Any] = {},
        logger: Optional[Any] = None,
    ) -> None:
        """
        Initializes the Convolutional Autoencoder.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary containing model parameters.
        logger : Optional[Any], optional
            Logger object for logging, by default None.
        """
        super().__init__()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
        """
        Fits the network over the training data for a number of epochs.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing training data
        epochs : int
            Number of epochs to train the network
        val_loader: DataLoader, optional
            DataLoader containing validation data, by default None

        Returns
        -------
        dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]
            Dicts of training loss, validation loss, and gradient norms
        """
        self.to(self.device)

        train_losses: dict[str, list[float]] = defaultdict(list)
        val_losses: dict[str, list[float]] = defaultdict(list)
        grad_norms: dict[str, list[float]] = defaultdict(list)

        for e in range(self.num_epochs):
            train_loss = self.train_one_epoch(train_loader, e)
            for loss_type, loss in train_loss.items():
                train_losses[loss_type].append(loss)

            if val_loader:
                val_loss = self.eval_one_epoch(val_loader)
                for loss_type, loss in val_loss.items():
                    val_losses[loss_type].append(loss)

            encoder_grad_norm = self._get_grad_norm(self.encoder)
            decoder_image_grad_norm = self._get_grad_norm(self.decoder_image)
            decoder_timepoint_grad_norm = self._get_grad_norm(self.decoder_timepoint)

            grad_norms["encoder"].append(encoder_grad_norm.item())
            grad_norms["decoder_image"].append(decoder_image_grad_norm.item())
            grad_norms["decoder_timepoint"].append(decoder_timepoint_grad_norm.item())

            if self.logger:
                if self.num_epochs > 10:
                    if (e + 1) % 10 == 0:
                        msg = f"Epoch {e+1}/{self.num_epochs}- Train loss: {train_loss['combined']} Val loss: {val_loss['combined']}"
                        self.logger.log(msg)
                else:
                    msg = f"Epoch {e+1}/{self.num_epochs}- Train loss: {train_loss['combined']} Val loss: {val_loss['combined']}"
                self.logger.log(msg)

        return (train_losses, val_losses, grad_norms)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Performs encoding and decoding."""
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input tensor."""
        pass

    @abstractmethod
    def train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Trains the network for one epoch."""
        pass

    @abstractmethod
    def eval_one_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """Validates the network during training."""
        pass

    def _create_layers(
        self,
        layer_configs: list[dict[str, str | int | list[int]]],
    ) -> list[nn.Module]:
        layers = []
        for config in layer_configs:
            layer_type = config.get("type")
            layer_class = getattr(nn, layer_type, None)  # type: ignore
            if layer_class is None:
                raise ValueError(f"Layer type {layer_type} not recognized")

            # Dynamically set the number of channels and latent dimension size
            if layer_type == "Conv2d":
                if config.get("in_channels") == "num_channels":
                    config["in_channels"] = self.num_channels
                if config.get("out_channels") == "num_channels":
                    config["out_channels"] = self.num_channels
            elif layer_type == "Linear":
                if config.get("out_features") == "latent_dim":
                    config["out_features"] = self.latent_dim
                if config.get("in_features") == "latent_dim":
                    config["in_features"] = self.latent_dim

                if config.get("in_features") == "num_channels":
                    config["in_features"] = self.image_size * self.image_size
                if config.get("out_features") == "num_channels":
                    config["out_features"] = self.image_size * self.image_size

            if layer_type == "Unflatten":
                shape = config.get("shape", [self.num_channels, self.image_size, self.image_size])
                layer = layer_class(1, tuple(shape))  # type: ignore
            else:
                layer = layer_class(**{k: v for k, v in config.items() if k != "type"})

            layers.append(layer)

        return layers

    def _get_grad_norm(self, layer: nn.Sequential) -> torch.Tensor:
        """Calculates the gradient norm of a model"""
        for i in range(1, len(layer) - 1):
            if hasattr(layer[-i], "weight"):
                return torch.norm(layer[-i].weight.grad)

        raise AttributeError(f"No layers have gradient attribute")

    def _get_device(self) -> str:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        return device
