from abc import ABC, abstractmethod
from typing import Optional, Any

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

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Any:
        """
        Fits the network over the training data for a number of epochs.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing training data.
        epochs : int
            Number of epochs to train the network.
        val_loader: DataLoader, optional
            DataLoader containing validation data, by default None.

        Returns
        -------
            dicts of losses
        """
        pass

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
            if layer_type == "Linear":
                if config.get("out_features") == "latent_dim":
                    config["out_features"] = self.latent_dim
                if config.get("in_features") == "latent_dim":
                    config["in_features"] = self.latent_dim
            if layer_type == "Unflatten":
                shape = config.get("shape")
                layer = layer_class(1, tuple(shape))  # type: ignore
            else:
                layer = layer_class(**{k: v for k, v in config.items() if k != "type"})

            layers.append(layer)

        return layers

    def _get_device(self) -> str:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        return device
