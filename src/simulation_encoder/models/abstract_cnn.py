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
