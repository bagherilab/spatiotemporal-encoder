from abc import ABC, abstractmethod
from typing import Any

import neuralop.models as neuralops_models
import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_encoder.logger import Logger


class BaseNN(ABC, nn.Module):
    """
    Abstract base class for autoencoder networks.
    """

    @abstractmethod
    def __init__(
        self,
        name: str = "",
        architecture: dict[str, list[dict[str, Any]]] = {},
        num_channels: int = 1,
        image_size: int = 128,
        num_epochs: int = 10,
        params: dict[str, Any] = {},
        logger: Logger | None = None,
    ) -> None:
        """
        Initializes the autoencoder.

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
        val_loader: DataLoader | None = None,
        pretrain: bool = False,
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
        """Fits the network over the training data for a number of epochs."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Performs encoding and decoding."""
        pass

    @abstractmethod
    def train_one_epoch(
        self,
        train_loader: DataLoader,
    ) -> dict[str, float]:
        """Trains the network for one epoch."""
        pass

    @abstractmethod
    def eval_one_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """Validates the network during training."""
        pass

    def _create_layers(
        self,
        layer_configs: list[dict],
    ) -> list[nn.Module]:
        layers = []
        for config in layer_configs:
            layer_type: str = config.get("type", "")
            if layer_type == "FNO" or layer_type == "TFNO":
                layer_class = getattr(neuralops_models, layer_type, None)
            else:
                layer_class = getattr(nn, layer_type, None)
            if layer_class is None:
                raise ValueError(f"Layer type {layer_type} not recognized")
            # Dynamically set the number of channels and latent dimension size
            if layer_type == "Conv2d" or layer_type == "ConvTranspose2d":
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
                    config["in_features"] = (
                        self.image_size * self.image_size * self.num_channels
                    )
                if config.get("out_features") == "num_channels":
                    config["out_features"] = (
                        self.image_size * self.image_size * self.num_channels
                    )

            elif layer_type == "BatchNorm1d":
                if config.get("num_features") == "latent_dim":
                    config["num_features"] = self.latent_dim
                if config.get("num_features") == "num_channels":
                    config["num_features"] = (
                        self.num_channels * self.image_size * self.image_size
                    )

            elif layer_type == "FNO":
                if config.get("in_channels") == "num_channels":
                    config["in_channels"] = self.num_channels
                if config.get("out_channels") == "num_channels":
                    config["out_channels"] = self.num_channels

                config.setdefault("n_modes", [16, 16])
                config.setdefault("hidden_channel", 64)

                fno_config = {k: v for k, v in config.items() if k != "type"}

                layer = layer_class(**fno_config)

            if layer_type == "Unflatten":
                shape = config.get(
                    "shape", [self.num_channels, self.image_size, self.image_size]
                )
                layer = layer_class(1, tuple(shape))  # type: ignore
            else:
                layer = layer_class(**{k: v for k, v in config.items() if k != "type"})

            layers.append(layer)

        return layers

    def _get_grad_norm(self, layer: nn.Sequential) -> torch.Tensor:
        """Calculates the gradient norm of a model"""
        for i in range(1, len(layer) - 1):
            try:
                if hasattr(layer[-i], "weight") and layer[-i].weight.grad is not None:
                    return torch.norm(layer[-i].weight.grad)
            except Exception as e:
                print(f"Error accessing gradient for layer {len(layer) - i}: {e}")

        return torch.tensor(0.0, device=next(layer.parameters()).device)

    def _get_device(self) -> str:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return device

    def _log(self, msg: str, level: str = "info") -> None:
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            else:
                self.logger.log(msg)
