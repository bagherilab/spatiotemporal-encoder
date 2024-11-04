from abc import ABC, abstractmethod
from typing import Optional, Any
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_encoder.models.rbm import RBM, CRBM
from simulation_encoder.loader import Loader
from simulation_encoder.logger import Logger


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
        image_size: int = 128,
        num_epochs: int = 10,
        params: dict[str, Any] = {},
        logger: Optional[Logger] = None,
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
        pretrain: bool = False,
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

        if pretrain:
            self._log(f"Pretraining encoder using RBM")
            self.pretrain_encoder_rbm(train_loader)

        train_losses: dict[str, list[float]] = defaultdict(list)
        val_losses: dict[str, list[float]] = defaultdict(list)
        grad_norms: dict[str, list[float]] = defaultdict(list)

        for e in range(self.num_epochs):
            train_loss = self.train_one_epoch(train_loader)
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

            msg = f"Epoch {e+1}/{self.num_epochs}- Train loss: {train_loss['weighted_loss']} Val loss: {val_loss['weighted_loss']}"
            self._log(msg)

        return (train_losses, val_losses, grad_norms)

    def pretrain_encoder_rbm(
        self,
        train_loader: DataLoader,
        rbm_epochs: int = 5,
        rbm_lr: float = 0.01,
        data_fraction: float = 0.2,
    ) -> None:
        """
        Pretrains the encoder using a Restricted Boltzmann Machine.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing training data
        rbm_epochs : int
            Number of epochs to train the RBM
        rbm_lr : float, optional
            Learning rate for the RBM
        data_fraction : float, optional
            Fraction of data to use for training
        """
        train_loader = Loader._subsample_loader(train_loader, data_fraction)

        num_layers = len(self.architecture["encoder"])
        is_flattened = False
        for i, (layer_params, layer) in enumerate(zip(self.architecture["encoder"], self.encoder)):
            if layer_params["type"] == "Conv2d":
                in_channels = layer_params["in_channels"]
                out_channels = layer_params["out_channels"]
                kernel_size = layer_params["kernel_size"]
                stride = layer_params["stride"]
                padding = layer_params["padding"]

                self._log(f"CRBM {in_channels} channels to {out_channels} channels")
                crbm = CRBM(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    logger=self.logger,
                    device=self.device,
                )
                crbm.train_machine(train_loader, rbm_epochs, rbm_lr)
                train_loader = Loader._transform_dataloader(
                    crbm.sample_hk, train_loader, self.device
                )

                crbm.W.data = nn.functional.normalize(crbm.W.data, p=2, dim=1)
                crbm.h_bias.data = nn.functional.normalize(crbm.h_bias.data, p=2, dim=0)

                layer.weight.data = crbm.W.data
                layer.bias.data = crbm.h_bias.data

            elif layer_params["type"] == "MaxPool2d":
                kernel_size = layer_params["kernel_size"]
                stride = layer_params["stride"]
                maxpool = nn.MaxPool2d(kernel_size, stride)
                train_loader = Loader._transform_dataloader(maxpool, train_loader, self.device)

            elif layer_params["type"] == "Linear":
                in_features = layer_params["in_features"]
                out_features = layer_params["out_features"]

                if not is_flattened:
                    train_loader = Loader._flatten_loader(train_loader)
                    is_flattened = True

                self._log(f"RBM {in_features} nodes to {out_features} nodes")
                gaussian = True if i == num_layers - 1 else False
                rbm = RBM(
                    in_features, out_features, gaussian, logger=self.logger, device=self.device
                )

                rbm.train_machine(train_loader, rbm_epochs, rbm_lr)

                train_loader = Loader._transform_dataloader(
                    rbm.sample_hk, train_loader, self.device
                )

                rbm.W.data = nn.functional.normalize(rbm.W.data, p=2, dim=1)
                rbm.h_bias.data = nn.functional.normalize(rbm.h_bias.data, p=2, dim=0)
                rbm.v_bias.data = nn.functional.normalize(rbm.v_bias.data, p=2, dim=0)

                layer.weight.data = rbm.W.t().data
                layer.bias.data = rbm.h_bias.data

                decoder_layer = self.decoder_image[num_layers - i - 1]
                decoder_layer.weight.data = rbm.W.data
                decoder_layer.bias.data = rbm.v_bias.data

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
            layer_type = config.get("type")
            layer_class = getattr(nn, layer_type, None)  # type: ignore
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
                    config["in_features"] = self.image_size * self.image_size * self.num_channels
                if config.get("out_features") == "num_channels":
                    config["out_features"] = self.image_size * self.image_size * self.num_channels

            elif layer_type == "BatchNorm1d":
                if config.get("num_features") == "latent_dim":
                    config["num_features"] = self.latent_dim
                if config.get("num_features") == "num_channels":
                    config["num_features"] = self.num_channels * self.image_size * self.image_size

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
        # device = "cpu"
        return device

    def _log(self, msg: str, level: str = "info") -> None:
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            else:
                self.logger.log(msg)
