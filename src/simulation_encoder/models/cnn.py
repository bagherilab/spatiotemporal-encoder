from abc import ABC, abstractmethod
from typing import Optional, Any

from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_encoder.logger import ExperimentLogger


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
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


class CAE(BaseCNN):
    """
    Convolutional neural network class for autoencoding image data.

    Parameters
    ----------
    name : str
        Name of the model
    architecture: dict{str: list[dict{str: Any}]}
        Dictionary containing the architecture of the network
    num_epochs : int
        Number of epochs to train the network
    params : dict{str: Any}
        Dictionary containing model hyperparameters
    logger : Logger, optional
        Logger object for logging, by default None
    """

    def __init__(
        self,
        name: str,
        architecture: dict[str, list[dict[str, Any]]],
        num_epochs: int = 5,
        params: dict[str, Any] = {},
        logger: Optional[ExperimentLogger] = None,
    ):
        super().__init__()

        self.name = name
        self.architecture = architecture
        self.num_epochs = num_epochs
        self.params = params
        self.logger = logger
        self.loss_weights = {
            "image": params.get("image_loss_weight", 1.0),
            "timepoint": params.get("timepoint_loss_weight", 1.0),
        }

        self.encoder = nn.Sequential(*self._create_layers(self.architecture["encoder"]))
        self.decoder_image = nn.Sequential(*self._create_layers(self.architecture["decoder_image"]))
        self.decoder_timepoint = nn.Sequential(
            *self._create_layers(self.architecture["decoder_timepoint"])
        )
        self.optimizers = {
            "combined": torch.optim.Adam(self.parameters(), lr=0.001),
        }
        self.criterion = {
            "image": nn.MSELoss(),
            "timepoint": nn.CrossEntropyLoss(),
        }

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
                msg = f"Epoch {e+1}/{self.num_epochs}- Train loss: {train_loss['combined']} Val loss: {val_loss['combined']}"
                self.logger.log(msg)

        return (train_losses, val_losses, grad_norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs encoding and several decoding heads"""
        z = self.encode(x)
        pred_image = self.decode_image(z)
        pred_timepoint = self.decode_timepoint(z)
        return pred_image, pred_timepoint

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input tensor"""
        return self.encoder(x)

    def decode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes latent tensor to reconstruct image"""
        return self.decoder_image(x)

    def decode_timepoint(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes latent tensor to predict sample timepoint"""
        return self.decoder_timepoint(x)

    def encode_loader(self, dataloader: DataLoader) -> torch.Tensor:
        """Encodes a loader of data"""
        self.eval()
        encoded = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                encoded.append(self.encode(inputs))

        return torch.cat(encoded, dim=0)

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """
        Trains the network for one epoch with batches of data.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing training data
        epoch : int
            Current epoch number

        Returns
        -------
        dict[str, float]
            Dictionary containing average loss for image, timepoint, and combined
        """
        self.train()  # Sets dropout and batch normalization layers to training mode

        optimizer_combined = self.optimizers["combined"]
        image_criteria = self.criterion["image"]
        timepoint_criteria = self.criterion["timepoint"]

        avg_loss: dict[str, float] = defaultdict(float)

        with tqdm(train_loader, unit=" batch", ncols=100, desc=f"Epoch {epoch + 1}") as tepoch:
            for inputs, labels in tepoch:
                optimizer_combined.zero_grad()

                pred_image, pred_timepoint = self(inputs)

                batch_loss = {
                    "image": image_criteria(pred_image, inputs),
                    "timepoint": timepoint_criteria(pred_timepoint, labels),
                }
                combined_loss, combined_loss_weighted = self._calc_combined_loss(batch_loss)

                combined_loss_weighted.backward()
                optimizer_combined.step()

                for key in batch_loss:
                    avg_loss[key] += batch_loss[key].item()
                avg_loss["combined"] += combined_loss.item()

                tepoch.set_postfix(
                    image_loss=round(batch_loss["image"].item(), 3),
                    timepoint_loss=round(batch_loss["timepoint"].item(), 3),
                )

        avg_loss = {key: value / len(train_loader) for key, value in avg_loss.items()}
        return avg_loss

    def eval_one_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """
        Validates the network during training with batches of data.

        Parameters
        ----------
        val_loader : DataLoader
            DataLoader containing validation data

        Returns
        -------
        dict[str, float]
            Dictionary containing average loss for image, timepoint, and combined
        """
        self.eval()  # Sets dropout and batch normalization layers to evaluation mode

        image_criteria = self.criterion["image"]
        timepoint_criteria = self.criterion["timepoint"]

        avg_loss: dict[str, float] = defaultdict(float)
        with torch.no_grad():
            for inputs, labels in val_loader:
                pred_image, pred_timepoint = self(inputs)

                batch_loss = {
                    "image": image_criteria(pred_image, inputs),
                    "timepoint": timepoint_criteria(pred_timepoint, labels),
                }
                combined_loss, _ = self._calc_combined_loss(batch_loss)

                for key in batch_loss:
                    avg_loss[key] += batch_loss[key].item()
                avg_loss["combined"] += combined_loss.item()

        avg_loss = {key: value / len(val_loader) for key, value in avg_loss.items()}
        return avg_loss

    def get_saliency_map(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the saliency map of the input tensor"""
        self.eval()

        image_criteria = self.criterion["image"]
        x.requires_grad = True
        pred_image, _ = self(x)

        loss_image = image_criteria(pred_image, x)
        loss_image.backward()

        saliency_map, _ = torch.max(x.grad.data.abs(), dim=1)
        return saliency_map

    def _create_layers(
        self, layer_configs: list[dict[str, str | int | list[int]]]
    ) -> list[nn.Module]:
        layers = []
        for config in layer_configs:
            layer_type = config.get("type")
            layer_class = getattr(nn, layer_type, None)  # type: ignore
            if layer_class is None:
                raise ValueError(f"Layer type {layer_type} not recognized")

            if layer_type == "Unflatten":
                shape = config.get("shape")
                layer = layer_class(1, tuple(shape))  # type: ignore
            else:
                layer = layer_class(**{k: v for k, v in config.items() if k != "type"})

            layers.append(layer)

        return layers

    def _calc_combined_loss(
        self, losses: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the combined loss from individual losses and weights"""
        combined_loss = sum(losses.values()).detach()
        combined_loss_weighted = sum(
            [losses[key] * self.loss_weights[key] for key in losses.keys()]
        )
        return combined_loss, combined_loss_weighted

    def _get_grad_norm(self, layer: nn.Module) -> torch.Tensor:
        """Calculates the gradient norm of a model"""
        try:
            return torch.norm(layer[-1].weight.grad)
        except AttributeError:
            raise AttributeError(f"{layer} does not have a gradient attribute")
