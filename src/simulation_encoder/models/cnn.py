from typing import Optional, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_encoder.logger import ExperimentLogger


class BaseCNN(nn.Module):
    """
    Base convolutional neural network class.

    Parameters
    ----------
    logger : Logger, optional
        Logger object for logging, by default None
    """

    def __init__(self, logger: Optional[ExperimentLogger] = None):
        super().__init__()
        self.logger = logger

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """General forward pass of the network"""
        raise NotImplementedError("forward method should be implemented in the subclass")

    def fit(
        self, train_loader: DataLoader, epochs: int, val_loader: Optional[DataLoader] = None
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """Fits the network over the training data for a number of epochs."""
        raise NotImplementedError("fit method should be implemented in the subclass")


class CAE(BaseCNN):
    """
    Convolutional neural network class for autoencoding image data.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary containing model parameters from yaml file
    logger : Logger, optional
        Logger object for logging, by default None
    """

    def __init__(
        self,
        params: dict[str, Any],
        logger: Optional[ExperimentLogger] = None,
    ):
        super().__init__(logger=logger)

        self.params = params

        encoder_layers = self._create_layers(self.params["architecture"]["encoder"])
        decoder_image = self._create_layers(self.params["architecture"]["decoder_image"])
        decoder_timepoint = self._create_layers(self.params["architecture"]["decoder_timepoint"])

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder_image = nn.Sequential(*decoder_image)
        self.decoder_timepoint = nn.Sequential(*decoder_timepoint)

        self.loss_weights = {"image": 0.5, "timepoint": 0.5}

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """
        Fits the netwrok over the training data for a number of epochs.

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
        dict[str, list[float]], dict[str, list[float]]
            Lists of various types of loss
        """
        optimizers = {
            "image": torch.optim.Adam(self.image_decoder_params, lr=0.001),
            "timepoint": torch.optim.Adam(self.timepoint_decoder_params, lr=0.001),
        }

        train_losses: dict[str, list[float]] = {"image": [], "timepoint": [], "combined": []}
        val_losses: dict[str, list[float]] = {"image": [], "timepoint": [], "combined": []}
        for e in range(epochs):
            train_loss = self.train_one_epoch(train_loader, optimizers)
            for loss_type, loss in train_loss.items():
                train_losses[loss_type].append(loss)

            if val_loader:
                val_loss = self.eval_one_epoch(val_loader)
                for loss_type, loss in val_loss.items():
                    val_losses[loss_type].append(loss)

            if self.logger:
                if (e + 1) % 1 == 0:
                    msg = f"Epoch {e+1}/{epochs}- Train loss: {train_loss['combined']} Val loss: {val_loss['combined']}"
                    self.logger.log(msg)

        return (train_losses, val_losses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs encoding and several decoding heads"""
        z = self.encode(x)
        image_reconstruction = self.decode_image(z)
        timepoint_prediction = self.decode_timepoint(z)
        return image_reconstruction, timepoint_prediction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input tensor"""
        return self.encoder(x)

    def decode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes latent tensor to reconstruct image"""
        return self.decoder_image(x)

    def decode_timepoint(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes latent tensor to predict sample timepoint"""
        return self.decoder_timepoint(x)

    @property
    def image_decoder_params(self) -> list[torch.nn.Parameter]:
        """Returns the parameters of the image decoder"""
        return self.decoder_image.parameters()

    @property
    def timepoint_decoder_params(self) -> list[torch.nn.Parameter]:
        """Returns the parameters of the timepoint decoder"""
        return self.decoder_timepoint.parameters()

    def encode_loader(self, dataloader: DataLoader) -> torch.Tensor:
        """Encodes a loader of data"""
        self.eval()
        encoded = []
        with torch.no_grad():
            for input, _ in dataloader:
                encoded.append(self.encode(input))

        return torch.cat(encoded, dim=0)

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> dict[str, float]:
        """
        Trains the network for one epoch with batches of data.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing training data

        Returns
        -------
        dict[str, float]
            Dictionary containing average loss for image, timepoint, and combined
        """
        self.train()  # Sets dropout and batch normalization layers to training mode

        image_optimizer = optimizers["image"]
        timepoint_optimizer = optimizers["timepoint"]

        image_criteria = nn.MSELoss()
        timepoint_criteria = nn.CrossEntropyLoss()

        avg_loss = {"image": 0.0, "timepoint": 0.0, "combined": 0.0}
        for inputs, labels in train_loader:
            image_optimizer.zero_grad()
            timepoint_optimizer.zero_grad()

            batch_loss = {"image": torch.zeros(1), "timepoint": torch.zeros(1)}

            image_reconstruction, timepoint_prediction = self(inputs)
            batch_loss["image"] = image_criteria(image_reconstruction, inputs)
            batch_loss["timepoint"] = timepoint_criteria(timepoint_prediction, labels)

            # combined_loss = self._calc_combined_loss(batch_loss)
            combined_loss = batch_loss["image"]
            combined_loss.backward()
            image_optimizer.step()
            # timepoint_optimizer.step()

            avg_loss["image"] += batch_loss["image"].item()
            avg_loss["timepoint"] += batch_loss["timepoint"].item()
            avg_loss["combined"] += combined_loss.item()

        avg_loss["image"] /= len(train_loader)
        avg_loss["timepoint"] /= len(train_loader)
        avg_loss["combined"] /= len(train_loader)

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

        image_criteria = nn.MSELoss()
        timepoint_criteria = nn.CrossEntropyLoss()

        avg_loss = {"image": 0.0, "timepoint": 0.0, "combined": 0.0}
        with torch.no_grad():
            for inputs, labels in val_loader:
                batch_loss = {"image": torch.zeros(1), "timepoint": torch.zeros(1)}
                image_reconstruction, timepoint_prediction = self(inputs)
                batch_loss["image"] = image_criteria(image_reconstruction, inputs)
                batch_loss["timepoint"] = timepoint_criteria(timepoint_prediction, labels)

                # combined_loss = self._calc_combined_loss(batch_loss)
                combined_loss = batch_loss["image"]

                avg_loss["image"] += batch_loss["image"].item()
                avg_loss["timepoint"] += batch_loss["timepoint"].item()
                avg_loss["combined"] += combined_loss.item()

        avg_loss["image"] /= len(val_loader)
        avg_loss["timepoint"] /= len(val_loader)
        avg_loss["combined"] /= len(val_loader)

        return avg_loss

    def _create_layers(self, layer_configs: dict[Any, Any]) -> list[nn.Module]:
        layers = []
        torch_layer_types = [
            "Conv2d",
            "ConvTranspose2d",
            "Linear",
            "MaxPool2d",
            "UpsamplingBilinear2d",
            "Flatten",
            "ReLU",
            "BatchNorm2d",
            "BatchNorm1d",
        ]
        for config in layer_configs:
            layer_type = config.pop("type")
            if layer_type in torch_layer_types:
                layer_type = getattr(nn, layer_type)
                layer = layer_type(**config)
                layers.append(layer)
            elif layer_type == "Unflatten":
                layer_type = getattr(nn, layer_type)
                shape = config.pop("shape")
                layer = layer_type(1, tuple(shape))
                layers.append(layer)
            else:
                raise ValueError(f"Layer type {layer_type} not recognized")

        return layers

    def _calc_combined_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculates the combined loss from individual losses and weights"""
        combined_loss = sum([losses[key] * self.loss_weights[key] for key in losses.keys()])
        return combined_loss
