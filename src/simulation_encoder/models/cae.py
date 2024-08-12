from typing import Optional, Any

from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_encoder.logger import Logger
from simulation_encoder.models.base_cnn import BaseCNN


class CAE(BaseCNN):
    """
    Convolutional autoencoder class for encoding image data.

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
    """

    def __init__(
        self,
        name: str,
        architecture: dict[str, list[dict[str, Any]]],
        num_channels: int = 1,
        num_epochs: int = 5,
        image_size: int = 128,
        params: dict[str, Any] = {},
        logger: Optional[Logger] = None,
    ):
        super().__init__()

        self.device = self._get_device()

        self.name = name
        self.architecture = architecture
        self.num_channels = num_channels
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.params = params
        self.logger = logger
        self.latent_dim = params.get("latent_dim", 32)
        self.loss_weights = {
            "image": params.get("image_loss_weight", 1.0),
            "timepoint": params.get("timepoint_loss_weight", 1.0),
        }

        self.encoder = nn.Sequential(*self._create_layers(self.architecture["encoder"].copy()))
        self.decoder_image = nn.Sequential(*self._create_layers(self.architecture["decoder_image"]))
        self.decoder_timepoint = nn.Sequential(
            *self._create_layers(self.architecture["decoder_timepoint"])
        )
        self.optimizers = {
            "combined": torch.optim.Adam(self.parameters(), lr=0.001),
        }
        self.criterion = {
            "image": nn.MSELoss(),
            "timepoint": nn.MSELoss(),
        }

        # Chosen arbitrarily
        # Factor to balance image and timepoint loss
        self.image_loss_factor = 10

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
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
        self.to(self.device)

        encoded = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                z = self.encode(inputs)
                encoded.append(z)
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

        optimizer_combined: torch.optim.Optimizer = self.optimizers["combined"]
        image_criteria: torch.nn.Module = self.criterion["image"]
        timepoint_criteria: torch.nn.Module = self.criterion["timepoint"]

        avg_loss: dict[str, float] = defaultdict(float)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer_combined.zero_grad()

            pred_image, pred_timepoint = self(inputs)
            pred_timepoint = pred_timepoint.squeeze().round()

            batch_loss = {
                "image": image_criteria(pred_image, inputs) * self.image_loss_factor,
                "timepoint": timepoint_criteria(pred_timepoint, labels.float()),
            }
            reconstruction_loss, reconstruction_loss_weighted = self._calc_reconstruction_loss(
                batch_loss
            )

            reconstruction_loss_weighted.backward()  # type: ignore
            optimizer_combined.step()

            for key in batch_loss:
                avg_loss[key] += batch_loss[key].item()
            avg_loss["reconstruction"] += reconstruction_loss.item()
            avg_loss["combined"] += reconstruction_loss_weighted.item()

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
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                pred_image, pred_timepoint = self(inputs)
                pred_timepoint = pred_timepoint.squeeze().round()

                batch_loss = {
                    "image": image_criteria(pred_image, inputs),
                    "timepoint": timepoint_criteria(pred_timepoint, labels),
                }
                reconstruciton_loss, _ = self._calc_reconstruction_loss(batch_loss)

                for key in batch_loss:
                    avg_loss[key] += batch_loss[key].item()
                avg_loss["reconstruction"] += reconstruciton_loss.item()
                avg_loss["combined"] += reconstruciton_loss.item()

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

        saliency_map, _ = torch.max(x.grad.data.abs(), dim=1)  # type: ignore
        return saliency_map

    def _calc_reconstruction_loss(
        self, losses: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the combined loss from individual losses and weights"""
        combined_loss = torch.Tensor(sum([losses[key] for key in losses.keys()])).detach()
        combined_loss_weighted = torch.Tensor(
            sum([losses[key] * self.loss_weights[key] for key in losses.keys()])
        )
        return combined_loss, combined_loss_weighted
