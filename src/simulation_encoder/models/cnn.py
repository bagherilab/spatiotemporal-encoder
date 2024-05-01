from typing import Optional, Any

from tqdm import tqdm
from collections import defaultdict

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

    # def fit(
    #     self, train_loader: DataLoader, epochs: int, val_loader: Optional[DataLoader] = None
    # ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    #     """Fits the network over the training data for a number of epochs."""
    #     raise NotImplementedError("fit method should be implemented in the subclass")


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

        self.optimizers = {
            "combined": torch.optim.Adam(self.parameters(), lr=0.001),
        }
        self.criterion = {
            "image": nn.MSELoss(),
            "timepoint": nn.CrossEntropyLoss(),
        }

        self.loss_weights = {"image": 0.8, "timepoint": 0.2}

    @property
    def image_decoder_params(self) -> list[torch.nn.Parameter]:
        """Returns the parameters of the image decoder"""
        return self.decoder_image.parameters()

    @property
    def timepoint_decoder_params(self) -> list[torch.nn.Parameter]:
        """Returns the parameters of the timepoint decoder"""
        return self.decoder_timepoint.parameters()

    @property
    def encoder_params(self) -> list[torch.nn.Parameter]:
        """Returns the parameters of the encoder"""
        return self.encoder.parameters()

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
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

        for e in range(epochs):
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
                if (e + 1) % 1 == 0:
                    msg = f"Epoch {e+1}/{epochs}- Train loss: {train_loss['combined']} Val loss: {val_loss['combined']}"
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

                batch_loss = {"image": torch.zeros(1), "timepoint": torch.zeros(1)}

                pred_image, pred_timepoint = self(inputs)
                batch_loss["image"] = image_criteria(pred_image, inputs)
                batch_loss["timepoint"] = timepoint_criteria(pred_timepoint, labels)

                combined_loss = self._calc_combined_loss(batch_loss)
                combined_loss.backward()
                optimizer_combined.step()

                avg_loss["image"] += batch_loss["image"].item()
                avg_loss["timepoint"] += batch_loss["timepoint"].item()
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
                batch_loss = {"image": torch.zeros(1), "timepoint": torch.zeros(1)}
                pred_image, pred_timepoint = self(inputs)
                batch_loss["image"] = image_criteria(pred_image, inputs)
                batch_loss["timepoint"] = timepoint_criteria(pred_timepoint, labels)

                combined_loss = self._calc_combined_loss(batch_loss)

                avg_loss["image"] += batch_loss["image"].item()
                avg_loss["timepoint"] += batch_loss["timepoint"].item()
                avg_loss["combined"] += combined_loss.item()

        avg_loss = {key: value / len(val_loader) for key, value in avg_loss.items()}

        return avg_loss
    
    def get_saliency_map(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the saliency map of the input tensor"""
        self.eval()
        image_criteria = self.criterion["image"]

        x.requires_grad = True
        
        pred_image, pred_timepoint = self(x)
        
        loss_image = image_criteria(pred_image, x)
        loss_image.backward()

        saliency_map, _ = torch.max(x.grad.data.abs(), dim=1)

        return saliency_map


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

    def _get_grad_norm(self, layer: nn.Module) -> torch.Tensor:
        """Calculates the gradient norm of a model"""
        try:
            return torch.norm(layer[-1].weight.grad)
        except AttributeError:
            raise AttributeError(f"{layer} does not have a gradient attribute")
