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
    verbose : bool
        Boolean to control if training information is printed to the console
    """

    def __init__(self, logger: Optional[ExperimentLogger] = None, verbose: bool = True):
        super().__init__()
        self.logger = logger
        self.verbose = verbose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """General forward pass of the network"""
        raise NotImplementedError("forward method should be implemented in the subclass")
   
    def fit(self, train_loader: DataLoader, epochs: int, val_loader: Optional[DataLoader] = None) -> tuple[list[float], list[float]]:
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

        encoder_layers = self._create_layers(self.params["architecture"]["encoder"])
        decoder_image = self._create_layers(self.params["architecture"]["decoder_image"])
        decoder_timepoint = self._create_layers(self.params["architecture"]["decoder_timepoint"])

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder_image = nn.Sequential(*decoder_image)
        self.decoder_timepoint = nn.Sequential(*decoder_timepoint)

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
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
            epoch_loss = self.train_one_epoch(train_loader)
            train_losses.append(epoch_loss)

            if val_loader:
                val_loss = self.eval_one_epoch(val_loader)
                best_vloss = min(val_loss, best_vloss)
                val_losses.append(val_loss)

            if self.verbose:
                if val_loader:
                    msg = f"Epoch {e+1}/{epochs}- Train loss: {epoch_loss} Val loss: {val_loss}"
                else:
                    msg = f"Epoch {e+1}/{epochs}- Train loss: {epoch_loss}"
                print(msg)

            if (e + 1) % 5 == 0:
                if self.logger:
                    self.logger.log(msg)

        return (train_losses, val_losses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs encoding and several decoding heads"""
        z = self.encode(x)
        image_reconstruction = self.image_decode(z)
        timepoint_prediction = self.timepoint_decode(z)
        return image_reconstruction, timepoint_prediction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input tensor"""
        return self.encoder(x)

    def image_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes latent tensor to reconstruct image"""
        return self.decoder_image(x)

    def timepoint_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes latent tensor to predict sample timepoint"""
        return self.decoder_timepoint(x)
 
    @property
    def image_decoder_params(self):
        """Returns the parameters of the image decoder"""
        return self.decoder_image.parameters()

    @property
    def timepoint_decoder_params(self):
        """Returns the parameters of the timepoint decoder"""
        return self.decoder_timepoint.parameters()

    def encode_loader(self, dataloader: DataLoader) -> torch.Tensor:
        """Encodes a loader of data"""
        self.eval()
        encoded = []
        with torch.no_grad():
            for inputs in dataloader:
                encoded.append(self.encode(inputs))

        return torch.cat(encoded, dim=0)

    def train_one_epoch(
        self, train_loader: DataLoader
    ) -> float:
        """
        Trains the network for one epoch with batches of data.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing training data

        Returns
        -------
        float
            Average loss
        """
        self.train()  # Sets dropout and batch normalization layers to training mode
        image_optimizer = torch.optim.Adam(self.image_decoder_params, lr=0.001)
        timepoint_optimizer = torch.optim.Adam(self.timepoint_decoder_params, lr=0.001)

        avg_loss = {"image": 0.0, "timepoint": 0.0, "combined": 0.0}
        for inputs in train_loader:
            image_optimizer.zero_grad()
            timepoint_optimizer.zero_grad()
            
            image_reconstruction, timepoint_prediction = self(inputs)
            reconstruction_loss = nn.MSELoss()(image_reconstruction, inputs)
            # TODO: Change loss
            timepoint_loss = nn.CrossEntropyLoss()(timepoint_prediction, inputs)

            reconstruction_loss_norm = (reconstruction_loss - min(reconstruction_loss)) / (max(reconstruction_loss) - min(reconstruction_loss))
            timepoint_loss_norm = (timepoint_loss - min(timepoint_loss)) / (max(timepoint_loss) - min(timepoint_loss))

            w_reconstruction = 0.5
            w_timepoint = 0.5

            combined_loss = (w_reconstruction * reconstruction_loss_norm) + (w_timepoint * timepoint_loss_norm)
            combined_loss.backward()
            image_optimizer.step()
            timepoint_optimizer.step()

            avg_loss["image"] += reconstruction_loss.item()
            avg_loss["timepoint"] += timepoint_loss.item()
            avg_loss["combined"] += combined_loss.item()

        avg_loss["image"] /= len(train_loader)
        avg_loss["timepoint"] /= len(train_loader)
        avg_loss["combined"] /= len(train_loader)

        return avg_loss

    def eval_one_epoch(self, val_loader: DataLoader) -> float:
        """
        Validates the network during training with batches of data.

        Parameters
        ----------
        val_loader : DataLoader
            DataLoader containing validation data

        Returns
        -------
        float
            Average validation loss
        """
        self.eval()  # Sets dropout and batch normalization layers to evaluation mode

        avg_loss = {"image": 0.0, "timepoint": 0.0, "combined": 0.0}
        with torch.no_grad():
            for inputs in val_loader:
                image_reconstruction, timepoint_prediction = self(inputs)
                reconstruction_loss = nn.MSELoss()(image_reconstruction, inputs)
                # TODO: Change loss
                timepoint_loss = nn.CrossEntropyLoss()(timepoint_prediction, inputs)

                reconstruction_loss_norm = (reconstruction_loss - min(reconstruction_loss)) / (max(reconstruction_loss) - min(reconstruction_loss))
                timepoint_loss_norm = (timepoint_loss - min(timepoint_loss)) / (max(timepoint_loss) - min(timepoint_loss))

                w_reconstruction = 0.5
                w_timepoint = 0.5

                combined_loss = (w_reconstruction * reconstruction_loss_norm) + (w_timepoint * timepoint_loss_norm)

                avg_loss["image"] += reconstruction_loss.item()
                avg_loss["timepoint"] += timepoint_loss.item()
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
        ]
        for config in layer_configs:
            layer_type = config.pop("type")
            if layer_type in torch_layer_types:
                layer_type = getattr(nn, layer_type)
                activation = config.pop("activation", None)
                layer = layer_type(**config)
                layers.append(layer)
                if activation:
                    activation = getattr(nn, activation)
                    layers.append(activation())
            elif layer_type == "Unflatten":
                layer_type = getattr(nn, layer_type)
                shape = config.pop("shape")
                layer = layer_type(1, tuple(shape))
                layers.append(layer)
            else:
                raise ValueError(f"Layer type {layer_type} not recognized")

        return layers
