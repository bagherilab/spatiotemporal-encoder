import copy
from typing import Optional, Any
from collections import defaultdict

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_encoder.logger import Logger
from simulation_encoder.loaders.loader import Loader
from simulation_encoder.models.base_nn import BaseNN


class AE(BaseNN):
    """
    Convolutional autoencoder class for encoding image data.

    Parameters
    ----------
    name : str
        Name of the model
    architecture: dict{str: list[dict{str: Any}]}
        Dictionary containing the architecture of the network
    num_channels: int
        The number of channels the input has
    num_epochs : int
        Number of epochs to train the network
    image_size: int
        Number of pixels for on one side of the square input image
    params : dict{str: Any}
        Dictionary containing model hyperparameters
    """

    def __init__(
        self,
        name: str,
        architecture: dict[str, list[dict[str, Any]]],
        num_channels: int = 1,
        num_timepoints: int = 1,
        num_epochs: int = 5,
        image_size: int = 128,
        params: dict[str, Any] = {},
        logger: Optional[Logger] = None,
        sequence: bool = False,
    ):
        super().__init__()

        self.device = self._get_device()

        self.name = name
        self.architecture = architecture
        self.num_channels = num_channels
        self.num_timepoints = num_timepoints
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.params = params
        self.logger = logger
        self.sequence = sequence
        self.latent_dim = params.get("latent_dim", 32)
        self.loss_weights = {
            "image": params.get("image_loss_weight", 1.0),
        }
        self.encoder = nn.Sequential(*self._create_layers(self.architecture["encoder"].copy()))
        self.decoder_image = nn.Sequential(*self._create_layers(self.architecture["decoder_image"]))
        
        if not self.sequence:
            self.decoder_timepoint = nn.Sequential(
                *self._create_layers(self.architecture["decoder_timepoint"])
            )
            self.loss_weights["timepoint"] = params.get("timepoint_loss_weight", 1.0)

        optimizer_config = params.get("optimizer", {})
        optimizer_type = optimizer_config.pop("type")
        self.optimizers = {"combined": optimizer_type(self.parameters(), **optimizer_config)}

        optimizer_name = optimizer_type.__name__
        self.params["optimizer"]["type"] = optimizer_name

        self.criterion = {
            "image": nn.MSELoss(),
        }
        if not self.sequence:
            self.criterion["timepoint"] = nn.CrossEntropyLoss()

        # Chosen arbitrarily
        # Factor to balance image and timepoint loss
        self.image_loss_lambda = 10

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Performs encoding and several decoding heads"""
        z = self.encode(x)
        pred_image = self.decode_image(z)
        if not self.sequence:
            pred_timepoint = self.decode_timepoint(z)
            return pred_image, pred_timepoint
        return (pred_image,)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
        """
        Fits the network over the training data for a number of epochs.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing training data
        val_loader: DataLoader, optional
            DataLoader containing validation data, by default None
        patience : int, optional
            Number of epochs to wait for improvement before early stopping, by default 5
        min_delta : float, optional
            Minimum change in loss to be considered an improvement, by default 0.0

        Returns
        -------
        dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]
            Dicts of training loss, validation loss, and gradient norms
        """
        self.to(self.device)

        train_losses: dict[str, list[float]] = defaultdict(list)
        val_losses: dict[str, list[float]] = defaultdict(list)
        grad_norms: dict[str, list[float]] = defaultdict(list)

        best_val_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        for e in range(self.num_epochs):
            train_loss = self.train_one_epoch(train_loader)
            for loss_type, loss in train_loss.items():
                train_losses[loss_type].append(loss)

            if val_loader:
                val_loss = self.eval_one_epoch(val_loader)
                for loss_type, loss in val_loss.items():
                    val_losses[loss_type].append(loss)

                # Early stopping
                current_val_loss = val_loss["weighted_loss"]
                if current_val_loss < best_val_loss - min_delta:
                    best_val_loss = current_val_loss
                    epochs_without_improvement = 0
                    best_weights = copy.deepcopy(self.state_dict())
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        self._log(
                            f"Early stopping at epoch {e+1}. Best validation loss: {round(best_val_loss, 6)}"
                        )
                        break

            encoder_grad_norm = self._get_grad_norm(self.encoder)
            decoder_image_grad_norm = self._get_grad_norm(self.decoder_image)
            grad_norms["encoder"].append(encoder_grad_norm.item())
            grad_norms["decoder_image"].append(decoder_image_grad_norm.item())
            
            if not self.sequence:
                decoder_timepoint_grad_norm = self._get_grad_norm(self.decoder_timepoint)
                grad_norms["decoder_timepoint"].append(decoder_timepoint_grad_norm.item())

            msg = f"Epoch {e+1}/{self.num_epochs}- Train loss: {round(train_loss['weighted_loss'], 6)} Val loss: {round(val_loss['weighted_loss'], 6)}"
            self._log(msg)

        if best_weights is not None:
            best_weights.pop("_metadata", None)
            self.load_state_dict(best_weights)

        return (train_losses, val_losses, grad_norms)

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
        
        if not self.sequence:
            timepoint_criteria: torch.nn.Module = self.criterion["timepoint"]

        avg_loss: dict[str, float] = defaultdict(float)

        with tqdm(train_loader, desc="Training", unit="batch", ncols=120) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer_combined.zero_grad()
                
                if self.sequence:
                    (pred_image,) = self(inputs)
                    batch_loss = {
                        "image": image_criteria(pred_image, inputs),
                    }
                else:
                    pred_image, pred_timepoint = self(inputs)
                    batch_loss = {
                        "image": image_criteria(pred_image, inputs),
                        "timepoint": timepoint_criteria(pred_timepoint, labels),
                    }

                _, reconstruction_loss_weighted = self._calc_reconstruction_loss(batch_loss)

                reconstruction_loss_weighted.backward()  # type: ignore
                optimizer_combined.step()

                for key in batch_loss:
                    avg_loss[key] += batch_loss[key].item()
                avg_loss["weighted_loss"] += reconstruction_loss_weighted.item()

                pbar.set_postfix({"Weighted Loss": reconstruction_loss_weighted.item()})

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
        if not self.sequence:
            timepoint_criteria = self.criterion["timepoint"]

        avg_loss: dict[str, float] = defaultdict(float)
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", unit="batch", ncols=120) as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    if self.sequence:
                        (pred_image,) = self(inputs)
                        batch_loss = {
                            "image": image_criteria(pred_image, inputs),
                        }
                    else:
                        pred_image, pred_timepoint = self(inputs)
                        batch_loss = {
                            "image": image_criteria(pred_image, inputs),
                            "timepoint": timepoint_criteria(pred_timepoint, labels),
                        }

                    reconstruciton_loss, reconstruciton_loss_weighted = (
                        self._calc_reconstruction_loss(batch_loss)
                    )
                    for key in batch_loss:
                        avg_loss[key] += batch_loss[key].item()
                    avg_loss["weighted_loss"] += reconstruciton_loss_weighted.item()

                    pbar.set_postfix({"Weighted Loss": reconstruciton_loss_weighted.item()})

        avg_loss = {key: value / len(val_loader) for key, value in avg_loss.items()}
        return avg_loss

    def get_saliency_map(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the saliency map of the input tensor"""
        self.eval()

        image_criteria = self.criterion["image"]
        x.requires_grad = True

        try:
            if self.sequence:
                (pred_image,) = self(x)
            else:
                pred_image, _ = self(x)
            loss_image = image_criteria(pred_image, x)
            loss_image.backward()

            if x.grad is not None:
                saliency_map, _ = torch.max(x.grad.data.abs(), dim=1)  # type: ignore
            else:
                raise RuntimeError("Gradient is None")

        except RuntimeError as e:
            print(f"Error during saliency map computation: {e}")
            saliency_map = torch.zeros_like(x[:, 0, :, :])

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