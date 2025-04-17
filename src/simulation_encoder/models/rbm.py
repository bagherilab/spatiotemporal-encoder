import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Optional
from abc import ABC, abstractmethod

from simulation_encoder.logger import Logger


class BaseRBM(ABC, nn.Module):
    def __init__(self, logger: Optional[Logger] = None):
        super(BaseRBM, self).__init__()
        self.logger = logger

    @abstractmethod
    def sample_h(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def sample_hk(self, v: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def sample_v(self, h: torch.Tensor) -> torch.Tensor:
        pass

    def train_machine(
        self, dataloader: DataLoader, num_epochs: int = 5, lr: float = 0.01, k: int = 1
    ) -> None:
        """Train the RBM"""
        loss = nn.MSELoss()
        for epoch in range(num_epochs):
            train_loss = 0
            for data, _ in dataloader:
                v0 = data.to(self.device)
                ph0, hk = self.sample_h(v0)

                # Gibbs sampling
                for _ in range(k):
                    vk = self.sample_v(hk)
                    phk, hk = self.sample_h(vk)

                momentum_coef = 0.5 if epoch < 5 else 0.9
                weight_decay = 2e-4
                batch_size = v0.size(0)

                self.update_weights(v0, vk, ph0, phk, lr, momentum_coef, weight_decay, batch_size)
                train_loss += loss(v0, vk).item()

            self._log(f"Epoch: {epoch+1}/{num_epochs} Loss: {train_loss/len(dataloader)}")

        return

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)


class RBM(BaseRBM):
    """
    Restricted boltzmann machine

    Parameters
    ----------
    visible_dim : int
        Number of visible units
    hidden_dim : int
        Number of hidden units
    device : str
        Device to run the model on

    """

    def __init__(
        self,
        visible_dim: int,
        hidden_dim: int,
        gaussian: bool,
        logger: Optional[Logger] = None,
        device: str = "cpu",
    ):
        super(RBM, self).__init__(logger)

        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.gaussian = gaussian
        self.device = device

        # Initialize parameters
        self.W = torch.randn(visible_dim, hidden_dim).to(self.device) * 0.1
        self.h_bias = torch.zeros(hidden_dim).to(self.device)
        self.v_bias = torch.zeros(visible_dim).to(self.device)

        # Parameters for learning with momentum
        self.W_momentum = torch.zeros(visible_dim, hidden_dim).to(self.device)
        self.h_bias_momentum = torch.zeros(hidden_dim).to(self.device)
        self.v_bias_momentum = torch.zeros(visible_dim).to(self.device)

        self.to(self.device)

    def sample_h(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units given visible units"""
        activation = torch.matmul(v, self.W) + self.h_bias
        if self.gaussian:
            return activation, torch.normal(activation, torch.ones_like(activation).to(self.device))
        p = torch.sigmoid(activation)
        return p, torch.bernoulli(p)

    def sample_hk(self, v: torch.Tensor) -> torch.Tensor:
        """Sample only the distribution"""
        activation = torch.matmul(v, self.W) + self.h_bias
        if self.gaussian:
            return torch.normal(activation, torch.ones_like(activation).to(self.device))
        p = torch.sigmoid(activation)
        return torch.bernoulli(p)

    def sample_v(self, h: torch.Tensor) -> torch.Tensor:
        """Sample visible units given hidden units"""
        activation = torch.matmul(h, self.W.t()) + self.v_bias
        p = torch.sigmoid(activation)
        return p

    def update_weights(
        self,
        v0: torch.Tensor,
        vk: torch.Tensor,
        ph0: torch.Tensor,
        phk: torch.Tensor,
        lr: float,
        momentum_coef: float,
        weight_decay: float,
        batch_size: int,
    ) -> None:
        """Update weights of the RBM"""
        self.W_momentum *= momentum_coef
        self.W_momentum += torch.matmul(v0.t(), ph0) - torch.matmul(vk.t(), phk)

        self.h_bias_momentum *= momentum_coef
        self.h_bias_momentum += torch.sum((ph0 - phk), 0)

        self.v_bias_momentum *= momentum_coef
        self.v_bias_momentum += torch.sum((v0 - vk), 0)

        self.W += lr * self.W_momentum / batch_size
        self.h_bias += lr * self.h_bias_momentum / batch_size
        self.v_bias += lr * self.v_bias_momentum / batch_size

        self.W -= self.W * weight_decay


class CRBM(BaseRBM):
    """
    Convolutional Restricted Boltzmann Machine

    Parameters
    ----------
    visible_dim : tuple
        Dimensions of visible units (channels, height, width)
    hidden_dim : tuple
        Dimensions of hidden units (channels, height, width)
    kernel_size : int or tuple
        Size of the convolution kernel
    stride : int
        Stride of the convolution
    padding : int
        Padding of the convolution
    device : str
        Device to run the model on
    """

    def __init__(
        self,
        visible_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        logger: Optional[Logger] = None,
        device: str = "cpu",
    ):
        super(CRBM, self).__init__(logger)

        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

        self.W = nn.Parameter(
            torch.randn(hidden_dim, visible_dim, kernel_size, kernel_size).to(self.device) * 0.1
        )
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim).to(self.device))
        self.v_bias = nn.Parameter(torch.zeros(visible_dim).to(self.device))

        self.W_momentum = torch.zeros(hidden_dim, visible_dim, kernel_size, kernel_size).to(
            self.device
        )
        self.h_bias_momentum = torch.zeros(hidden_dim).to(self.device)
        self.v_bias_momentum = torch.zeros(visible_dim).to(self.device)

        if self.stride > 1:
            self.output_padding = 1
        else:
            self.output_padding = 0

        self.to(self.device)

    def sample_h(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units given visible units"""
        conv_output = F.conv2d(v, self.W, stride=self.stride, padding=self.padding)
        activation = conv_output + self.h_bias.view(1, -1, 1, 1)
        p = torch.sigmoid(activation)
        return p, torch.bernoulli(p)

    def sample_hk(self, v: torch.Tensor) -> torch.Tensor:
        """Sample only the distribution"""
        conv_output = F.conv2d(v, self.W, stride=self.stride, padding=self.padding)
        activation = conv_output + self.h_bias.view(1, -1, 1, 1)
        p = torch.sigmoid(activation)
        return torch.bernoulli(p)

    def sample_v(self, h: torch.Tensor) -> torch.Tensor:
        """Sample visible units given hidden units"""
        conv_transpose_output = F.conv_transpose2d(
            h, self.W, stride=self.stride, padding=self.padding, output_padding=self.output_padding
        )
        activation = conv_transpose_output + self.v_bias.view(1, -1, 1, 1)
        p = torch.sigmoid(activation)
        return p

    def update_weights(
        self,
        v0: torch.Tensor,
        vk: torch.Tensor,
        ph0: torch.Tensor,
        phk: torch.Tensor,
        lr: float,
        momentum_coef: float,
        weight_decay: float,
        batch_size: int,
    ) -> None:
        """Update weights of the CRBM"""
        pos_phase = F.conv2d(
            v0.transpose(0, 1), ph0.transpose(0, 1), padding=self.padding, stride=self.stride
        )
        neg_phase = F.conv2d(
            vk.transpose(0, 1), phk.transpose(0, 1), padding=self.padding, stride=self.stride
        )

        grad_W = (pos_phase - neg_phase) / batch_size
        grad_W = grad_W.transpose(0, 1)

        grad_W = grad_W[:, :, : self.kernel_size, : self.kernel_size]

        grad_h_bias = torch.sum(ph0 - phk, dim=[0, 2, 3]) / batch_size
        grad_v_bias = torch.sum(v0 - vk, dim=[0, 2, 3]) / batch_size

        grad_W = grad_W[
            :,
            :,
        ]
        self.W_momentum = momentum_coef * self.W_momentum + grad_W
        self.h_bias_momentum = momentum_coef * self.h_bias_momentum + grad_h_bias
        self.v_bias_momentum = momentum_coef * self.v_bias_momentum + grad_v_bias

        # Update weights and biases
        self.W.data += lr * self.W_momentum
        self.h_bias.data += lr * self.h_bias_momentum
        self.v_bias.data += lr * self.v_bias_momentum

        # Apply weight decay
        self.W.data -= weight_decay * self.W
