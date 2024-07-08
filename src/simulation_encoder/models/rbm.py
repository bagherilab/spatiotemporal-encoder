import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class RBM(nn.Module):
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

    def __init__(self, visible_dim: int, hidden_dim: int, gaussian: bool, device: str = "cpu"):
        super().__init__()

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

    def train(self, dataloader: DataLoader, num_epochs: int, lr: float = 0.01, k: int = 1) -> None:
        """Train the RBM"""
        loss = nn.MSELoss()
        for epoch in range(num_epochs):
            train_loss = 0
            for data, _ in dataloader:
                v0 = data.to(self.device)
                ph0, hk = self.sample_h(v0)
                for _ in range(k):
                    _, hk = self.sample_h(v0)
                    vk = self.sample_v(hk)
                phk, _ = self.sample_h(vk)

                momentum_coef = 0.5 if epoch < 5 else 0.9
                weight_decay = 2e-4
                batch_size = v0.size(0)

                self.update_weights(v0, vk, ph0, phk, lr, momentum_coef, weight_decay, batch_size)

                train_loss += loss(v0, vk).item()
            print(f"Epoch: {epoch+1}/{num_epochs} Loss: {train_loss/len(dataloader)}")

        return

    def sample_h(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units given visible units"""
        activation = torch.matmul(v, self.W) + self.h_bias
        if self.gaussian:
            return activation, torch.normal(activation, torch.ones_like(activation).to(self.device))
        p = torch.sigmoid(activation)
        return p, torch.bernoulli(p)

    def sample_hk(self, v: torch.Tensor) -> torch.Tensor:
        """Sample hidden units given visible units"""
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


class CRBM(nn.Module):
    """
    Convolutional restricted boltzmann machine

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Size of the kernel
    stride : int
        Stride of the convolution
    padding : int
        Padding of the convolution
    device : str
        Device to run the model on
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        gaussian: bool,
        device: str = "cpu",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.gaussian = gaussian
        self.device = device

        # Initialize parameters
        self.W = (
            torch.randn(out_channels, in_channels, kernel_size, kernel_size).to(self.device) * 0.1
        )
        self.h_bias = torch.zeros(out_channels).to(self.device)
        self.v_bias = torch.zeros(in_channels).to(self.device)

        # Parameters for learning with momentum
        self.W_momentum = torch.zeros(out_channels, in_channels, kernel_size, kernel_size).to(
            self.device
        )
        self.h_bias_momentum = torch.zeros(out_channels).to(self.device)
        self.v_bias_momentum = torch.zeros(in_channels).to(self.device)

        self.to(self.device)

    def train(self, dataloader: DataLoader, num_epochs: int, lr: float = 0.1, k: int = 1):
        """Train the CRBM"""
        loss = nn.MSELoss()
        for epoch in range(num_epochs):
            train_loss = 0
            for data, _ in dataloader:
                v0 = data.to(self.device)
                ph0, _ = self.sample_h(v0)
                for _ in range(k):
                    _, hk = self.sample_h(v0)
                    vk = self.sample_v(hk)
                phk, _ = self.sample_h(vk)

                momentum_coef = 0.5 if epoch < 5 else 0.9
                weight_decay = 2e-4
                batch_size = v0.size(0)

                self.update_weights(v0, vk, ph0, phk, lr, momentum_coef, weight_decay, batch_size)

                train_loss += loss(v0, vk).item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}")

        return

    def sample_h(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units given visible units"""
        conv_op = F.conv2d(v, self.W, self.h_bias, stride=self.stride, padding=self.padding)
        p_h = torch.sigmoid(conv_op)
        if self.gaussian:
            return p_h, torch.normal(p_h, torch.ones_like(p_h).to(self.device))

        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h: torch.Tensor) -> torch.Tensor:
        """Sample visible units given hidden units"""
        deconv_op = F.conv_transpose2d(
            h, self.W, self.v_bias, stride=self.stride, padding=self.padding, output_padding=1
        )
        p_v = torch.sigmoid(deconv_op)
        return p_v

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
        self.W_momentum *= momentum_coef

        pos_grad = F.conv2d(
            v0, ph0, stride=self.stride, padding=self.padding, groups=self.in_channels
        )
        neg_grad = F.conv2d(
            vk, phk, stride=self.stride, padding=self.padding, groups=self.in_channels
        )

        pos_grad = pos_grad.squeeze(2)
        neg_grad = neg_grad.squeeze(2)

        self.W_momentum += pos_grad - neg_grad

        self.h_bias_momentum *= momentum_coef
        self.h_bias_momentum += torch.sum((ph0 - phk), dim=(1, 2, 3))

        self.v_bias_momentum *= momentum_coef
        self.v_bias_momentum += torch.sum((v0 - vk), dim=(1, 2, 3, 4))

        self.W.data += lr * self.W_momentum / batch_size
        self.h_bias.data += lr * self.h_bias_momentum / batch_size
        self.v_bias.data += lr * self.v_bias_momentum / batch_size

        self.W.data -= self.W.data * weight_decay
