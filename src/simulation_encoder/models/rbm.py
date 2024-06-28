import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class RBM(nn.Module):
    def __init__(self, n_visible: int, n_hidden: int):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Initialize the weight matrix and biases
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_from_p(self, p: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.sign(p - torch.rand(p.size())).detach())

    def v_to_h(self, v: torch.Tensor)   -> torch.Tensor:
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h, self.sample_from_p(p_h)

    def h_to_v(self, h: torch.Tensor) -> torch.Tensor:
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v, self.sample_from_p(p_v)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        p_h, h = self.v_to_h(v)
        p_v, v = self.h_to_v(h)
        return v

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()
