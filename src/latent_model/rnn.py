from typing import Optional, Any

from collections import defaultdict

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_encoder.logger import Logger

class RNN(nn.Module):
    """
    Recurrent neural network class for encoding image data.

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

    def __init__(self):
        super().__init__()