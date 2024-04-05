import os
from typing import Any, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from simulation_encoder.logger import ExperimentLogger


class UnlabeledImageDataset(Dataset):
    """
    Dataset class for loading unlabeled images from a directory.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing the images.
    logger : Logger, optional
        Logger object for logging missing images, by default None
    healthy_flag : bool, optional
        Flag to include healthy tissue images, by default True
    """

    def __init__(
        self, image_dir: str, logger: Optional[ExperimentLogger] = None, healthy_flag: bool = True
    ):
        self.image_dir = image_dir
        self.logger = logger
        self.healthy_flag = healthy_flag
        self.groups = self._get_image_groups()

    def _get_image_groups(self) -> list[Any]:
        """
        Returns groups of images based on the filename format.

        """
        groups = {}
        for filename in os.listdir(self.image_dir):
            if filename.endswith(".png"):
                parts = filename.split("_")
                context = parts[0]  # 'CH' for healthy tissue or 'C' colony
                vasc_type = parts[1]
                seed = int(parts[2])
                timepoint = int(parts[3])
                if parts[4].split(".")[0] == "graph":
                    image_type = parts[4].split(".")[0]
                elif parts[4].split(".")[0] == "cells":
                    image_type = parts[5].split(".")[0]
                    if not self.healthy_flag and image_type == "healthy":
                        continue
                else:
                    raise ValueError(
                        f"Invalid name format for image file. Should be \
                            'context_vasc-type_seed_timepoint_image-type.png' Got: {filename}"
                    )
                group_key = (context, vasc_type, seed, timepoint)
                if group_key not in groups:
                    if self.healthy_flag:
                        groups[group_key] = {"cancer": "", "healthy": "", "graph": ""}
                    else:
                        groups[group_key] = {"cancer": "", "graph": ""}
                groups[group_key][image_type] = os.path.join(self.image_dir, filename)

        if self.logger:
            for group_key, group in groups.items():
                if "" in group.values():
                    missing_images = [key for key, value in group.items() if value == ""]
                    self.logger.warning(f"Missing images from {group_key}: {missing_images}")

        return list(groups.values())

    @staticmethod
    def display_tensor(tensor: torch.Tensor, name: str) -> None:
        """
        Saves a given tensor as an image

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to save as an image
        name : str
            Name of the file to save the image as
        """
        array = np.moveaxis(tensor.detach().numpy() * 255, 0, -1)
        array = array.squeeze()
        image = Image.fromarray(array.astype("uint8"))
        image.save(name)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> torch.Tensor:
        group = self.groups[idx]
        cancer_path = group["cancer"]
        healthy_path = group.get("healthy", "")
        graph_path = group["graph"]

        transformation = transforms.Compose([transforms.ToTensor()])

        cancer_tensor = transformation(Image.open(cancer_path).convert("L")).squeeze()
        graph_tensor = transformation(Image.open(graph_path).convert("L")).squeeze()

        if self.healthy_flag and healthy_path:
            healthy_tensor = transformation(Image.open(healthy_path).convert("L")).squeeze()
            return torch.stack((cancer_tensor, healthy_tensor, graph_tensor), dim=0)
        return torch.stack((cancer_tensor, graph_tensor), dim=0)
