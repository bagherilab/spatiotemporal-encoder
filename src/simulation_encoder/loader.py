import os
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from simulation_encoder.logger import ExperimentLogger


class PNGLoader(Dataset):
    """
    Loader class for loading unlabeled images from a directory.

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
        self,
        image_dir: str,
        test_split: float = 0.2,
        logger: Optional[ExperimentLogger] = None,
        healthy_flag: bool = True,
        random_seed: int = 42,
    ):
        self.image_dir = image_dir
        self.test_split = test_split
        self.logger = logger
        self.healthy_flag = healthy_flag
        self.random_seed = random_seed
        self._get_image_groups()
        self._split_data()

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

    def get_train_indices(self) -> list[int]:
        """
        Returns a list of training indices for the dataset
        """
        return self._train_indices

    def get_test_indices(self) -> list[int]:
        """
        Returns a list of test indices for the dataset
        """
        return self._test_indices

    def _get_image_groups(self) -> None:
        """
        Returns groups of images based on the filename format.
        """
        groups = {}
        for filename in os.listdir(self.image_dir):
            if filename.endswith(".png"):
                context, vasc_type, seed, timepoint, image_type = self._parse_filename(filename)
                if image_type == "healthy" and not self.healthy_flag:
                    continue

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

        self.groups = list(groups.values())

    def _split_data(self, shuffle: bool = True) -> None:
        n_datapoints = len(self)
        indices = list(range(n_datapoints))
        split = int(np.floor(self.test_split * n_datapoints))
        if shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        self._train_indices, self._test_indices = indices[split:], indices[:split]

    def _parse_filename(self, filename: str) -> tuple[str, str, int, int, str]:
        parts = filename.split("_")
        context = parts[0]  # 'CH' for healthy tissue or 'C' colony
        vasc_type = parts[1]
        seed = int(parts[2])
        timepoint = int(parts[3])
        if parts[4].split(".")[0] == "graph":
            image_type = parts[4].split(".")[0]
        elif parts[4].split(".")[0] == "cells":
            image_type = parts[5].split(".")[0]
        else:
            raise ValueError(
                f"Invalid name format for image file. Should be \
                            'context_vasc-type_seed_timepoint_image-type.png' Got: {filename}"
            )
        return context, vasc_type, seed, timepoint, image_type

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
