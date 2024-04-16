import os
import json
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from simulation_encoder.logger import ExperimentLogger
from simulation_encoder.writer import Writer


class PNGLoader(Dataset):
    """
    Loader class for loading unlabeled images from a directory.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing the images.
    logger : Logger, optional
        Logger object for logging missing images, by default None
    writer : Writer, optional
        Writer object for saving things to disk
    healthy_flag : bool, optional
        Flag to include healthy tissue images, by default True
    """

    def __init__(
        self,
        image_dir: str,
        test_split: float = 0.2,
        batch_size: int = 10,
        logger: Optional[ExperimentLogger] = None,
        writer: Optional[Writer] = None,
        healthy_flag: bool = False,
        random_seed: int = 42,
        indices_file: Optional[str] = None,
        uuid: Optional[str] = None,
    ):
        self.image_dir = image_dir
        self.test_split = test_split
        self.batch_size = batch_size
        self.logger = logger
        self.writer = writer
        self.healthy_flag = healthy_flag
        self.random_seed = random_seed
        self.indices_file = indices_file
        self.uuid = uuid
        self._get_image_groups()

        if indices_file and os.path.exists(indices_file):
            self._load_from_indices(indices_file)
        else:
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

    def get_train_dataloader(self) -> DataLoader:
        """Returns training DataLoader"""
        train_dataset = Subset(self, self._train_indices)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_test_dataloader(self) -> DataLoader:
        """Returns test DataLoader"""
        test_dataset = Subset(self, self._test_indices)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_cv_splits(self, k_folds: int) -> list[tuple[DataLoader, DataLoader]]:
        """Returns list of k-folds of training and validation DataLoader"""
        indices = self._train_indices
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        fold_size = len(indices) // k_folds
        folds = []
        for i in range(k_folds):
            val_indices = indices[i * fold_size : (i + 1) * fold_size]
            train_indices = [idx for idx in indices if idx not in val_indices]
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
            folds.append(
                (
                    DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                    DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
                )
            )

        return folds

    def get_timepoints(self, idx: int) -> int:
        """Returns timepoint of the image at the given index"""
        group = self.groups[idx]
        file_name = group["cancer"]
        timepoint = int(file_name.split("_")[3])
        return timepoint

    @property
    def n_train(self) -> int:
        """Number of training points"""
        return len(self._train_indices)

    @property
    def n_test(self) -> int:
        """Number of test points"""
        return len(self._test_indices)

    def _get_image_groups(self) -> None:
        """Returns groups of images based on the filename format."""
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

        if self.writer:
            self.writer.write_indices(self._train_indices, self._test_indices)

    def _load_from_indices(self, indices_file: str) -> None:
        with open(indices_file, "r", encoding="utf-8") as i_file:
            indices = json.load(i_file)
        self._train_indices = indices["train"]
        self._test_indices = indices["test"]

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
