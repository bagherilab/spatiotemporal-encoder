import os
import json
from typing import Optional, Callable
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from simulation_encoder.logger import ExperimentLogger
from simulation_encoder.writer import Writer


class Augmentation:
    """
    Class for image augmentation technique.

    Parameters
    ----------
    transform : Callable[[torch.Tensor], torch.Tensor]
        Transformation function to apply to the image tensor.
    name : str
        Name of the augmentation technique.
    """

    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor], name: str):
        self.transform = transform
        self.name = name

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:  # Single image tensor
            return self.transform(tensor)

        if tensor.ndim == 4:  # Stack of image tensors
            transformed_tensors = []
            for i in range(tensor.size(0)):
                transformed_tensors.append(self.transform(tensor[i]))
            return torch.stack(transformed_tensors, dim=0)

        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


class PNGLoader(Dataset):
    """
    Loader class for loading unlabeled images from a directory.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing the images.
    keys : list[str]
        List of keys to filter the images by.
    val_split : float, optional
        Fraction of the data to use for validation, by default 0.2
    test_split : float, optional
        Fraction of the data to use for testing, by default 0.2
    batch_size : int, optional
        Batch size for the DataLoader, by default 10
    logger : Logger, optional
        Logger object for logging missing images, by default None
    writer : Writer, optional
        Writer object for saving things to disk
    augmentations : list[Augmentation], optional
        List of augmentations to apply to the images, by default None
    indices_file : str, optional
        Path to a file containing train and test indices, by default None
    random_seed : int, optional
        Random seed for shuffling the data, by default 42

    """

    def __init__(
        self,
        image_dir: str,
        keys: list[str],
        val_split: float = 0.2,
        test_split: float = 0.2,
        batch_size: int = 10,
        logger: Optional[ExperimentLogger] = None,
        writer: Optional[Writer] = None,
        augmentations: Optional[list[Augmentation]] = None,
        indices_file: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.image_dir = image_dir
        self.keys = keys
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.logger = logger
        self.writer = writer
        self.augmentations = {aug.name: aug for aug in (augmentations or [])}
        self.indices_file = indices_file
        self.random_seed = random_seed

        self._get_image_groups()

        if indices_file:
            self._load_from_indices(indices_file)
        else:
            self._split_data()

        self._augment_training_data()

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        group = self.groups[idx]
        timepoint = int(group["timepoint"])
        image_tensor = self._get_image_tensors(group)

        return image_tensor, timepoint

    @property
    def n_train(self) -> int:
        """Number of training points"""
        return len(self._train_indices)

    @property
    def n_test(self) -> int:
        """Number of test points"""
        return len(self._test_indices)

    @property
    def n_channels(self) -> int:
        """Number of channels in the images"""
        return self[0][0].shape[0]

    @property
    def image_shape(self) -> tuple[int, int]:
        """Shape of the images"""
        return self[0][0].shape[1:]

    def get_train_dataloader(self) -> DataLoader:
        """Returns training DataLoader"""
        train_dataset = Subset(self, self._train_indices)
        return DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn
        )

    def get_val_split(self) -> tuple[DataLoader, DataLoader]:
        """Returnsvalidation DataLoader"""
        val_dataset = Subset(self, self._val_indices)
        return DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn
        )

    def get_test_dataloader(self) -> DataLoader:
        """Returns test DataLoader"""
        test_dataset = Subset(self, self._test_indices)
        return DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn
        )

    def get_cv_splits(self, k_folds: int) -> list[tuple[DataLoader, DataLoader]]:
        """Returns list of k-folds of training and validation DataLoader"""
        indices = self._train_indices + self._val_indices
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
                    DataLoader(
                        train_dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        collate_fn=self._collate_fn,
                    ),
                    DataLoader(
                        val_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        collate_fn=self._collate_fn,
                    ),
                )
            )

        return folds

    def get_timepoint(self, idx: int) -> int:
        """Returns the timepoint of the group at index `idx`"""
        return int(self.groups[idx]["timepoint"])

    def _get_image_groups(self) -> None:
        """Returns groups of images based on the filename format."""
        groups: dict[str, dict[str, str]] = defaultdict(
            lambda: {"cancer": "", "graph": "", "timepoint": "", "augmentation": "original"}
        )

        for file_name in os.listdir(self.image_dir):
            if file_name.endswith(".png") and self._in_keys(file_name):
                context, vasc_type, seed, timepoint, image_type = self._parse_ARCADE_filename(
                    file_name
                )
                group_key = f"{context}_{vasc_type}_{seed}_{timepoint}"
                timepoint_short = str((timepoint // 10) - 1)
                groups[group_key]["timepoint"] = timepoint_short
                groups[group_key][image_type] = os.path.join(self.image_dir, file_name)
                groups[group_key]["seed_key"] = f"{context}_{vasc_type}_{seed}"

        if self.logger:
            for group_key, group in groups.items():
                if "" in group.values():
                    missing_images = [key for key, value in group.items() if value == ""]
                    self.logger.warning(f"Missing images from {group_key}: {missing_images}")

        self.groups = list(groups.values())

    def _augment_training_data(self) -> None:
        augmented_groups = []
        for index in self._train_indices:
            original_group = self.groups[index]
            for aug_name, _ in self.augmentations.items():
                aug_group = dict(original_group.items())
                aug_group["augmentation"] = aug_name
                augmented_groups.append(aug_group)
        self.groups.extend(augmented_groups)
        self._train_indices.extend(
            range(len(self.groups) - len(augmented_groups), len(self.groups))
        )

    def _get_image_tensors(self, group: dict) -> torch.Tensor:
        transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[:3]),  # Remove the alpha channel
                transforms.Grayscale(num_output_channels=1),
                transforms.Lambda(lambda x: x.squeeze()),
            ]
        )

        cancer_tensor = transformation(Image.open(group["cancer"]))
        graph_tensor = transformation(Image.open(group["graph"]))
        full_tensor = torch.stack((cancer_tensor, graph_tensor), dim=0)

        augmentation_name = group["augmentation"]
        if augmentation_name == "original":
            return full_tensor

        augmentation = self.augmentations[augmentation_name]
        return augmentation(full_tensor)

    def _split_data(self, shuffle: bool = True) -> None:
        all_indices = list(range(len(self)))
        groups = self._get_indices_of_groups(all_indices)

        group_keys = list(groups.keys())
        if shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(group_keys)

        test_groups_count = int(np.floor(len(group_keys) * self.test_split))
        val_groups_count = int(np.floor(len(group_keys) * self.val_split))

        test_group_keys = group_keys[:test_groups_count]
        val_group_keys = group_keys[test_groups_count : test_groups_count + val_groups_count]
        train_group_keys = group_keys[test_groups_count + val_groups_count :]

        train_indices = []
        val_indices = []
        test_indices = []

        for key in train_group_keys:
            train_indices.extend(groups[key])
        for key in val_group_keys:
            val_indices.extend(groups[key])
        for key in test_group_keys:
            test_indices.extend(groups[key])

        self._train_indices = train_indices
        self._val_indices = val_indices
        self._test_indices = test_indices

        if self.writer:
            self.writer.write_indices(self._train_indices, self._val_indices, self._test_indices)

    def _parse_ARCADE_filename(self, filename: str) -> tuple[str, str, int, int, str]:
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

    def _get_indices_of_groups(self, indices: list[int]) -> dict[str, list[int]]:
        groups = defaultdict(list)
        for idx in indices:
            group = self.groups[idx]
            key = group["seed_key"]
            groups[key].append(idx)
        return groups

    def _load_from_indices(self, indices_file: str) -> None:
        if not os.path.exists(indices_file):
            raise FileNotFoundError(f"Indices file not found: {indices_file}")

        with open(indices_file, "r", encoding="utf-8") as i_file:
            indices = json.load(i_file)
        self._train_indices = indices["train"]
        self._val_indices = indices["val"]
        self._test_indices = indices["test"]

    def _in_keys(self, file_name: str) -> bool:
        file_chunks = file_name.split("_")[0:2]
        prefix = "_".join(file_chunks)
        return prefix in self.keys

    def _collate_fn(
        self, batch: list[tuple[torch.Tensor, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels
