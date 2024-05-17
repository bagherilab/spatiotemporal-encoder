import os
import json
from typing import Optional, Callable, Any
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from simulation_encoder.logger import ExperimentLogger


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
    label_dir : str
        Path to the directory containing the labels.
    val_split : float, optional
        Fraction of the data to use for validation, by default 0.2
    test_split : float, optional
        Fraction of the data to use for testing, by default 0.2
    batch_size : int, optional
        Batch size for the DataLoader, by default 10
    logger : Logger, optional
        Logger object for logging missing images, by default None
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
        label_dir: Optional[str] = None,
        val_split: float = 0.2,
        test_split: float = 0.2,
        batch_size: int = 10,
        logger: Optional[ExperimentLogger] = None,
        augmentations: Optional[dict[str, Any]] = None,
        indices_file: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.image_dir = image_dir
        self.keys = keys
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.logger = logger
        self.indices_file = indices_file
        self.random_seed = random_seed

        self.label_loader = LabelLoader(label_dir) if label_dir else None
        self.augmentations: dict[str, Augmentation] = self._get_augmentations(augmentations) or {}

        self.metrics = ["activity", "growth", "symmetry"]

        self._get_image_groups()
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
        return len(self._train_indices) + len(self._val_indices)

    @property
    def n_test(self) -> int:
        """Number of test points"""
        return len(self._test_indices)

    @property
    def n_channels(self) -> int:
        """Number of channels in the images"""
        return self[0][0].shape[0]

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Shape of the images"""
        return tuple(self[0][0].shape[1:])

    def get_train_dataloader(self) -> DataLoader:
        """Returns training DataLoader"""
        train_dataset = Subset(self, self._train_indices)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def get_val_dataloader(self) -> DataLoader:
        """Returnsvalidation DataLoader"""
        val_dataset = Subset(self, self._val_indices)
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def get_test_dataloader(self) -> DataLoader:
        """Returns test DataLoader"""
        test_dataset = Subset(self, self._test_indices)
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
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

    def get_metric(self, idx: int, label_name: str) -> float:
        """Returns metric the group at index `idx`"""
        if label_name in self.metrics:
            metric = self.groups[idx]["metrics"][label_name]
            return 0.0 if metric == "nan" else metric

        raise ValueError(f"Invalid label name: {label_name}")

    def get_seed_key(self, idx: int) -> str:
        """Returns the seed and key of the group at index `idx`"""
        return self.groups[idx]["seed_key"]

    def get_indices(self) -> tuple[list[int], list[int], list[int]]:
        """Returns the train, validation, and test indices"""
        return self._train_indices, self._val_indices, self._test_indices

    def _get_image_groups(self) -> None:
        """Returns groups of images based on the filename format."""
        groups: dict[str, Any] = defaultdict(
            lambda: {
                "cancer": "",
                "graph": "",
                "timepoint": "",
                "seed_key": "",
                "augmentation": "original",
                "metrics": defaultdict(float),
            }
        )

        for file_name in os.listdir(self.image_dir):
            if not file_name.endswith(".png") or not self._in_keys(file_name):
                continue

            context, vasc_type, seed, timepoint, image_type = self._parse_ARCADE_filename(file_name)
            group_key = f"{context}_{vasc_type}_{seed}_{timepoint}"
            timepoint_short = str((timepoint // 10) - 1)
            groups[group_key]["timepoint"] = timepoint_short
            groups[group_key][image_type] = os.path.join(self.image_dir, file_name)
            groups[group_key]["seed_key"] = f"{context}_{vasc_type}_{seed}"

            for metric in self.metrics:
                if metric in groups[group_key]["metrics"] or not self.label_loader:
                    continue
                key = f"{context}_{vasc_type}"
                metric_upper = metric.upper()
                timepoint_float = float(timepoint_short)
                groups[group_key]["metrics"][metric] = self.label_loader.get_metrics(
                    metric_upper, key, timepoint_float, seed
                )

        if self.logger:
            missing_images = {
                group_key: [key for key, value in group.items() if value == ""]
                for group_key, group in groups.items()
                if "" in group.values()
            }
            for group_key, missing_list in missing_images.items():
                self.logger.warning(f"Missing images from {group_key}: {missing_list}")

        self.groups = list(groups.values())

    def _get_augmentations(
        self, augmentations: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Augmentation]]:
        if not augmentations:
            return None
        augmentation_map: dict[str, Callable[..., Any]] = {
            "rotate": lambda degree: transforms.RandomRotation(degrees=degree),
        }

        augmentations_dict = {}
        for aug_name, args in augmentations.items():
            args = args.values()
            if aug_name not in augmentation_map:
                raise ValueError(f"Invalid augmentation name: {aug_name}")

            transform = augmentation_map[aug_name](*map(int, args))
            augmentations_dict[aug_name] = Augmentation(transform, aug_name)

        return augmentations_dict

    def _augment_training_data(self) -> None:
        if not self.augmentations:
            return

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
        if self.indices_file:
            self._load_from_indices(self.indices_file)
            return

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

        self._train_indices = [index for key in train_group_keys for index in groups[key]]
        self._val_indices = [index for key in val_group_keys for index in groups[key]]
        self._test_indices = [index for key in test_group_keys for index in groups[key]]

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
        images_stack = torch.stack(images)
        labels_tensor = torch.tensor(labels)
        return images_stack, labels_tensor


class LabelLoader:
    """
    Loads metrics (activity, growth, and symmetry) to match with corresponding images.

    Parameters
    ----------
    label_dir : str
        Path to the directory containing the labels.

    """

    def __init__(self, label_dir: str):
        self.label_dir = label_dir

    def get_metrics(self, metric: str, key: str, time: float, seed: int) -> float:
        file = f"{self.label_dir}/VASCULAR_FUNCTION_{key}.SEEDS.{metric}.json"

        with open(file, "r", encoding="utf-8") as f:
            vals = json.load(f)
        for val in vals:
            if val["time"] == time:
                return val["_"][seed]

        return float("nan")
