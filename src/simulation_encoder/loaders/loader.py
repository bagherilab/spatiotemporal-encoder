import os
import json
from typing import Optional, Callable, Any
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import transforms

from src.simulation_encoder.logger import Logger


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


class Loader(ABC, Dataset):
    """
    Abstract class for data loaders.
    """

    def __init__(
        self,
        channels: list[str],
        val_split: float = 0.2,
        test_split: float = 0.2,
        batch_size: int = 10,
        indices_file: Optional[str] = None,
        logger: Optional[Logger] = None,
        random_seed: int = 42,
    ):
        self.channels = channels
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.indices_file = indices_file
        self.logger = logger
        self.random_seed = random_seed

        self._train_indices: list[int] = []
        self._val_indices: list[int] = []
        self._test_indices: list[int] = []

        self.groups: list[dict[str, Any]] = []
        self.augmentations: dict[str, Augmentation] = {}

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        group = self.groups[idx]
        timepoint = int(group["timepoint"])
        image_tensor = self._get_image_tensors(group, self.channels)

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
        return len(self.channels)

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Shape of the images"""
        return Image.open(self.groups[0][self.channels[0]]).size

    def get_dataloader(self, dataset_type: str) -> DataLoader:
        """Returns DataLoader for the specified dataset type (train, val, test)"""
        indices_attr = f"_{dataset_type}_indices"
        if not hasattr(self, indices_attr):
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        indices = getattr(self, indices_attr)
        dataset = Subset(self, indices)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(dataset_type == "train"),  # Shuffle only for training data
            collate_fn=Loader._collate_fn,
        )

    def get_indices(self) -> tuple[list[int], list[int], list[int]]:
        """Returns the train, validation, and test indices"""
        return self._train_indices, self._val_indices, self._test_indices

    def get_group_feature(self, idx: int, feature: str) -> str:
        """Returns a given feature of the group at index `idx`"""
        return self.groups[idx][feature]

    def get_timepoints(self, dataset_type: str) -> torch.Tensor:
        """Returns timepoints for the specified dataset type (train, val, test)"""
        indices_attr = f"_{dataset_type}_indices"
        if not hasattr(self, indices_attr):
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        indices = getattr(self, indices_attr)
        timepoints = [int(self.get_group_feature(idx, "timepoint")) for idx in indices]
        return torch.tensor(timepoints, requires_grad=False)

    def get_seed_keys(self, dataset_type: str) -> list[str]:
        """Returns seed keys for the specified dataset type (train, val, test)"""
        indices_attr = f"_{dataset_type}_indices"
        if not hasattr(self, indices_attr):
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        indices = getattr(self, indices_attr)
        seed_keys = [self.get_group_feature(idx, "seed_key") for idx in indices]
        return seed_keys

    def _get_image_tensors(self, group: dict, channels: list) -> torch.Tensor:
        transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: x[:3]),
                transforms.Grayscale(num_output_channels=1),
                transforms.Lambda(lambda x: x.squeeze()),
                transforms.Lambda(lambda x: x / x.max())
            ]
        )

        tensors = []
        for channel in channels:
            if group[channel] != "":
                tensors.append(transformation(Image.open(group[channel])))
            else:
                zero_channel = Image.fromarray(np.zeros(self.image_shape, dtype=np.uint8))
                tensors.append(transformation(zero_channel))

        full_tensor = torch.stack(tensors, dim=0)

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

    def _load_from_indices(self, indices_file: str) -> None:
        if not os.path.exists(indices_file):
            raise FileNotFoundError(f"Indices file not found: {indices_file}")

        with open(indices_file, "r", encoding="utf-8") as i_file:
            indices = json.load(i_file)
        self._train_indices = indices["train"]
        self._val_indices = indices["val"]
        self._test_indices = indices["test"]

    def _get_indices_of_groups(self, indices: list[int]) -> dict[str, list[int]]:
        groups = defaultdict(list)
        for idx in indices:
            group = self.groups[idx]
            key = group["seed_key"]
            groups[key].append(idx)
        return groups
    
    def _get_augmentations(
        self, augmentations: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Augmentation]]:
        if not augmentations:
            return None

        augmentation_map: dict[str, Callable[..., Any]] = {
            "rotate": lambda degree: transforms.RandomRotation(degrees=degree),
        }

        augmentations_dict = {}
        for augmentation in augmentations:
            aug_name, arg = augmentation.popitem()
            if aug_name not in augmentation_map:
                raise ValueError(f"Invalid augmentation name: {aug_name}")

            transform = augmentation_map[aug_name](int(arg))

            full_aug_name = f"{aug_name}_{arg}"
            augmentations_dict[full_aug_name] = Augmentation(transform, aug_name)

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

    def _log(self, msg: str, level: str = "info") -> None:
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            else:
                self.logger.log(msg)

    @staticmethod
    def _subsample_loader(data_loader: DataLoader, frac: float) -> DataLoader:
        if frac >= 1.0:
            return data_loader

        total_num_samples = len(data_loader.dataset)  # type: ignore
        num_samples = int(total_num_samples * frac)
        indices = np.random.choice(total_num_samples, num_samples, replace=False).tolist()
        dataset = Subset(data_loader.dataset, indices)
        return DataLoader(
            dataset, batch_size=data_loader.batch_size, shuffle=True, collate_fn=Loader._collate_fn
        )

    @staticmethod
    def _flatten_loader(data_loader: DataLoader) -> DataLoader:
        flat_input = []
        labels = []
        batch_size = data_loader.batch_size

        for data in data_loader:
            features, targets = data
            flat_features = features.view(features.size(0), -1)
            flat_input.append(flat_features)
            labels.append(targets)

        flat_input_tensor = torch.cat(flat_input, dim=0)  # type: ignore
        labels_tensor = torch.cat(labels, dim=0)  # type: ignore

        dataset = TensorDataset(flat_input_tensor, labels_tensor)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=Loader._collate_fn
        )

    @staticmethod
    def _transform_dataloader(
        func: Callable[[torch.Tensor], torch.Tensor], data_loader: DataLoader, device: str = "cpu"
    ) -> DataLoader:
        transformed_data = []
        labels = []
        for feature, label in data_loader:
            feature = feature.to(device)
            transformed_features = func(feature)
            transformed_data.append(transformed_features)
            labels.append(label)

        transformed_data_tensor = torch.cat(transformed_data, dim=0)  # type: ignore
        labels_tensor = torch.cat(labels, dim=0)  # type: ignore

        dataset = TensorDataset(transformed_data_tensor, labels_tensor)
        return DataLoader(
            dataset, batch_size=data_loader.batch_size, shuffle=True, collate_fn=Loader._collate_fn
        )

    @staticmethod
    def _collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = zip(*batch)
        images_stack = torch.stack(images)
        labels_tensor = torch.tensor(labels)
        return images_stack, labels_tensor