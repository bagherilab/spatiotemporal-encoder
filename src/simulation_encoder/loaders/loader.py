import os
import json
from typing import Optional, Callable, Any
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from simulation_encoder.logger import Logger
from simulation_encoder.loaders.dataset_utils.augmentation import AugmentationsManager
from simulation_encoder.loaders.dataset_utils.data_splitter import DatasetSplitter
from simulation_encoder.loaders.dataset_utils.image_dataset import ImageDataset


class Loader(ABC):
    """
    Abstract class for data loaders.
    """

    def __init__(
        self,
        image_dir: str,
        keys: list[str],
        channels: list[str],
        val_split: float,
        test_split: float,
        batch_size: int,
        augmentations: Optional[list[dict[str, Any]]],
        indices_file: Optional[str],
        logger: Optional[Logger],
        random_seed: int,
    ):
        self.image_dir = image_dir
        self.keys = keys
        self.channels = channels
        self.batch_size = batch_size
        self.logger = logger

        self.augmentation_manager = AugmentationsManager(augmentations)
        self.data = self._retrieve_data()

        if indices_file:
            self._train_indices, self._val_indices, self._test_indices = self._load_from_indices(
                indices_file
            )
            self._reconstruct_augmented_groups()
        else:
            splitter = DatasetSplitter(self.data, val_split, test_split, random_seed)
            self._train_indices, self._val_indices, self._test_indices = splitter.get_splits()

        self._augment_training_data()

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

    @abstractmethod
    def _retrieve_data(self) -> list[dict[str, Any]]:
        """Returns groups of images based on the filename format."""
        raise NotImplementedError("Implement this in a subclass.")

    def get_dataloader(self, dataset_type: str) -> DataLoader:
        """Returns DataLoader for the specified dataset type (train, val, test)"""
        indices = {
            "train": self._train_indices,
            "val": self._val_indices,
            "test": self._test_indices,
        }.get(dataset_type)

        if indices is None:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        dataset = ImageDataset(
            self.image_dir, self.data, self.channels, self.augmentation_manager, indices
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(dataset_type == "train"),  # Shuffle only for training data
            collate_fn=Loader._collate_fn,
        )

    def get_indices(self) -> tuple[list[int], list[int], list[int]]:
        """Returns the train, validation, and test indices"""
        return self._train_indices, self._val_indices, self._test_indices

    def get_timepoints(self, dataset_type: str) -> torch.Tensor:
        """Returns timepoints for the specified dataset type (train, val, test)"""
        indices_attr = f"_{dataset_type}_indices"
        if not hasattr(self, indices_attr):
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        indices = getattr(self, indices_attr)
        timepoints = [int(self._get_data_feature(idx, "timepoint")) for idx in indices]
        return torch.tensor(timepoints, requires_grad=False)

    def get_sample_ids(self, dataset_type: str) -> list[str]:
        """Returns sample IDs for the specified dataset type (train, val, test)"""
        indices_attr = f"_{dataset_type}_indices"
        if not hasattr(self, indices_attr):
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        indices = getattr(self, indices_attr)
        sample_ids = [self._get_data_feature(idx, "sample_id") for idx in indices]
        return sample_ids

    def _augment_training_data(self) -> None:
        augmented_groups = []
        for transform_dict in self.augmentation_manager.transforms:
            ((aug_name, aug),) = transform_dict.items()
            if aug_name == "identity":
                continue

            for index in self._train_indices:
                group = self.data[index]
                aug_group = dict(group)
                aug_group["augmentation"] = {aug_name: aug}
                aug_group["original_index"] = index
                augmented_groups.append(aug_group)

        start_idx = len(self.data)
        self.data.extend(augmented_groups)

        new_indices = list(range(start_idx, len(self.data)))
        self._train_indices.extend(new_indices)

    def _reconstruct_augmented_groups(self) -> None:
        original_length = len(self.data)
        original_train_indices = [idx for idx in self._train_indices if idx < original_length]

        augmented_groups = []
        for transform_dict in self.augmentation_manager.transforms:
            ((aug_name, aug),) = transform_dict.items()
            if aug_name == "identity":
                continue

            for index in original_train_indices:
                original_group = self.data[index]
                aug_group = dict(original_group.items())
                aug_group["augmentation"] = {aug_name: aug}
                augmented_groups.append(aug_group)
        self.data.extend(augmented_groups)

    def _load_from_indices(self, indices_file: str) -> tuple[list[int], ...]:
        if not os.path.exists(indices_file):
            raise FileNotFoundError(f"Indices file not found: {indices_file}")

        with open(indices_file, "r", encoding="utf-8") as f:
            indices = json.load(f)
        return indices["train"], indices["val"], indices["test"]

    def _get_data_feature(self, idx: int, feature: str) -> str:
        """Returns a given feature of the data at index `idx`"""
        return self.data[idx][feature]

    def _log(self, msg: str, level: str = "info") -> None:
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            else:
                self.logger.log(msg)

    def _log_missing_images(self, image_groups: dict[str, Any]) -> None:
        """Log missing images for each channel."""
        missing_images_count = {channel: 0 for channel in self.channels}

        for group in image_groups.values():
            for channel in self.channels:
                if not group[channel]:
                    missing_images_count[channel] += 1

        for channel, count in missing_images_count.items():
            if count > 0:
                self._log(f"Number of missing images in {channel} - {count}", "warning")

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
