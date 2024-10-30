import os
import json
from typing import Optional, Any
from collections import defaultdict

import torch

from src.simulation_encoder.loaders.loader import Loader
from src.simulation_encoder.logger import Logger

class ARCADELoader(Loader):
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
        channels: list[str],
        name: Optional[str] = None,
        batch_size: int = 10,
        val_split: float = 0.2,
        test_split: float = 0.2,
        labels: list[str] = [],
        label_dir: Optional[str] = None,
        logger: Optional[Logger] = None,
        augmentations: Optional[dict[str, Any]] = None,
        indices_file: Optional[str] = None,
        random_seed: int = 42,
    ):

        super().__init__(
            channels=channels,
            val_split=val_split,
            test_split=test_split,
            batch_size=batch_size,
            indices_file=indices_file,
            random_seed=random_seed,
        )

        self.image_dir = image_dir
        self.name = name
        self.keys = keys
        self.labels = labels
        self.logger = logger

        self.label_loader = LabelLoader(label_dir) if label_dir else None
        self.augmentations: dict[str, Augmentation] = self._get_augmentations(augmentations) or {}

        self._get_image_groups()
        self._split_data()
        self._augment_training_data()

    def get_labels(self, label: str, dataset_type: str) -> torch.Tensor:
        """Returns labels for the specified dataset type (train, val, test)"""
        indices_attr = f"_{dataset_type}_indices"
        if not hasattr(self, indices_attr):
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        indices = getattr(self, indices_attr)
        try:
            labels = [self.get_label(idx, label) for idx in indices]
            return torch.tensor(labels, requires_grad=False)
        except ValueError:
            raise ValueError(f"Invalid target name: {label}")

    def get_label(self, idx: int, label_name: str) -> float:
        """Returns labels the group at index `idx`"""
        if label_name in self.labels:
            label = self.groups[idx]["labels"][label_name]
            return 0.0 if label == "nan" else label

        raise ValueError(f"Invalid label name: {label_name}")

    def _get_image_groups(self) -> None:
        """Returns groups of images based on the filename format."""
        groups: dict[str, Any] = defaultdict(
            lambda: {
                **{channel: "" for channel in self.channels},
                "timepoint": "",
                "seed_key": "",
                "augmentation": "original",
                "labels": defaultdict(float),
            }
        )

        for file_name in os.listdir(self.image_dir):
            if not file_name.endswith(".png") or not self._in_keys(file_name):
                continue
            # Skip images that are not in the specified channels
            if not any(channel in file_name for channel in self.channels):
                continue

            context, vasc_type, seed, timepoint, image_type = self._parse_ARCADE_filename(file_name)
            group_key = f"{context}_{vasc_type}_{seed}_{timepoint}"
            timepoint_short = str((timepoint // 10) - 1)
            groups[group_key]["timepoint"] = timepoint_short
            groups[group_key][image_type] = os.path.join(self.image_dir, file_name)
            groups[group_key]["seed_key"] = f"{context}_{vasc_type}_{seed}"

            if not self.labels:
                continue

            for label in self.labels:
                if label in groups[group_key]["labels"] or not self.label_loader:
                    continue
                key = f"{context}_{vasc_type}"
                label_upper = label.upper()
                timepoint_float = float(timepoint_short)
                groups[group_key]["labels"][label] = self.label_loader.get_labels(
                    label_upper, key, timepoint_float, seed
                )

        missing_images_count = {channel: 0 for channel in self.channels}
        for group_key, group in groups.items():
            for key, value in group.items():
                if value == "" and key in self.channels:
                    missing_images_count[key] += 1
        for channel, count in missing_images_count.items():
            if count > 0:
                self._log(f"Number of missing images in {channel} - {count}", "warning")

        self.groups = list(groups.values())

    def _parse_ARCADE_filename(self, filename: str) -> tuple[str, str, int, int, str]:
        parts = filename.split("_")
        context = parts[0]  # 'CH' for healthy tissue or 'C' colony
        vasc_type = parts[1]
        seed = int(parts[2])
        timepoint = int(parts[3])
        for part in parts[4:]:
            for channel in self.channels:
                if channel in part:
                    image_type = channel
                    return context, vasc_type, seed, timepoint, image_type

        raise ValueError(
            f"Invalid name format for image file. Should be \
        'context_vasc-type_seed_timepoint_image-type.png' Got: {filename}"
        )

    def _in_keys(self, file_name: str) -> bool:
        file_chunks = file_name.split("_")[0:2]
        prefix = "_".join(file_chunks)
        return prefix in self.keys