import os
from collections import defaultdict
from typing import Any

import torch

from simulation_encoder.loaders.dataset_utils.label_loaders import LabelLoader
from simulation_encoder.loaders.loader import Loader
from simulation_encoder.logger import Logger


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
        name: str | None = None,
        batch_size: int = 16,
        val_split: float = 0.2,
        test_split: float = 0.2,
        labels: list[str] = [],
        label_dir: str | None = None,
        logger: Logger | None = None,
        augmentations: list[dict[str, Any]] | None = [],
        indices_file: str | None = None,
        random_seed: int = 42,
    ):
        self.name = name
        self.labels = labels

        super().__init__(
            image_dir=image_dir,
            keys=keys,
            channels=channels,
            val_split=val_split,
            test_split=test_split,
            batch_size=batch_size,
            augmentations=augmentations,
            indices_file=indices_file,
            logger=logger,
            random_seed=random_seed,
        )

        self.label_loader = LabelLoader(label_dir) if label_dir else None

    def get_labels(self, label: str, dataset_type: str) -> torch.Tensor:
        """Returns labels for the specified dataset type (train, val, test)"""
        if dataset_type == "train":
            indices = self._train_indices
        elif dataset_type == "val":
            indices = self._val_indices
        elif dataset_type == "test":
            indices = self._test_indices
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

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

    def _retrieve_data(self) -> list[dict[str, Any]]:
        """Returns groups of images based on the filename format."""
        image_groups: dict[str, Any] = defaultdict(
            lambda: {
                **{channel: "" for channel in self.channels},
                "timepoint": "",
                "sample_id": "",
                "augmentation": {"identity": ""},
                "labels": defaultdict(float),
            }
        )

        for file_name in os.listdir(self.image_dir):
            if not file_name.endswith(".png") or not self._in_keys(file_name):
                continue

            if not any(channel in file_name for channel in self.channels):
                continue

            context, vasc_type, seed, timepoint, image_type = self._parse_filename(
                file_name
            )
            sample_id = f"{context}_{vasc_type}_{seed}"
            simulation_id = f"{sample_id}_{timepoint}"
            timepoint_day = str((timepoint // 10) - 1)

            group = image_groups[simulation_id]
            group["timepoint"] = timepoint_day
            group["sample_id"] = sample_id
            group[image_type] = os.path.join(self.image_dir, file_name)

            if self.labels and self.label_loader:
                for label in self.labels:
                    if label not in group["labels"]:
                        key = f"{context}_{vasc_type}"
                        group["labels"][label] = self.label_loader.get_labels(
                            label.upper(), key, float(timepoint_day), seed
                        )

        self._log_missing_images(image_groups)
        return list(image_groups.values())

    def _parse_filename(self, filename: str) -> tuple[str, str, int, int, str]:
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
