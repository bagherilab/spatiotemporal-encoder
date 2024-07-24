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
        random_seed: int = 42,
    ):
        self.channels = channels
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.indices_file = indices_file
        self.random_seed = random_seed

        self._train_indices: list[int] = []
        self._val_indices: list[int] = []
        self._test_indices: list[int] = []

        self.groups: list[dict[str, Any]] = []
        self.augmentations: dict[str, Augmentation] = {}

        self.images = self._preload_images()

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        group = self.groups[idx]
        timepoint = int(group["timepoint"])
        image_tensor = self.images[idx]

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
                transforms.Lambda(lambda x: x[:3]),
                transforms.Grayscale(num_output_channels=1),
                transforms.Lambda(lambda x: x.squeeze()),
            ]
        )

        tensors = []
        for channel in channels:
            tensors.append(transformation(Image.open(group[channel])))

        full_tensor = torch.stack(tensors, dim=0)

        augmentation_name = group["augmentation"]
        if augmentation_name == "original":
            return full_tensor

        augmentation = self.augmentations[augmentation_name]
        return augmentation(full_tensor)
    
    def _preload_images(self) -> dict[int, torch.Tensor]:
        images = {}
        for idx, group in enumerate(self.groups):
            for image_idx in group['indices']:
                images[image_idx] = self._get_image_tensors(group, self.channels)
        return images

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

        flat_input = torch.cat(flat_input, dim=0)
        labels = torch.cat(labels, dim=0)

        dataset = TensorDataset(flat_input, labels)
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
        
        transformed_data = torch.cat(transformed_data, dim=0)
        labels = torch.cat(labels, dim=0)

        dataset = TensorDataset(transformed_data, labels)
        return DataLoader(
            dataset, batch_size=data_loader.batch_size, shuffle=True, collate_fn=Loader._collate_fn
        )

    @staticmethod
    def _collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = zip(*batch)
        images_stack = torch.stack(images)
        labels_tensor = torch.tensor(labels)
        return images_stack, labels_tensor


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
        labels: list[str] = [],
        label_dir: Optional[str] = None,
        val_split: float = 0.2,
        test_split: float = 0.2,
        batch_size: int = 10,
        logger: Optional[ExperimentLogger] = None,
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

        if self.logger:
            missing_images = {
                group_key: [
                    key for key, value in group.items() if value == "" and key in self.channels
                ]
                for group_key, group in groups.items()
                if any(value == "" and key in self.channels for key, value in group.items())
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

            full_aug_name = f"{aug_name}_{'_'.join(map(str, args))}"
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


class AlphaNumericLoader(Loader):
    """
    Loader class for loading
    labeled images from a directory.
    """

    def __init__(
        self,
        image_dir: str,
        keys: list[str],
        channels: list[str],
        val_split: float = 0.2,
        test_split: float = 0.2,
        batch_size: int = 10,
        logger: Optional[ExperimentLogger] = None,
        indices_file: Optional[str] = None,
        augmentations: Optional[dict[str, Any]] = None,
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
        self.keys = keys
        self.logger = logger

        self._get_image_groups()
        self._split_data()

    def _get_image_groups(self) -> None:
        """Returns groups of images based on the filename format."""
        groups: dict[str, Any] = defaultdict(
            lambda: {
                "image": "",
                "character": "",
                "angle": "",
                "timepoint": "",
                "seed_key": "",
                "augmentation": "original",
            }
        )

        for file_name in os.listdir(self.image_dir):
            if not file_name.endswith(".png") or not self._in_keys(file_name):
                continue

            character, seed, angle, timepoint = self._parse_alphanumeric_filename(file_name)
            group_key = f"{character}_{seed}_{angle}_{timepoint}"
            groups[group_key]["image"] = os.path.join(self.image_dir, file_name)
            groups[group_key]["character"] = character
            groups[group_key]["angle"] = angle
            groups[group_key]["timepoint"] = timepoint
            groups[group_key]["seed_key"] = f"{character}_{seed}_{angle}"

        self.groups = list(groups.values())

    def _parse_alphanumeric_filename(self, filename: str) -> tuple[str, int, int, int]:
        parts = filename.split("_")
        character = parts[0]
        seed = int(parts[1])
        angle = int(parts[3])
        timepoint = int(parts[5].split(".")[0])

        return character, seed, angle, timepoint

    def _in_keys(self, file_name: str) -> bool:
        file_chunks = file_name.split("_")[0:3]
        prefix = file_chunks[0]
        return prefix in self.keys


class CSVLoader(Dataset):
    """
    Loader class for loading labeled data from a CSV file.

    Parameters
    ----------
    exp_id : str
        Experiment ID.
    labels : list[str]
        List of target labels.
    """

    def __init__(self, conf_name: str, exp_id: str, model: str, labels: list[str]) -> None:
        self.exp_id = exp_id
        self.data_path = f"results/{conf_name}/{exp_id}/{model}/"
        self.labels = labels
        self._load_data()

    def __len__(self) -> int:
        return len(self._X_train) + len(self._X_val) + len(self._X_test)

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("This method is not implemented yet. Use get_dataloader instead.")

    def get_data(
        self,
        dataset_type: str,
        feature_timepoint: Optional[int] = None,
        response_timepoint: Optional[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if dataset_type == "train":
            X, y = self._X_train, self._y_train
        elif dataset_type == "val":
            X, y = self._X_val, self._y_val
        elif dataset_type == "test":
            X, y = self._X_test, self._y_test
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        if feature_timepoint:
            X = X[X["timepoint"] == feature_timepoint]

        if response_timepoint:
            y = y[y["timepoint"] == response_timepoint]

        # Align based on seed_key
        if feature_timepoint is not None or response_timepoint is not None:
            common_seed_keys = set(X["seed_key"]).intersection(y["seed_key"])
            X = X[X["seed_key"].isin(common_seed_keys)]
            y = y[y["seed_key"].isin(common_seed_keys)]

            # Sort by seed_key to ensure alignment
            X = X.sort_values("seed_key").reset_index(drop=True)
            y = y.sort_values("seed_key").reset_index(drop=True)

        return X.drop(["timepoint", "seed_key"], axis=1), y.drop(["timepoint", "seed_key"], axis=1)

    def _load_data(self) -> None:
        self._X_train, self._y_train = self._load_csv("train")
        self._X_val, self._y_val = self._load_csv("val")
        self._X_test, self._y_test = self._load_csv("test")

    def _load_csv(self, dataset_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        file_path = os.path.join(self.data_path, f"encoded_data/{dataset_type}.csv")
        data = pd.read_csv(file_path)
        self.feature_cols = [col for col in data.columns if col.startswith("dim_")]
        X, y = (
            data.loc[:, self.feature_cols + ["timepoint", "seed_key"]],
            data.loc[:, self.labels + ["timepoint", "seed_key"]],
        )
        return X, y


class LabelLoader:
    """
    Loads labels (activity, growth, and symmetry) to match with corresponding images.

    Parameters
    ----------
    label_dir : str
        Path to the directory containing the labels.

    """

    def __init__(self, label_dir: str):
        self.label_dir = label_dir

    def get_labels(self, label: str, key: str, time: float, seed: int) -> float:
        file = f"{self.label_dir}/VASCULAR_FUNCTION_{key}.SEEDS.{label}.json"

        with open(file, "r", encoding="utf-8") as f:
            vals = json.load(f)
        for val in vals:
            if val["time"] == time:
                return val["_"][seed]

        return float("nan")
