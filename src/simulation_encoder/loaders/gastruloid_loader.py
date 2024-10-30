import os
from typing import Optional, Any
from collections import defaultdict

from src.simulation_encoder.loaders.loader import Loader
from src.simulation_encoder.logger import Logger

class GastruloidLoader(Loader):
    """
    Loader class for loading
    labeled images from a directory.
    """

    def __init__(
        self,
        image_dir: str,
        keys: list[str],
        channels: list[str],
        name: Optional[str] = None,
        val_split: float = 0.2,
        test_split: float = 0.2,
        batch_size: int = 10,
        logger: Optional[Logger] = None,
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
        self.name = name
        self.keys = keys
        self.labels = None
        self.logger = logger

        self.augmentations: dict[str, Augmentation] = self._get_augmentations(augmentations) or {}
        self._get_image_groups()
        self._split_data()
        self._augment_training_data()

    def _get_image_groups(self) -> None:
        """Returns groups of images based on the filename format."""
        groups: dict[str, Any] = defaultdict(
            lambda: {
                **{channel: "" for channel in self.channels},
                "timepoint": "",
                "seed_key": "",
                "augmentation": "original",
            }
        )

        for file_name in os.listdir(self.image_dir):
            if not file_name.endswith(".png") or not self._in_keys(file_name):
                continue

            if not any(channel in file_name for channel in self.channels):
                continue

            modality, array, raft, timepoint = self._parse_gastruloid_filename(file_name)
            group_key = f"{modality}_{array}_{raft}_{timepoint}"
            groups[group_key]["timepoint"] = timepoint
            groups[group_key][modality] = os.path.join(self.image_dir, file_name)
            groups[group_key]["seed_key"] = f"{modality}_{array}_{raft}"

        self.groups = list(groups.values())

    def _parse_gastruloid_filename(self, filename: str) -> tuple[str, int, int]:
        parts = filename.split("_")
        modality = parts[0]
        array = parts[1]
        raft = int(parts[2])
        timepoint = int(parts[3].split(".")[0])

        return modality, array, raft, timepoint

    def _in_keys(self, file_name: str) -> bool:
        file_chunks = file_name.split("_")
        prefix = file_chunks[1]
        return prefix in self.keys
