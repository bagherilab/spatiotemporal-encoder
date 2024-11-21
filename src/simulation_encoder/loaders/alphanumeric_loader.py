import os
from typing import Optional, Any
from collections import defaultdict

from simulation_encoder.logger import Logger
from simulation_encoder.loaders.loader import Loader, Augmentation


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
        logger: Optional[Logger] = None,
        indices_file: Optional[str] = None,
        augmentations: Optional[dict[str, Any]] = None,
        random_seed: int = 42,
    ):
        self.labels = None
        self.logger = logger
        super().__init__(
            image_dir=image_dir,
            keys=keys,
            channels=channels,
            val_split=val_split,
            test_split=test_split,
            batch_size=batch_size,
            augmentations=augmentations,
            indices_file=indices_file,
            random_seed=random_seed,
        )

        self.augmentations: dict[str, Augmentation] = self._get_augmentations(augmentations) or {}

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
