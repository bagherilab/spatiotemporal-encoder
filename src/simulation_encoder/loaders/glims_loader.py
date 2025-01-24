import os
from typing import Optional, Any
from collections import defaultdict

from simulation_encoder.logger import Logger
from simulation_encoder.loaders.loader import Loader, Augmentation


class GlimsLoader(Loader):
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
        batch_size: int = 10,
        val_split: float = 0.2,
        test_split: float = 0.2,
        logger: Optional[Logger] = None,
        augmentations: Optional[list[dict[str, Any]]] = None,
        indices_file: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.name = name
        self.labels = None

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

    def _get_image_groups(self) -> None:
        """Returns groups of images based on the filename format."""
        groups: dict[str, Any] = defaultdict(
            lambda: {
                "image": "",
                "diffusivity": "",
                "proliferation": "",
                "timepoint": "",
                "seed_key": "",
                "augmentation": {"original": ""},
            }
        )

        for file_name in os.listdir(self.image_dir):
            if not file_name.endswith(".png") or not self._in_keys(file_name):
                continue

            diff, prolif, timepoint = self._parse_filename(file_name)

            group_key = f"{diff}_{prolif}_{timepoint}"
            groups[group_key]["image"] = os.path.join(self.image_dir, file_name)
            groups[group_key]["diffusivity"] = diff
            groups[group_key]["proliferation"] = prolif
            groups[group_key]["timepoint"] = timepoint
            groups[group_key]["seed_key"] = f"{prolif}_{diff}"

        self.groups = list(groups.values())

    def _parse_filename(self, filename: str) -> tuple[float, float, int]:
        parts = filename.split("_")
        diff = float(parts[1]) / 10**5
        prolif = float(parts[3]) / 10**5
        timepoint = int(parts[4].split(".")[0])

        return diff, prolif, timepoint

    def _in_keys(self, file_name: str) -> bool:
        file_chunks = file_name.split("_")[0:3]
        prefix = file_chunks[0]
        return prefix in self.keys
