import os
from typing import Optional, Any
from collections import defaultdict

from simulation_encoder.logger import Logger
from simulation_encoder.loaders.loader import Loader


class GlimsLoader(Loader):
    """Loader class for loading labeled images from a directory."""

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

    def _retrieve_data(self) -> list[dict[str, Any]]:
        """Returns groups of images based on the filename format."""
        image_groups: dict[str, Any] = defaultdict(
            lambda: {
                "image": "",
                "diffusivity": "",
                "proliferation": "",
                "timepoint": "",
                "sample_id": "",
                "augmentation": {"identity": ""},
            }
        )

        for file_name in os.listdir(self.image_dir):
            if not file_name.endswith(".png") or not self._in_keys(file_name):
                continue

            diff, prolif, timepoint = self._parse_filename(file_name)
            sample_id = f"{prolif}_{diff}"
            simulation_id = f"{sample_id}_{timepoint}"

            group = image_groups[simulation_id]
            group["image"] = os.path.join(self.image_dir, file_name)
            group["diffusivity"] = diff
            group["proliferation"] = prolif
            group["timepoint"] = timepoint
            group["sample_id"] = f"{prolif}_{diff}"

            for transform_dict in self.augmentation_manager.transforms:
                ((aug_name, aug),) = transform_dict.items()
                if aug_name == "identity":
                    continue
                aug_simulation_id = f"{simulation_id}_{aug_name}"
                aug_group = dict(group)
                aug_group["augmentation"] = {aug_name: aug}
                image_groups[aug_simulation_id] = aug_group

        return list(image_groups.values())

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
