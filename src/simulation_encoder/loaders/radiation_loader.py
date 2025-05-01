import os
from typing import Optional, Any
from collections import defaultdict

from simulation_encoder.logger import Logger
from simulation_encoder.loaders.loader import Loader


class RadiationLoader(Loader):
    """
    Loader class for loading labeled images from a directory.

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
        batch_size: int = 16,
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
                **{channel: "" for channel in self.channels},
                "timepoint": "",
                "sample_id": "",
                "dose": "",
                "radiation_type": "",
                "augmentation": {"identity": ""},
            }
        )

        for file_name in os.listdir(self.image_dir):
            if not file_name.endswith(".png") or not self._in_keys(file_name):
                continue

            # if not any(channel in file_name for channel in self.channels):
            #     continue

            plate_well, id, dose, radiation_type, timepoint = self._parse_filename(file_name)
            sample_id = f"{plate_well}_{id}_{radiation_type}"
            simulation_id = f"{sample_id}_{timepoint}"
            group = image_groups[simulation_id]
            group["timepoint"] = timepoint
            group["sample_id"] = sample_id
            group["dose"] = dose
            group["radiation_type"] = radiation_type
            group["image"] = os.path.join(self.image_dir, file_name)

            for transform_dict in self.augmentation_manager.transforms:
                ((aug_name, aug),) = transform_dict.items()
                if aug_name == "identity":
                    continue
                aug_simulation_id = f"{simulation_id}_{aug_name}"
                aug_group = dict(group)
                aug_group["augmentation"] = {aug_name: aug}
                image_groups[aug_simulation_id] = aug_group

        self._log_missing_images(image_groups)
        return list(image_groups.values())

    def _parse_filename(self, filename: str) -> tuple[str, str, float, str, int]:
        file_chunks = filename.split("_")
        plate_well = file_chunks[:2]
        id = file_chunks[2:4]
        dose = file_chunks[4][:-2]
        radiation_type = file_chunks[5]
        timepoint = file_chunks[6].split(".")[0]

        return (
            plate_well,
            id,
            float(dose),
            str(radiation_type),
            int(timepoint),
        )

    def _in_keys(self, file_name: str) -> bool:
        file_chunks = file_name.split("_")
        prefix = file_chunks[0]
        
        return prefix in self.keys
