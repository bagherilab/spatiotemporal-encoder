import yaml

from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetParams:
    image_dir: str
    label_dir: str
    batch_size: int
    val_split: float
    test_split: float
    keys: list[str]
    augmentations: dict[str, dict[str, float]]
    
@dataclass
class ModelParams:
    architecture: str
    num_epochs: int
    params: dict[str, Any]