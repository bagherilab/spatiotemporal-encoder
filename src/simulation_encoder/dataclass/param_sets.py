from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetParams:
    image_dir: str
    label_dir: str
    batch_size: int
    val_split: float
    test_split: float
    keys: list[str]
    labels: list[str]
    augmentations: dict[str, dict[str, float]]


@dataclass
class ModelParams:
    name: str
    architecture: dict[str, Any]
    num_epochs: int
    params: dict[str, Any] = field(default_factory=lambda: {"latent_dim": 2})
