from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetParams:
    loader: str
    image_dir: str
    batch_size: int
    val_split: float
    test_split: float
    keys: list[str]
    augmentations: dict[str, dict[str, float]]
    label_dir: str = field(default_factory=str)
    labels: list[str] = field(default_factory=list)


@dataclass
class ModelParams:
    name: str
    architecture: dict[str, Any]
    num_epochs: int
    params: dict[str, Any] = field(default_factory=lambda: {"latent_dim": 2})
