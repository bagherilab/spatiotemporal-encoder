from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetParams:
    loader: str
    image_dir: str
    channels: list[str]
    batch_size: int
    val_split: float
    test_split: float
    keys: list[str]
    name: str | None = None
    labels: list[str] | None = None
    label_dir: str | None = None
    augmentations: dict | None = None


@dataclass
class ModelParams:
    name: str
    model_type: str
    architecture: dict[str, Any]
    num_channels: int
    num_epochs: int
    params: dict[str, Any] = field(default_factory=lambda: {"latent_dim": 2})
