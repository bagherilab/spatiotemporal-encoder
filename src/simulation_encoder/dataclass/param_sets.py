from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DatasetParams:
    loader: str
    image_dir: str
    channels: list[str]
    batch_size: int
    val_split: float
    test_split: float
    keys: list[str]
    name: Optional[str] = None
    labels: Optional[list[str]] = None
    label_dir: Optional[str] = None
    augmentations: Optional[dict] = None


@dataclass
class ModelParams:
    name: str
    model_type: str
    architecture: dict[str, Any]
    num_channels: int
    num_epochs: int
    params: dict[str, Any] = field(default_factory=lambda: {"latent_dim": 2})
