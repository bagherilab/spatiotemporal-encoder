from typing import Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

""" Pydantic models for the configuration files """


class DatasetConfig(BaseModel):
    loader: str = Field(..., description="Name of the dataset loader")
    image_dir: str = Field(..., description="Path to the directory with images")
    label_dir: str = Field(..., description="Path to the directory with labels")
    channels: list[str] = Field(..., description="List of channel names")
    batch_size: int = Field(
        ..., gt=0, description="Batch size for training, must be greater than 0"
    )
    val_split: float = Field(
        ..., ge=0.0, le=1.0, description="Validation split ratio, between 0 and 1"
    )
    test_split: float = Field(..., ge=0.0, le=1.0, description="Test split ratio, between 0 and 1")
    keys: list[str] = Field(..., description="List of keys")
    augmentations: Optional[dict] = None
    labels: Optional[list[str]] = None

    @model_validator(mode="before")
    def check_split_ratios(cls, values: dict) -> dict:
        val_split = values.get("val_split")
        test_split = values.get("test_split")
        if val_split is not None and test_split is not None and (val_split + test_split >= 1):
            raise ValueError("The sum of val_split and test_split must be less than 1")
        return values


class ModelParamsConfig(BaseModel):
    architecture: str = Field(..., description="Model architecture name")
    num_channels: int = Field(
        ..., description="Number of channels (should be the same across datasets)"
    )
    image_size: int = Field(..., gt=0, description="Size of the images, must be greater than 0")
    params: str = Field(..., description="Path or name of the hyperparameter configuration")

    @field_validator("image_size")
    def check_image_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("image_size must be greater than 0")
        return value

    @field_validator("num_channels")
    def check_num_channels(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("num_channels must have at least one element")
        return value


class GeneralConfig(BaseModel):
    pretrain: bool = Field(..., description="Whether to use pretraining")
    verbose: bool = Field(..., description="Whether to print verbose logs")


class ExperimentConfig(BaseModel):
    datasets: list[str]
    model: ModelParamsConfig
    general_configs: GeneralConfig


class MainConfig(BaseModel):
    experiment_name: str = Field(..., description="Name of the experiment")
    experiments: dict[str, ExperimentConfig] = Field(
        ..., description="Dictionary of experiment configurations"
    )


""" Hyperparameter configuration """


class HyperparameterRangeConfig(BaseModel):
    range: list[float] = Field(..., description="Range of continuous hyperparameters")
    search: str = Field(..., description="Search method for hyperparameters")
    num_samples: int = Field(..., gt=0, description="Number of samples to generate")

    @field_validator("range")
    def check_range(cls, value: list[float]) -> list[float]:
        if len(value) != 2 or value[0] >= value[1]:
            raise ValueError(
                "range must be a list with two elements where the first is less than the second"
            )
        return value


class HyperparameterDiscreteConfig(BaseModel):
    values: list[int] = Field(..., description="List of discrete hyperparameter values")


class HyperparameterConfig(BaseModel):
    num_epochs: int = Field(..., gt=0, description="Number of epochs for training")
    continuous: Optional[dict[str, HyperparameterRangeConfig]] = None
    discrete: Optional[dict[str, HyperparameterDiscreteConfig]] = None


""" Model architecture and type """


class LayerConfig(BaseModel):
    type: str = Field(..., description="Name of layer in PyTorch")
    in_features: Optional[Union[int, str]] = None
    out_features: Optional[Union[int, str]] = None
    num_features: Optional[Union[int, str]] = None
    in_channels: Optional[Union[int, str]] = None
    out_channels: Optional[Union[int, str]] = None
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    shape: Optional[list[int]] = None
    scale_factor: Optional[int] = None
    padding: Optional[Union[int, str]] = None
    p: Optional[float] = None
    output_padding: Optional[int] = None
    activation: Optional[str] = None

    @model_validator(mode="before")
    def check_layers(cls, values: dict) -> dict:
        layer_type = values.get("type")
        if layer_type is None:
            raise ValueError("Missing layer type")
        if layer_type == "Linear" and (
            values.get("in_features") is None or values.get("out_features") is None
        ):
            raise ValueError("Missing in_features for Linear layer")
        if layer_type in ["BatchNorm1d", "BatchNorm2d"] and values.get("num_features") is None:
            raise ValueError("Missing num_features for BatchNorm layer")
        if layer_type in ["Conv2d", "ConvTranspose2d"] and (
            values.get("in_channels") is None or values.get("out_channels") is None
        ):
            raise ValueError("Missing in_channels or out_channels for Conv2d layer")

        return values


class EncoderDecoderConfig(BaseModel):
    encoder: Optional[list[LayerConfig]] = None
    decoder_image: list[LayerConfig]
    decoder_timepoint: list[LayerConfig]

    @model_validator(mode="before")
    def check_encoders_decoders(cls, values: dict) -> dict:
        if "decoder_image" not in values:
            raise ValueError("Missing image decoder configuration")
        if "decoder_timepoint" not in values:
            raise ValueError("Missing timepoint decoder configuration")
        return values


class ModelArchitectureConfig(BaseModel):
    type: str = Field(..., description="Type of model architecture")
    architecture: EncoderDecoderConfig
