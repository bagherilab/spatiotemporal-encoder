from typing import Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

class DataConfig(BaseModel):
    image_dir: str = Field(..., description="Path to the directory with images")
    label_dir: str = Field(..., description="Path to the directory with labels")
    image_size: int = Field(..., gt=0, description="Size of the images, must be greater than 0")
    augmentations: Optional[dict[str, dict[str, Union[int, float, bool]]]] = None

    @field_validator('image_size')
    def check_image_size(cls, value):
        if value <= 0:
            raise ValueError("image_size must be greater than 0")
        return value

class ModelConfig(BaseModel):
    architecture: str = Field(..., description="Model architecture name")
    params: str = Field(..., description="Path or name of the hyperparameter configuration")

class GeneralConfig(BaseModel):
    batch_size: int = Field(..., gt=0, description="Batch size for training, must be greater than 0")
    val_split: float = Field(..., ge=0.0, le=1.0, description="Validation split ratio, between 0 and 1")
    test_split: float = Field(..., ge=0.0, le=1.0, description="Test split ratio, between 0 and 1")
    channels: list[str] = Field(..., description="List of channel names")
    pretrain: bool = Field(..., description="Whether to use pretraining")
    verbose: bool = Field(..., description="Whether to print verbose logs")

    @model_validator(mode="before")
    def check_split_ratios(cls, values):
        val_split = values.get('val_split')
        test_split = values.get('test_split')
        if val_split is not None and (val_split < 0 or val_split > 1):
            raise ValueError("val_split must be between 0 and 1")
        if test_split is not None and (test_split < 0 or test_split > 1):
            raise ValueError("test_split must be between 0 and 1")
        if val_split is not None and test_split is not None and (val_split + test_split >= 1):
            raise ValueError("The sum of val_split and test_split must be less than 1")
        return values
        
class ExperimentConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    general_configs: GeneralConfig
    keys: list[str] = Field(..., description="List of keys")
    emulator_targets: Optional[list[str]] = None

class MainConfig(BaseModel):
    experiment_name: str = Field(..., description="Name of the experiment")
    experiments: dict[str, ExperimentConfig] = Field(..., description="Dictionary of experiment configurations")

class HyperparameterRangeConfig(BaseModel):
    range: list[float] = Field(..., description="Range of continuous hyperparameters")
    search: str = Field(..., description="Search method for hyperparameters")
    num_samples: int = Field(..., gt=0, description="Number of samples to generate")

    @field_validator('range')
    def check_range(cls, value):
        if len(value) != 2 or value[0] >= value[1]:
            raise ValueError("range must be a list with two elements where the first is less than the second")
        return value

class HyperparameterDiscreteConfig(BaseModel):
    values: list[int] = Field(..., description="List of discrete hyperparameter values")

class HyperparameterConfig(BaseModel):
    num_epochs: int = Field(..., gt=0, description="Number of epochs for training")
    continuous: Optional[dict[str, HyperparameterRangeConfig]] = None
    discrete: Optional[dict[str, HyperparameterDiscreteConfig]] = None