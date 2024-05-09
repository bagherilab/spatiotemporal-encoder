import os
import sys
from pathlib import Path

import yaml

# For local imports in the module
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

from simulation_encoder.runner import Runner

MODEL_YAML_DIR = "src/conf/models"


def main() -> None:
    """Entry point for script"""
    with open("src/conf/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    image_dir = config["image_dir"]
    label_dir = config["label_dir"]
    models = config["models"]
    model_configs = config["model_configs"]
    keys = config["keys"]

    num_epochs = model_configs["num_epochs"]
    batch_size = model_configs["batch_size"]
    val_split = model_configs["val_split"]
    test_split = model_configs["test_split"]
    augmentations = model_configs["augmentations"]
    verbose = model_configs["verbose"]

    model_files = get_model_files(models)

    runner = Runner(augmentations, verbose)
    runner.add_models(model_files)
    runner.add_dataset(image_dir, label_dir, keys, val_split, test_split, batch_size)
    runner.run(num_epochs)


def get_model_files(models: list[str]) -> list[str]:
    """Reads model list from config file"""
    model_files = []
    for model in models:
        model_yaml = f"{model}.yaml"
        if not os.path.exists(f"{MODEL_YAML_DIR}/{model_yaml}"):
            raise FileNotFoundError(f"Model config file {model_yaml} not found in {MODEL_YAML_DIR}")
        model_files.append(f"{MODEL_YAML_DIR}/{model_yaml}")
    return model_files


if __name__ == "__main__":
    main()
