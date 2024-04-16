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

    data_dir = config["data_dir"]
    models = config["models"]
    model_configs = config["model_configs"]

    num_epochs = model_configs["num_epochs"]
    batch_size = model_configs["batch_size"]
    test_split = model_configs["test_split"]
    verbose = model_configs["verbose"]

    model_files = get_model_files(models)

    runner = Runner(verbose)
    runner.add_models(model_files)
    runner.add_dataset(data_dir, test_split, batch_size)
    runner.train_models(num_epochs)
    runner.eval_models()


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
