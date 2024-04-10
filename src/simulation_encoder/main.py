import os
import sys
from pathlib import Path

import yaml

# For local imports in the module
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

from simulation_encoder.runner import Runner


def main() -> None:
    """
    Entry point for script
    """
    with open("src/conf/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    exp_name = config["experiment_name"]
    data_dir = config["data_dir"]
    model_configs = config["model_configs"]

    num_epochs = model_configs["num_epochs"]
    batch_size = model_configs["batch_size"]
    test_split = model_configs["test_split"]
    verbose = model_configs["verbose"]

    models = model_configs["models"]

    runner = Runner(exp_name, verbose)

    model_yaml_dir = "src/conf/models"
    for model in models:
        model_yaml = f"{model}.yaml"
        if not os.path.exists(f"{model_yaml_dir}/{model_yaml}"):
            raise FileNotFoundError(f"Model config file {model_yaml} not found in {model_yaml_dir}")
        runner.add_model(f"{model_yaml_dir}/{model_yaml}")

    runner.add_dataset(data_dir, test_split, batch_size)
    runner.run_models(num_epochs=num_epochs)
    runner.save_results()


if __name__ == "__main__":
    main()
