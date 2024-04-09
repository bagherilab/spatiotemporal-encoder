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

    batch_size = model_configs["batch_size"]
    test_split = model_configs["test_split"]
    verbose = model_configs["verbose"]

    runner = Runner(exp_name, verbose)

    for model_yaml in os.listdir("src/conf/models"):
        if model_yaml.endswith(".yaml"):
            runner.add_model(model_yaml)

    runner.add_dataset(data_dir, test_split, batch_size)
    runner.run_models()
    runner.save_results()


if __name__ == "__main__":
    main()
