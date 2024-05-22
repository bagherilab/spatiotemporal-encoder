import os #used for interacting with the operating system
import sys #provides access to variables used by python interperter
from pathlib import Path #used to work with file directory paths

import yaml #used to read YAML files

#main job of main --> read variables from the config file and use them

# For local imports in the module
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))
    #If the parent directory of the current file is not already in the list of directories where Python looks for modules (sys.path), then add it to that list.

from simulation_encoder.runner import Runner
#imports the runner class from the simcode.runner module
#i order for this file to find runner, you have to be in a certain directory when you run the code
#when you run the tests (from the tests folder), it's gonna look in a different place

MODEL_YAML_DIR = "src/conf/models"
#holds the directory path to where the yaml files are located

#MAIN sets up parameters for running a simulation, creates an instance of the Runner class, 
#adds models, datasets, and runs the simulation
def main() -> None:
    """Entry point for script""" #prvides documentation for the function w 3 """ (it's like  a comment)
    with open("src/conf/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) #reads a YAML configuration file which contains settings for the simulation.

    image_dir = config["image_dir"]
    label_dir = config["label_dir"]
    models = config["models"]
    model_configs = config["model_configs"]
    keys = config["keys"]
    # extract various parameters from the configuration file

    num_epochs = model_configs["num_epochs"]
    batch_size = model_configs["batch_size"]
    val_split = model_configs["val_split"]
    test_split = model_configs["test_split"]
    augmentations = model_configs["augmentations"]
    verbose = model_configs["verbose"]
    #extract parameters related to the models from the configuration

    model_files = get_model_files(models)
    #get the paths of model YAML files based on the model names extracted from the configuration
    #reads model specific yaml file

    runner = Runner(augmentations, verbose)
    runner.add_models(model_files)
    runner.add_dataset(image_dir, label_dir, keys, val_split, test_split, batch_size)
    runner.run(num_epochs)
#makes a runner class with specific verboseness and aug


def get_model_files(models: list[str]) -> list[str]:
    """Reads model list from config file"""
    model_files = []
    for model in models:
        model_yaml = f"{model}.yaml"
        if not os.path.exists(f"{MODEL_YAML_DIR}/{model_yaml}"):
            raise FileNotFoundError(f"Model config file {model_yaml} not found in {MODEL_YAML_DIR}")
        model_files.append(f"{MODEL_YAML_DIR}/{model_yaml}")
    return model_files
#takes a list of model names and returns a list of paths to their YAML configuration files

if __name__ == "__main__":
    main()
