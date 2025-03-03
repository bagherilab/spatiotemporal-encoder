import os
import json
from simulation_encoder.loaders.loader import Loader
from simulation_encoder.loaders.arcade_loader import ARCADELoader
from simulation_encoder.loaders.gastruloid_loader import GastruloidLoader
from simulation_encoder.loaders.alphanumeric_loader import AlphanumericLoader
from simulation_encoder.loaders.glims_loader import GlimsLoader


def load_loaders(
    results_path: str, image_base_dir: str, data_labels: list[str] | None = None
) -> dict[str, Loader]:
    """
    Load data loaders for each dataset based on results path.

    Parameters:
    -----------
    results_path : str
        Path to the results directory containing experiment folders.
    image_base_dir : str
        Base directory where the image datasets are stored.
    data_labels : list[str] | None, default=None
        List of data labels for the loader.

    Returns:
    --------
    dict[str, Loader]
        Dictionary mapping dataset_name -> loader
    """
    loaders = {}
    processed_datasets = set()

    for experiment in os.listdir(results_path):
        experiment_path = f"{results_path}/{experiment}"
        if not os.path.isdir(experiment_path):
            continue

        for model_name in os.listdir(experiment_path):
            model_path = f"{experiment_path}/{model_name}"
            if not os.path.isdir(model_path):
                continue

            for dataset_name in os.listdir(model_path):
                dataset_dir = f"{model_path}/{dataset_name}"

                if dataset_name in processed_datasets or not os.path.isdir(dataset_dir):
                    continue

                results_file = f"{dataset_dir}/results.json"
                if not os.path.exists(results_file):
                    continue

                try:
                    loader = _create_loader(
                        experiment_path=results_path,
                        experiment=experiment,
                        dataset_name=dataset_name,
                        dataset_dir=dataset_dir,
                        image_base_dir=image_base_dir,
                        data_labels=data_labels,
                    )

                    if loader:
                        loaders[dataset_name] = loader
                        processed_datasets.add(dataset_name)

                except Exception as e:
                    print(f"Error loading loader for {dataset_name}: {str(e)}")

    return loaders


def _create_loader(
    experiment_path: str,
    experiment: str,
    dataset_name: str,
    dataset_dir: str,
    image_base_dir: str,
    data_labels: list[str] | None = None,
) -> Loader:
    """
    Helper function to create a loader instance for a specific dataset.

    Parameters:
    -----------
    experiment_path : str
        Path to the experiment directory.
    experiment : str
        Name of the experiment.
    dataset_name : str
        Name of the dataset.
    dataset_dir : str
        Path to the dataset directory (contains results.json).
    image_base_dir : str
        Base directory where the image datasets are stored.
    data_labels : list[str] | None, default=None
        List of data labels for the loader.

    Returns:
    --------
    Loader
        The appropriate loader instance.
    """
    with open(f"{dataset_dir}/results.json") as file:
        results = json.load(file)

    channels = results["channels"]
    keys = results["data_keys"]
    augmentations = results["data_augmentations"]

    loader_type = results["loader"]

    image_path = f"{image_base_dir}/{dataset_name}/images"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image directory not found: {image_path}")

    label_path = (
        f"{image_base_dir}/{dataset_name}/labels"
        if os.path.exists(f"{image_base_dir}/{dataset_name}/labels")
        else None
    )
    indices_path = f"{experiment_path}/{experiment}/{dataset_name}_indices.json"

    if loader_type == "ARCADELoader":
        return ARCADELoader(
            image_dir=image_path,
            channels=channels,
            label_dir=label_path,
            keys=keys,
            labels=data_labels,
            batch_size=1,
            augmentations=augmentations,
            indices_file=indices_path,
        )
    elif loader_type == "GastruloidLoader":
        return GastruloidLoader(
            image_dir=image_path,
            channels=channels,
            keys=keys,
            batch_size=1,
            augmentations=augmentations,
            indices_file=indices_path,
        )
    elif loader_type == "AlphanumericLoader":
        return AlphanumericLoader(
            image_dir=image_path,
            channels=channels,
            keys=keys,
            batch_size=1,
            augmentations=augmentations,
            indices_file=indices_path,
        )
    elif loader_type == "GlimsLoader":
        return GlimsLoader(
            image_dir=image_path,
            channels=channels,
            keys=keys,
            batch_size=1,
            augmentations=augmentations,
            indices_file=indices_path,
        )
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
