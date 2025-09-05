import cProfile
import pstats
import time
from copy import deepcopy

from simulation_encoder.runner import Runner
from simulation_encoder.writer import Writer
from simulation_encoder.plotter import Plotter
from simulation_encoder.logger import Logger

from simulation_encoder.dataclass.param_sets import DatasetParams, ModelParams
from simulation_encoder.dataclass.config_schemas import (
    MainConfig,
    ExperimentConfig,
    ModelParamsConfig,
)
from simulation_encoder.utils.generate_hyperparams import generate_hyperparameters
from simulation_encoder.utils.yaml_utils import (
    load_model_yaml,
    load_dataset_yaml,
    load_study_yaml,
    load_hyperparam_yaml,
    load_yaml,
)

CONFIG_YAML = "src/conf/config.yaml"
STUDIES_DIR = "src/conf/studies"
RESULTS_DIR = "results"


def main() -> None:
    """Entry point for the simulation encoder. This function reads the config.yaml file and
    creates the necessary objects to run the simulation encoder. The main components are the
    dataset parameters, model parameters, and the runner object. The runner object is responsible
    for running the simulation encoder with the given dataset and model parameters. The dataset
    parameters are created from the config.yaml file, and the model parameters are created from the
    hyperparameter yaml files. The runner object is created with the dataset and model parameters,
    and the run method is called to start the simulation encoder."""

    main_config = load_yaml(CONFIG_YAML, MainConfig)

    study_name = main_config.study_name  # type: ignore
    data_quantity_experiment = main_config.data_quantity_experiment  # type: ignore
    study_config = load_study_yaml(study_name)

    fractions = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05] if data_quantity_experiment else [1.0]

    for experiment_key, experiment_config in study_config.experiments.items():  # type: ignore
        verbose = experiment_config.general_configs.verbose

        # Loader params (same for all fractions in this experiment)
        loader_params = create_loader_params(experiment_config)

        # Create original loaders *once* for this experiment
        temp_runner = Runner(Logger(log_name="temp", verbose=False), verbose=False)
        temp_runner.add_loader_params(loader_params)
        original_loaders = {}
        for dataset_name in loader_params.keys():
            loader = temp_runner.create_loader(dataset_name)
            original_loaders[dataset_name] = loader  # store full dataset version

        for frac in fractions:
            frac_key = f"{experiment_key}_frac{int(frac*100)}"
            logger = Logger(log_name=f"{study_name}_{frac_key}", verbose=verbose)
            runner = Runner(logger, verbose)
            writer = Writer(results_dir=f"{RESULTS_DIR}/{study_name}", experiment_key=frac_key)
            plotter = Plotter(results_dir=f"{RESULTS_DIR}/{study_name}", experiment_key=frac_key)

            runner.add_loader_params(loader_params)
            first_dataset_name = next(iter(loader_params))
            first_dataset_params = loader_params[first_dataset_name]
            num_input_channels = len(first_dataset_params.channels)

            for dataset_name, base_loader in original_loaders.items():
                loader_copy = (
                    base_loader.clone() if hasattr(base_loader, "clone") else deepcopy(base_loader)
                )
                if frac < 1.0:
                    loader_copy.subsample_train_indices(frac=frac)
                writer.write_train_test_indices(dataset_name, loader_copy.get_indices())
                runner.replace_loader(dataset_name, loader_copy)

            # Model
            model_param_sets = create_model_param_sets(experiment_config.model, num_input_channels)
            runner.add_model_params(model_param_sets)

            # Run
            encoder_results = runner.run_encoder(study_name)
            handle_encoder_results(encoder_results, runner, writer, plotter, save_all_models=False)


def create_loader_params(
    experiment_config: ExperimentConfig,
) -> dict[str, DatasetParams]:
    """
    Create loader parameter sets from the experiment config.

    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration containing dataset information

    Returns
    -------
    dict[str, DatasetParams]
        Dictionary mapping dataset names to their parameter sets
    """
    loader_params = {}

    dataset_names = experiment_config.datasets

    for dataset_name in dataset_names:
        dataset_config = load_dataset_yaml(dataset_name)
        dataset_params = DatasetParams(
            loader=dataset_config.loader,
            image_dir=dataset_config.image_dir,
            label_dir=dataset_config.label_dir,
            image_size=dataset_config.image_size,
            channels=dataset_config.channels,
            batch_size=dataset_config.batch_size,
            val_split=dataset_config.val_split,
            test_split=dataset_config.test_split,
            keys=dataset_config.keys,
            augmentations=dataset_config.augmentations,
            labels=dataset_config.labels,
            name=dataset_name,
        )
        loader_params[dataset_name] = dataset_params

    return loader_params


def create_model_param_sets(
    model_config: ModelParamsConfig, num_channels: int
) -> list[ModelParams]:
    """Create the model parameters from model config files and hyperparameter yaml files."""
    model_name = model_config.architecture
    model_yaml = load_model_yaml(model_name)

    model_params = load_hyperparam_yaml(model_config.params)
    num_epochs = model_params.num_epochs
    continuous_params = model_params.continuous
    discrete_params = model_params.discrete
    param_sets = generate_hyperparameters(continuous_params, discrete_params)

    model_param_sets = []
    for param_set in param_sets:
        model_params = ModelParams(
            name=model_name,
            model_type=model_yaml.type,  # type: ignore
            architecture=model_yaml.architecture.model_dump(exclude_none=True),  # type: ignore
            num_channels=num_channels,
            num_timepoints=model_config.num_timepoints,
            num_epochs=num_epochs,
            params=param_set,
        )
        model_param_sets.append(model_params)

    return model_param_sets


def handle_encoder_results(
    encoder_results: dict,
    runner: Runner,
    writer: Writer,
    plotter: Plotter,
    save_all_models: bool = False,
) -> None:
    """Handles writing and plotting of encoder results"""
    best_model_info = None
    best_val_loss = float("inf")

    for dataset_name, dataset_results in encoder_results.items():
        for model_id, data in dataset_results.items():
            model = data["model"]
            loader = runner.create_loader(dataset_name)

            if save_all_models:
                writer.write_model_state(model_id, dataset_name, model)

            writer.write_encoder_results(
                model_id,
                loader,
                model,
                data["losses"],
            )

            plotter.line_plot(
                model_id,
                dataset_name,
                data["grad_norms"],
                "grad_norms",
                "Epoch",
                "Gradient Norm",
            )

            # Access LossData object for losses
            losses = data["losses"]
            plotter.loss_plot(
                model_id,
                dataset_name,
                losses.losses_train.get("weighted_loss", []),
                losses.losses_val.get("weighted_loss", []),
            )

            if losses.losses_val.get("weighted_loss"):
                final_val_loss = losses.losses_val["weighted_loss"][-1]
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_model_info = {
                        "model": model,
                        "dataset_name": dataset_name,
                        "losses": losses,
                    }

    if best_model_info is not None:
        best_model = best_model_info["model"]
        best_model_dataset_name = best_model_info["dataset_name"]
        best_model_loader = runner.create_loader(best_model_dataset_name)

        writer.write_encoder_results(
            "_best_model", best_model_loader, best_model, best_model_info["losses"]
        )
        writer.write_encoded_data(
            "_best_model", best_model_loader, best_model_dataset_name, best_model
        )
        writer.write_model_state("_best_model", best_model_dataset_name, best_model)


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        start_time = time.time()
        main()
        end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats("profile_output.prof")
