from collections import defaultdict

from simulation_encoder.logger import Logger
from latent_model.loaders.sequence_loader import SequenceLoader
from latent_model.models.temporal_models import TemporalModel


class Runner:
    """
    Class for managing the training and evaluation of RNN models

    Parameters
    ----------
    verbose : bool
        Controls if model training is output to console

    Attributes
    ----------
    UUID : uuid.UUID
        Unique identifier for the run
    models : dict[str, RNN]
        Dictionary of model names and their corresponding RNN models
    loaders : dict[str, SequenceLoader]
        Dataset loaders to be used for training and evaluation
    losses : dict[str, LossData]
        Dictionary of model names and their corresponding loss data
    """

    def __init__(self, logger: Logger = None, verbose: bool = False) -> None:
        self.logger = logger
        self.verbose = verbose
        self.models = {}
        self.loaders = {}

    def add_loaders(self, loaders: dict[str, SequenceLoader]) -> None:
        """Set the loaders on which models should be trained"""
        self.loaders = loaders

    def add_models(self, models: dict[str, dict[str, list[TemporalModel]]]) -> None:
        """Add models to be trained by the runner"""
        self.models = models

    def get_loader(self, model_name: str, dataset_name: str) -> SequenceLoader:
        """Returns loader"""
        return self.loaders[model_name][dataset_name]

    def get_models(self, model_name: str, dataset_name: str) -> list[TemporalModel]:
        """Returns potential models"""
        return self.models[model_name][dataset_name]

    def run_temporal_model(self) -> dict:
        """Runs the training and evaluation of RNN models"""
        if not self.loaders:
            raise ValueError("No loaders have been added to the runner.")
        if not self.models:
            raise ValueError("No models have been added to the runner.")

        results = defaultdict(dict)

        for encoder_model_name, dataset_loaders in self.loaders.items():
            for dataset_name, loader in dataset_loaders.items():
                best_model = None
                best_loss = float("inf")

                print(
                    f"Finding optimal model for {dataset_name} data encoded with {encoder_model_name}"
                )
                for model in self.models[encoder_model_name][dataset_name]:
                    train_losses, val_losses = self._train_model(model, loader)

                    if val_losses[-1] < best_loss:
                        best_loss = val_losses[-1]
                        best_model = model

                results[encoder_model_name][dataset_name] = {
                    "best_model": best_model,
                    "best_val_loss": best_loss,
                }

                # test_loss = self._eval_model(best_model, loader)

        return results

    def _train_model(self, model: TemporalModel, loader: SequenceLoader) -> tuple:
        """Trains an RNN model on the dataset"""
        train_loader = loader.get_dataloader(dataset_type="train")
        val_loader = loader.get_dataloader(dataset_type="val")

        losses, val_losses = model.fit(
            train_loader, val_loader=val_loader, patience=5, min_delta=0.001, max_epochs=1
        )

        return losses, val_losses

    def _eval_model(self, model: TemporalModel, loader: SequenceLoader) -> float:
        model.eval()
        test_loader = loader.get_dataloader(dataset_type="test")
        test_loss = model.eval_one_epoch(test_loader)
        return test_loss

    def _log(self, msg: str, level: str = "info") -> None:
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            else:
                self.logger.log(msg)
