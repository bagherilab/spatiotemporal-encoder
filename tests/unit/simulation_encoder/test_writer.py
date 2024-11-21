import unittest
from unittest.mock import patch, MagicMock

from simulation_encoder.writer import Writer
from simulation_encoder.dataclass.loss_data import LossData


class TestWriter(unittest.TestCase):
    @patch("simulation_encoder.logger.Logger")
    def setUp(self, mock_logger):
        self.mock_runner = MagicMock()

        self.mock_runner.models = MagicMock()
        self.mock_runner.models.name = "test_model"
        self.mock_runner.models.params = {"test_key": "test_value"}
        self.mock_runner.models.architecture = {"architecture": "test_model"}
        self.mock_runner.models.num_channels = 1

        self.mock_runner.dataset = MagicMock()
        self.mock_runner.dataset.keys = ["test_key"]
        self.mock_runner.dataset.augmentations = ["rotate_180", "rotate_90"]
        self.mock_runner.dataset.channels = ["test_image"]

        mock_logger.log = MagicMock()
        Writer._create_dir = MagicMock()
        self.mock_runner.logger = mock_logger

        self.mock_runner.writer = Writer()

    @patch("builtins.open")
    @patch("json.dump")
    def test_save_results_formats_correctly(self, mock_dump, mock_open):
        train_loss = {
            "image": [0.1, 0.2, 0.3],
            "timepoint": [0.4, 0.5, 0.6],
            "weighted_loss": [0.7, 0.8, 0.9],
        }
        val_loss = {
            "image": [0.2, 0.1, 0.3],
            "timepoint": [0.5, 0.4, 0.6],
            "weighted_loss": [0.8, 0.7, 0.9],
        }
        test_loss = {"image": 0.25, "timepoint": 0.55, "weighted_loss": 0.85}

        self.mock_runner.losses = LossData()
        self.mock_runner.losses.add_train_loss(train_loss)
        self.mock_runner.losses.add_val_loss(val_loss)
        self.mock_runner.losses.add_test_loss(test_loss)

        expected_results = {
            "model": "test_model",
            "dataset": self.mock_runner.dataset.name,
            "architecture": self.mock_runner.model.name,
            "channels": ["test_image"],
            "params": self.mock_runner.model.params,
            "data_augmentations": ["rotate_180", "rotate_90"],
            "keys": ["test_key"],
            "losses": {
                "train": train_loss,
                "val": val_loss,
                "test": test_loss,
            },
        }

        self.mock_runner.writer.write_encoder_results(
            model_name="test_model",
            dataset=self.mock_runner.dataset,
            model=self.mock_runner.model,
            losses=self.mock_runner.losses,
        )

        actual_call_args = mock_dump.call_args[0][0]
        print(actual_call_args)
        print(expected_results)
        self.assertEqual(actual_call_args, expected_results)


if __name__ == "__main__":
    unittest.main()
