import unittest
from unittest.mock import patch, MagicMock

from simulation_encoder.writer import Writer
from simulation_encoder.dataclass.loss_data import LossData


class TestWriter(unittest.TestCase):
    @patch("simulation_encoder.runner.Runner")
    @patch("simulation_encoder.logger.ExperimentLogger")
    @patch("builtins.open")
    @patch("json.dump")
    def test_save_results_formats_correctly(self, mock_dump, mock_open, mock_logger, mock_runner):
        mock_logger.log = MagicMock()
        Writer._create_dir = MagicMock()
        mock_runner.writer = Writer(uuid="1234")
        mock_runner.models = {"test_model": None}

        train_loss={"image": [0.1, 0.2, 0.3], "timepoint": [0.4, 0.5, 0.6], "combined": [0.7, 0.8, 0.9]}
        val_loss={"image": [0.2, 0.1, 0.3], "timepoint": [0.5, 0.4, 0.6], "combined": [0.8, 0.7, 0.9]}
        test_loss={"image": 0.25, "timepoint": 0.55, "combined": 0.85}

        mock_runner.losses = LossData()
        mock_runner.losses.add_train_loss(train_loss)
        mock_runner.losses.add_val_loss(val_loss)
        mock_runner.losses.add_test_loss(test_loss)

        expected_results = {
            "model": "test_model",
            "combined_loss": {
                "train": [0.7, 0.8, 0.9],
                "val": [0.8, 0.7, 0.9],
                "test": 0.85,
            },
            "reconstruction_loss": {
                "train": [0.1, 0.2, 0.3],
                "val": [0.2, 0.1, 0.3],
                "test": 0.25,
            },
            "timepoint_loss": {
                "train": [0.4, 0.5, 0.6],
                "val": [0.5, 0.4, 0.6],
                "test": 0.55,
            }
        }

        mock_runner.writer.write_results("test_model", mock_runner.losses)
        actual_call_args = mock_dump.call_args[0][0]
        self.assertEqual(actual_call_args, expected_results)


if __name__ == "__main__":
    unittest.main()
