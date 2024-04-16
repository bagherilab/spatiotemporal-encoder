import unittest
from unittest.mock import patch, MagicMock

from simulation_encoder.writer import Writer


class TestWriter(unittest.TestCase):
    @patch("simulation_encoder.runner.Runner")
    @patch("simulation_encoder.logger.ExperimentLogger")
    @patch("builtins.open")
    @patch("json.dump")
    def test_save_results_formats_correctly(self, mock_dump, mock_open, mock_logger, mock_runner):
        mock_logger.log = MagicMock()
        mock_runner.writer = Writer(uuid="1234")
        mock_runner.models = {"test_model": None}
        mock_runner.losses = {
            "test_model": {
                "train_loss": [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]],
                "val_loss": [[0.2, 0.1, 0.3], [0.9, 0.8, 1.0]],
                "test_loss": 0.25,
            },
        }

        expected_results = {
            "model": "test_model",
            "reconstruction_loss": {
                "train": [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]],
                "val": [[0.2, 0.1, 0.3], [0.9, 0.8, 1.0]],
                "test": 0.25,
            },
        }
        mock_runner.writer.write_results("test_model", mock_runner.losses)
        actual_call_args = mock_dump.call_args[0][0]
        self.assertEqual(actual_call_args, expected_results)


if __name__ == "__main__":
    unittest.main()
