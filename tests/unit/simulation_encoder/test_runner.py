import unittest
from unittest.mock import patch, mock_open, MagicMock

import json

from simulation_encoder.runner import Runner


class TestRunner(unittest.TestCase):
    def test_save_results_formats_correctly(self):
        runner = Runner("test_experiment_name")
        runner.models = {"model1": None, "model2": None}
        runner.losses = {
            "model1": [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]],
            "model2": [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]],
        }
        runner.val_losses = {
            "model1": [[0.2, 0.1, 0.3], [0.9, 0.8, 1.0]],
            "model2": [[0.3, 0.4, 0.5], [0.8, 0.9, 1.0]],
        }
        runner.test_losses = {
            "model1": 0.25,
            "model2": 0.1,
        }

        expected_results = {
            "models": {
                "model1": {
                    "train_loss": [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]],
                    "val_loss": [[0.2, 0.1, 0.3], [0.9, 0.8, 1.0]],
                    "test_loss": 0.25,
                },
                "model2": {
                    "train_loss": [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]],
                    "val_loss": [[0.3, 0.4, 0.5], [0.8, 0.9, 1.0]],
                    "test_loss": 0.1,
                },
            }
        }
        with patch("builtins.open", mock_open()) as mock_file, patch("json.dump") as mock_dump:
            runner.save_results()
            # Get the actual value passed to json.dump() in the call
            actual_call_args = mock_dump.call_args[0][0]
            # Compare only the values of the models and losses dictionaries
            self.assertEqual(actual_call_args["models"], expected_results["models"])


if __name__ == "__main__":
    unittest.main()
