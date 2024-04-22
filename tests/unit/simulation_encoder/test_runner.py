import unittest
from unittest.mock import patch, mock_open, MagicMock, PropertyMock

import os
import tempfile
import yaml

from simulation_encoder.runner import Runner


class TestRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_writer = patch("simulation_encoder.runner.Writer", spec=True)
        self.mock_logger = patch("simulation_encoder.runner.ExperimentLogger", spec=True)

        self.mock_writer.start()
        self.mock_logger.start()

        self.runner = Runner(verbose=False)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.mock_writer.stop()
        self.mock_logger.stop()
        self.temp_dir.cleanup()

    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_add_models(self, mock_yaml_load, mock_open):
        yaml_contents = {
            "architecture": {
                "encoder": [
                    {"type": "Conv2d", "in_channels": 3, "out_channels": 16, "kernel_size": 3}
                ],
                "decoder_image": [
                    {
                        "type": "ConvTranspose2d",
                        "in_channels": 16,
                        "out_channels": 3,
                        "kernel_size": 3,
                    }
                ],
                "decoder_timepoint": [
                    {"type": "Conv2d", "in_channels": 16, "out_channels": 1, "kernel_size": 3}
                ],
            }
        }

        mock_yaml_load.return_value = yaml_contents
        yaml_path = os.path.join(self.temp_dir.name, "test_model.yaml")
        self.runner.add_models([yaml_path])
        self.assertIn("test_model", self.runner.models)


if __name__ == "__main__":
    unittest.main()
