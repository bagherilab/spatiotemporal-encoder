import unittest
from unittest.mock import patch

import os
import tempfile

from simulation_encoder.runner import Runner
from simulation_encoder.dataclass.param_sets import ModelParams


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
    def test_add_models(self, mock_open):
        model_params = ModelParams(
            name="test_model",
            architecture={
                "encoder": [{"type": "Linear", "in_features": 1, "out_features": 1}],
                "decoder_image": [{"type": "Linear", "in_features": 1, "out_features": 1}],
                "decoder_timepoint": [{"type": "Linear", "in_features": 1, "out_features": 1}],
            },
            model_type="CAE",
            num_epochs=1,
            params={"latent_dim": 2},
        )

        self.runner.add_models([model_params])
        model_id = self.runner.models.keys()
        self.assertIn("test_model_2d_0", model_id)


if __name__ == "__main__":
    unittest.main()
