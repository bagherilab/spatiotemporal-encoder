import unittest
from unittest.mock import patch

import os
import tempfile

from simulation_encoder.runner import Runner
from simulation_encoder.dataclass.param_sets import ModelParams


class TestRunner(unittest.TestCase):
    # This function is run at the beginning and all the tests can use the information here
    def setUp(self) -> None:
        # Writer and Logger are other classes that write to files
        # Here we are mocking them to avoid writing to files
        self.mock_writer = patch("simulation_encoder.runner.Writer", spec=True)
        self.mock_logger = patch("simulation_encoder.runner.ExperimentLogger", spec=True)

        self.mock_writer.start()
        self.mock_logger.start()

        # Create the object we actually want to test
        self.runner = Runner(verbose=False)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.mock_writer.stop()
        self.mock_logger.stop()
        self.temp_dir.cleanup()

    def test_add_models_creates_correct_id(self):
        # Creates a fake set of model parameters (an object to store model information)
        model_params = ModelParams(
            name="test_model",
            architecture={
                "encoder": [{"type": "Linear", "in_features": 1, "out_features": 1}],
                "decoder_image": [{"type": "Linear", "in_features": 1, "out_features": 1}],
                "decoder_timepoint": [{"type": "Linear", "in_features": 1, "out_features": 1}],
            },
            model_type="CAE",
            num_channels=1,
            num_epochs=2,
            params={"latent_dim": 2},
        )

        # Adds the model to the runner
        self.runner.add_models([model_params])
        model_id = self.runner.models.keys()

        # Checks that the model was added
        self.assertIn("test_model_2d_0", model_id)


    """
    YOUR TURN 
    These tests are a little trick (hence why i havent done them yet)
    But you could conssider mocking a dataset and a model in the run_encoder test and
    maybe using the assert_called_with method to check that the model is being called correct?"""
    def test_add_dataset(self):
        assert False

    def run_encoder(self):
        assert False



if __name__ == "__main__":
    unittest.main()
