import tempfile
import unittest

from simulation_encoder.dataclass.param_sets import ModelParams
from simulation_encoder.runner import Runner


class TestRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = Runner(verbose=False)
        self.temp_dir = tempfile.TemporaryDirectory()

    def test_add_models_creates_correct_id(self):
        model_params = ModelParams(
            name="test_model",
            architecture={
                "encoder": [{"type": "Linear", "in_features": 1, "out_features": 1}],
                "decoder_image": [
                    {"type": "Linear", "in_features": 1, "out_features": 1}
                ],
                "decoder_timepoint": [
                    {"type": "Linear", "in_features": 1, "out_features": 1}
                ],
            },
            model_type="CAE",
            num_channels=1,
            num_epochs=2,
            params={"latent_dim": 2},
        )

        self.runner.add_models([model_params])
        model_id = self.runner.models.keys()

        self.assertIn("test_model_0", model_id)


if __name__ == "__main__":
    unittest.main()
