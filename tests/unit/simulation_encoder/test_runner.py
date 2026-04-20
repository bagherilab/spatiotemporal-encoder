import unittest
from unittest.mock import MagicMock

from simulation_encoder.runner import Runner
from simulation_encoder.dataclass.param_sets import ModelParams, DatasetParams


def _make_model_params(name="test_model", model_type="AE"):
    """Minimal ModelParams for tests that don't need a real model instance."""
    return ModelParams(
        name=name,
        model_type=model_type,
        architecture={
            "encoder": [{"type": "Linear", "in_features": 1, "out_features": 1}],
            "decoder_image": [{"type": "Linear", "in_features": 1, "out_features": 1}],
            "decoder_timepoint": [{"type": "Linear", "in_features": 1, "out_features": 1}],
        },
        num_channels=1,
        num_timepoints=2,
        num_epochs=1,
        params={"latent_dim": 2},
    )


def _make_dataset_params(loader="arcade", name="my_ds"):
    return DatasetParams(
        loader=loader,
        image_dir="/tmp/images",
        image_size=64,
        channels=["cancer"],
        batch_size=4,
        val_split=0.1,
        test_split=0.1,
        keys=["CH_typeA"],
        name=name,
    )


class TestRunnerAddModelParams(unittest.TestCase):
    def setUp(self):
        self.runner = Runner(verbose=False)

    def test_creates_correct_model_id(self):
        self.runner.add_model_params([_make_model_params()])
        self.assertIn("test_model_0", self.runner.models)

    def test_multiple_models_get_sequential_ids(self):
        self.runner.add_model_params([_make_model_params("a"), _make_model_params("b")])
        self.assertIn("a_0", self.runner.models)
        self.assertIn("b_1", self.runner.models)

    def test_model_params_stored_correctly(self):
        params = _make_model_params()
        self.runner.add_model_params([params])
        stored = self.runner.models["test_model_0"]
        self.assertEqual(stored.name, "test_model")
        self.assertEqual(stored.model_type, "AE")

    def test_loss_data_entry_initialised_per_model(self):
        self.runner.add_model_params([_make_model_params("m1"), _make_model_params("m2")])
        self.assertIn("m1_0", self.runner.losses)
        self.assertIn("m2_1", self.runner.losses)

    def test_successive_add_calls_use_independent_counters(self):
        # Each add_model_params call starts its internal counter from 0,
        # so both get suffix _0 here.
        self.runner.add_model_params([_make_model_params("a")])
        self.runner.add_model_params([_make_model_params("b")])
        self.assertIn("a_0", self.runner.models)
        self.assertIn("b_0", self.runner.models)


class TestRunnerAddLoaderParams(unittest.TestCase):
    def setUp(self):
        self.runner = Runner(verbose=False)

    def test_loader_params_stored(self):
        self.runner.add_loader_params({"ds": _make_dataset_params()})
        self.assertIn("ds", self.runner.loader_params)

    def test_adding_params_clears_loader_cache(self):
        self.runner._loader_cache["stale"] = object()
        self.runner.add_loader_params({"ds": _make_dataset_params()})
        self.assertEqual(self.runner._loader_cache, {})

    def test_multiple_datasets_all_stored(self):
        self.runner.add_loader_params(
            {"ds1": _make_dataset_params(name="ds1"), "ds2": _make_dataset_params(name="ds2")}
        )
        self.assertIn("ds1", self.runner.loader_params)
        self.assertIn("ds2", self.runner.loader_params)


class TestRunnerCreateLoader(unittest.TestCase):
    def setUp(self):
        self.runner = Runner(verbose=False)

    def test_unknown_loader_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.runner.create_loader("does_not_exist")

    def test_unknown_loader_type_raises_value_error(self):
        self.runner.add_loader_params({"ds": _make_dataset_params(loader="unsupported_type")})
        with self.assertRaises(ValueError):
            self.runner.create_loader("ds")

    def test_cached_loader_returned_on_second_call(self):
        sentinel = object()
        self.runner._loader_cache["ds"] = sentinel
        self.runner.loader_params["ds"] = _make_dataset_params()
        result = self.runner.create_loader("ds")
        self.assertIs(result, sentinel)


class TestRunnerCreateModel(unittest.TestCase):
    def setUp(self):
        self.runner = Runner(verbose=False)

    def test_unknown_model_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.runner.create_model("nonexistent_99")

    def test_unrecognised_model_type_raises_value_error(self):
        self.runner.add_model_params([_make_model_params(model_type="UNKNOWN")])
        with self.assertRaises(ValueError):
            self.runner.create_model("test_model_0")


class TestRunnerRunEncoderPreconditions(unittest.TestCase):
    def setUp(self):
        self.runner = Runner(verbose=False)
        self.runner.logger = MagicMock()
        self.runner.logger.set_study_name = MagicMock()

    def test_run_encoder_with_no_loaders_raises(self):
        self.runner.add_model_params([_make_model_params()])
        with self.assertRaises(ValueError):
            self.runner.run_encoder("study")

    def test_run_encoder_with_no_models_raises(self):
        self.runner.add_loader_params({"ds": _make_dataset_params()})
        with self.assertRaises(ValueError):
            self.runner.run_encoder("study")


if __name__ == "__main__":
    unittest.main()
