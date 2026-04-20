import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from simulation_encoder.writer import Writer
from simulation_encoder.dataclass.loss_data import LossData


def _make_mock_dataset(name="ds1"):
    dataset = MagicMock()
    dataset.name = name
    dataset.channels = ["cancer"]
    dataset.keys = ["CH_typeA"]
    dataset.augmentation_manager.augmentations = []
    return dataset


def _make_mock_model(name="ae_small"):
    model = MagicMock()
    model.name = name
    model.params = {
        "latent_dim": 4,
        "image_loss_weight": 0.9,
        "timepoint_loss_weight": 0.1,
        "optimizer": {"type": "Adam", "lr": 0.001},
    }
    return model


def _make_losses():
    losses = LossData()
    losses.add_train_loss({"image": [0.5, 0.4], "weighted_loss": [0.9, 0.7]})
    losses.add_val_loss({"image": [0.45, 0.35], "weighted_loss": [0.85, 0.65]})
    losses.add_test_loss({"image": 0.3, "weighted_loss": 0.6})
    return losses


class TestWriterSetup(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_results_directory_created_on_init(self):
        subdir = os.path.join(self.temp_dir, "results")
        Writer(results_dir=subdir, experiment_key="exp")
        self.assertTrue(os.path.isdir(subdir))

    def test_experiment_subdirectory_created_on_init(self):
        Writer(results_dir=self.temp_dir, experiment_key="my_exp")
        self.assertTrue(os.path.isdir(os.path.join(self.temp_dir, "my_exp")))


class TestWriterEncoderResults(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.writer = Writer(results_dir=self.temp_dir, experiment_key="exp")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _result_path(self, model_name="model_0", dataset_name="ds1"):
        return os.path.join(self.temp_dir, "exp", model_name, dataset_name, "results.json")

    def test_creates_results_json(self):
        self.writer.write_encoder_results(
            "model_0", _make_mock_dataset(), _make_mock_model(), _make_losses()
        )
        self.assertTrue(os.path.exists(self._result_path()))

    def test_model_name_written_correctly(self):
        self.writer.write_encoder_results(
            "model_0", _make_mock_dataset(), _make_mock_model(), _make_losses()
        )
        with open(self._result_path()) as f:
            data = json.load(f)
        self.assertEqual(data["model"], "model_0")

    def test_architecture_name_written_correctly(self):
        self.writer.write_encoder_results(
            "model_0", _make_mock_dataset(), _make_mock_model("ae_small"), _make_losses()
        )
        with open(self._result_path()) as f:
            data = json.load(f)
        self.assertEqual(data["architecture"], "ae_small")

    def test_channels_written_correctly(self):
        self.writer.write_encoder_results(
            "model_0", _make_mock_dataset(), _make_mock_model(), _make_losses()
        )
        with open(self._result_path()) as f:
            data = json.load(f)
        self.assertEqual(data["channels"], ["cancer"])

    def test_data_keys_written_correctly(self):
        self.writer.write_encoder_results(
            "model_0", _make_mock_dataset(), _make_mock_model(), _make_losses()
        )
        with open(self._result_path()) as f:
            data = json.load(f)
        self.assertEqual(data["data_keys"], ["CH_typeA"])

    def test_all_loss_splits_present(self):
        self.writer.write_encoder_results(
            "model_0", _make_mock_dataset(), _make_mock_model(), _make_losses()
        )
        with open(self._result_path()) as f:
            data = json.load(f)
        self.assertIn("train", data["losses"])
        self.assertIn("val", data["losses"])
        self.assertIn("test", data["losses"])

    def test_train_loss_values_correct(self):
        self.writer.write_encoder_results(
            "model_0", _make_mock_dataset(), _make_mock_model(), _make_losses()
        )
        with open(self._result_path()) as f:
            data = json.load(f)
        self.assertEqual(data["losses"]["train"]["image"], [0.5, 0.4])

    def test_test_loss_values_correct(self):
        self.writer.write_encoder_results(
            "model_0", _make_mock_dataset(), _make_mock_model(), _make_losses()
        )
        with open(self._result_path()) as f:
            data = json.load(f)
        self.assertAlmostEqual(data["losses"]["test"]["image"], 0.3)

    def test_different_model_ids_create_separate_directories(self):
        dataset = _make_mock_dataset()
        model = _make_mock_model()
        losses = _make_losses()
        self.writer.write_encoder_results("model_0", dataset, model, losses)
        self.writer.write_encoder_results("model_1", dataset, model, losses)
        self.assertTrue(os.path.exists(self._result_path("model_0")))
        self.assertTrue(os.path.exists(self._result_path("model_1")))


class TestWriterTrainTestIndices(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.writer = Writer(results_dir=self.temp_dir, experiment_key="exp")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _indices_path(self, dataset_name="my_ds"):
        return os.path.join(self.temp_dir, "exp", f"{dataset_name}_indices.json")

    def test_creates_indices_json(self):
        self.writer.write_train_test_indices("my_ds", ([0, 1], [2], [3]))
        self.assertTrue(os.path.exists(self._indices_path()))

    def test_train_indices_written_correctly(self):
        self.writer.write_train_test_indices("my_ds", ([0, 1], [2], [3]))
        with open(self._indices_path()) as f:
            data = json.load(f)
        self.assertEqual(data["train"], [0, 1])

    def test_val_indices_written_correctly(self):
        self.writer.write_train_test_indices("my_ds", ([0, 1], [2], [3]))
        with open(self._indices_path()) as f:
            data = json.load(f)
        self.assertEqual(data["val"], [2])

    def test_test_indices_written_correctly(self):
        self.writer.write_train_test_indices("my_ds", ([0, 1], [2], [3]))
        with open(self._indices_path()) as f:
            data = json.load(f)
        self.assertEqual(data["test"], [3])

    def test_empty_splits_written(self):
        self.writer.write_train_test_indices("my_ds", ([], [], []))
        with open(self._indices_path()) as f:
            data = json.load(f)
        self.assertEqual(data["train"], [])
        self.assertEqual(data["val"], [])
        self.assertEqual(data["test"], [])


class TestWriterModelState(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.writer = Writer(results_dir=self.temp_dir, experiment_key="exp")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_creates_model_state_file(self):
        import torch

        model = MagicMock()
        model.state_dict.return_value = {"weight": torch.tensor([1.0, 2.0])}
        self.writer.write_model_state("model_0", "ds1", model)
        state_path = os.path.join(self.temp_dir, "exp", "model_0", "ds1", "model_state.pth")
        self.assertTrue(os.path.exists(state_path))

    def test_saved_state_dict_is_loadable(self):
        import torch

        model = MagicMock()
        state = {"layer.weight": torch.tensor([0.5, 1.5])}
        model.state_dict.return_value = state
        self.writer.write_model_state("model_0", "ds1", model)
        state_path = os.path.join(self.temp_dir, "exp", "model_0", "ds1", "model_state.pth")
        loaded = torch.load(state_path, weights_only=True, map_location="cpu")
        self.assertIn("layer.weight", loaded)


if __name__ == "__main__":
    unittest.main()
