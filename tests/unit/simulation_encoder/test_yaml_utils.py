import os
import tempfile
import unittest

import yaml
from pydantic import ValidationError

from simulation_encoder.dataclass.config_schemas import (
    HyperparameterConfig,
    MainConfig,
    ModelArchitectureConfig,
)
from simulation_encoder.utils.yaml_utils import (
    load_hyperparam_yaml,
    load_model_yaml,
    load_yaml,
)


class TestLoadYaml(unittest.TestCase):
    """Tests for the generic load_yaml helper."""

    def _write_temp_yaml(self, data: dict) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(data, f)
        f.close()
        return f.name

    def test_valid_main_config_loaded(self):
        path = self._write_temp_yaml(
            {"study_name": "test", "data_quantity_experiment": False, "debug": False}
        )
        try:
            cfg = load_yaml(path, MainConfig)
            self.assertEqual(cfg.study_name, "test")
            self.assertFalse(cfg.debug)
        finally:
            os.unlink(path)

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_yaml("/nonexistent/path/does_not_exist.yaml", MainConfig)

    def test_invalid_yaml_data_raises_validation_error(self):
        # Missing required fields for MainConfig
        path = self._write_temp_yaml({"study_name": "test"})
        try:
            with self.assertRaises((ValidationError, Exception)):
                load_yaml(path, MainConfig)
        finally:
            os.unlink(path)


class TestLoadHyperparamYaml(unittest.TestCase):
    """Tests that load_hyperparam_yaml correctly reads the real grid_optimizers.yaml."""

    def test_returns_hyperparam_config_instance(self):
        cfg = load_hyperparam_yaml("grid_optimizers")
        self.assertIsInstance(cfg, HyperparameterConfig)

    def test_num_epochs_positive(self):
        cfg = load_hyperparam_yaml("grid_optimizers")
        self.assertGreater(cfg.num_epochs, 0)

    def test_continuous_params_present(self):
        cfg = load_hyperparam_yaml("grid_optimizers")
        self.assertIsNotNone(cfg.continuous)
        self.assertIn("image_loss_weight", cfg.continuous)

    def test_discrete_params_present(self):
        cfg = load_hyperparam_yaml("grid_optimizers")
        self.assertIsNotNone(cfg.discrete)
        self.assertIn("latent_dim", cfg.discrete)

    def test_missing_hyperparam_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_hyperparam_yaml("nonexistent_hyperparam_xyz")


class TestLoadModelYaml(unittest.TestCase):
    """Tests that load_model_yaml correctly reads real model yaml files."""

    def test_ae_small_returns_model_architecture_config(self):
        cfg = load_model_yaml("ae_small")
        self.assertIsInstance(cfg, ModelArchitectureConfig)

    def test_ae_small_type_is_ae(self):
        cfg = load_model_yaml("ae_small")
        self.assertEqual(cfg.type, "AE")

    def test_ae_small_has_decoder_image(self):
        cfg = load_model_yaml("ae_small")
        self.assertIsNotNone(cfg.architecture.decoder_image)
        self.assertGreater(len(cfg.architecture.decoder_image), 0)

    def test_ae_small_has_decoder_timepoint(self):
        cfg = load_model_yaml("ae_small")
        self.assertIsNotNone(cfg.architecture.decoder_timepoint)
        self.assertGreater(len(cfg.architecture.decoder_timepoint), 0)

    def test_ae_small_has_encoder(self):
        cfg = load_model_yaml("ae_small")
        self.assertIsNotNone(cfg.architecture.encoder)
        self.assertGreater(len(cfg.architecture.encoder), 0)

    def test_vit_small_loads_successfully(self):
        cfg = load_model_yaml("vit_small")
        self.assertIsInstance(cfg, ModelArchitectureConfig)
        self.assertEqual(cfg.type, "AE")

    def test_flat_cnn_uses_legacy_format(self):
        # flat_cnn.yaml uses the legacy CAE architecture format
        # (encoder_layers/decoder_layers) which does not conform to
        # ModelArchitectureConfig. Loading it should raise an exception.
        with self.assertRaises(Exception):
            load_model_yaml("flat_cnn")

    def test_missing_model_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_model_yaml("nonexistent_model_xyz")

    def test_yaml_extension_optional(self):
        """load_model_yaml should append .yaml automatically."""
        cfg_no_ext = load_model_yaml("ae_small")
        cfg_with_ext = load_model_yaml("ae_small.yaml")
        self.assertEqual(cfg_no_ext.type, cfg_with_ext.type)


if __name__ == "__main__":
    unittest.main()
