import unittest
from unittest.mock import patch, MagicMock

import os
import tempfile
from PIL import Image

from simulation_encoder.loader import PNGLoader


class TestPNGLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_files = [
            "CH_type1_1_1_cells_cancer.png",
            "CH_type1_1_1_cells_healthy.png",
            "CH_type1_1_1_graph.png",
            "CH_type1_1_2_cells_cancer.png",
            "CH_type1_1_2_cells_healthy.png",
            "CH_type1_1_2_graph.png",
            "C_type1_1_1_cells_cancer.png",
            "C_type1_1_1_cells_healthy.png",
            "C_type1_1_1_graph.png",
        ]
        self.keys = ["CH_type1", "C_type1"]
        for filename in self.image_files:
            # Mock the creation of images
            Image.new = MagicMock(return_value=Image.new("L", (10, 10)))
            Image.new("L", (10, 10)).save(os.path.join(self.temp_dir.name, filename))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_image_groups_creates_groups(self):
        dataset = PNGLoader(self.temp_dir.name, keys=self.keys)
        groups = dataset.groups
        self.assertEqual(len(groups), 3)

        for group in groups:
            self.assertIn("cancer", group)
            self.assertIn("healthy", group)
            self.assertIn("graph", group)

    def test_getitem_returns_correct_shape(self):
        dataset = PNGLoader(self.temp_dir.name, keys=self.keys)
        expected_shape = (3, 10, 10)
        actual_shape = (dataset.n_channels, *dataset.image_shape)
        self.assertEqual(actual_shape, expected_shape)

    def test_train_test_loaders_have_correct_lengths(self):
        test_split = 0.4
        dataset = PNGLoader(
            self.temp_dir.name,
            keys=self.keys,
            test_split=test_split,
            batch_size=1,
            random_seed=123,
        )
        train_loader = dataset.get_train_dataloader()
        test_loader = dataset.get_test_dataloader()

        expected_test_size = int(len(dataset) * test_split)
        expected_train_size = len(dataset) - expected_test_size

        self.assertEqual(len(train_loader), expected_train_size)
        self.assertEqual(len(test_loader), expected_test_size)

    def test_train_test_split_gives_disjoint_sets(self):
        dataset = PNGLoader(self.temp_dir.name, keys=self.keys, test_split=0.2, batch_size=1, random_seed=123)
        train_indices = dataset._train_indices
        test_indices = dataset._test_indices

        self.assertTrue(set(train_indices).isdisjoint(set(test_indices)))

    @patch("simulation_encoder.logger.ExperimentLogger")
    def test_missing_image_logging(self, mock_logger):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_files = [
            "CH_type1_1_1_cells_cancer.png",
            "CH_type1_1_1_graph.png",
            "CH_type1_1_2_cells_cancer.png",
            "CH_type1_1_2_cells_healthy.png",
            "CH_type1_1_2_graph.png",
        ]
        for filename in self.image_files:
            Image.new = MagicMock(return_value=Image.new("L", (10, 10)))
            Image.new("L", (10, 10)).save(os.path.join(self.temp_dir.name, filename))

        _ = PNGLoader(self.temp_dir.name, keys=["CH_type1"], logger=mock_logger)

        missing_key = ("CH", "type1", 1, 1)
        missing_images = ["healthy"]
        expected_message = f"Missing images from {missing_key}: {missing_images}"
        mock_logger.warning.assert_called_once_with(expected_message)


if __name__ == "__main__":
    unittest.main()
