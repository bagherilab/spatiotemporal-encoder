import unittest
from unittest.mock import patch, MagicMock

import os
import tempfile
from PIL import Image

import numpy as np

from simulation_encoder.loader import ARCADELoader, AlphaNumericLoader


class TestARCADELoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_files = [
            "CH_typeA_1_1_cells_cancer.png",
            "CH_typeA_1_1_cells_healthy.png",
            "CH_typeA_1_1_graph.png",
            "CH_typeA_1_2_cells_cancer.png",
            "CH_typeA_1_2_cells_healthy.png",
            "CH_typeA_1_2_graph.png",
            "CH_typeAB_1_2_cells_healthy.png",
            "CH_typeAB_1_2_graph.png",
            "C_typeA_1_1_cells_cancer.png",
            "C_typeA_1_1_cells_healthy.png",
            "C_typeA_1_1_graph.png",
        ]
        self.keys = ["CH_typeA", "CH_typeAB", "C_typeA"]
        self.labels = ["test_label"]
        for filename in self.image_files:
            Image.new = MagicMock(return_value=Image.new("L", (10, 10)))
            Image.new("L", (10, 10)).save(os.path.join(self.temp_dir.name, filename))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_image_groups_creates_groups(self):
        dataset = ARCADELoader(self.temp_dir.name, keys=self.keys, test_split=0.5)
        groups = dataset.groups
        self.assertEqual(len(groups), 4)

        for group in groups:
            self.assertIn("cancer", group)
            self.assertIn("graph", group)

    def test_getitem_returns_correct_shape(self):
        dataset = ARCADELoader(self.temp_dir.name, keys=self.keys)
        expected_shape = (1, 10, 10)
        actual_shape = (dataset.n_channels, *dataset.image_shape)
        self.assertEqual(actual_shape, expected_shape)

    # def test_train_test_loaders_have_correct_lengths(self):
    #     test_split = 0.34
    #     val_split = 0.34
    #     dataset = PNGLoader(
    #         self.temp_dir.name,
    #         keys=self.keys,
    #         test_split=test_split,
    #         val_split=val_split,
    #         batch_size=1,
    #         random_seed=123,
    #     )
    #     train_loader = dataset.get_train_dataloader()
    #     val_loader = dataset.get_val_dataloader()
    #     test_loader = dataset.get_test_dataloader()

    #     expected_val_size = 1
    #     expected_test_size = 1
    #     expected_train_size = 2

    #     self.assertEqual(len(train_loader), expected_train_size)
    #     self.assertEqual(len(val_loader), expected_val_size)
    #     self.assertEqual(len(test_loader), expected_test_size)

    def test_train_test_split_gives_disjoint_sets(self):
        dataset = ARCADELoader(
            self.temp_dir.name, keys=self.keys, test_split=0.2, batch_size=1, random_seed=123
        )
        train_indices = dataset._train_indices
        val_indices = dataset._val_indices
        test_indices = dataset._test_indices

        self.assertTrue(set(train_indices).isdisjoint(set(test_indices)))
        self.assertTrue(set(train_indices).isdisjoint(set(val_indices)))
        self.assertTrue(set(val_indices).isdisjoint(set(test_indices)))

    def test_keys_load_correct_data(self):
        dataset = ARCADELoader(self.temp_dir.name, keys=["CH_typeA"])
        self.assertEqual(len(dataset), 2)

    def test_file_parsing_returns_correct_chunks(self):
        dataset = ARCADELoader(self.temp_dir.name, keys=self.keys)

        expected_keys = [
            ("CH", "typeA", 1, 1, "cancer"),
            ("CH", "typeA", 1, 1, "healthy"),
            ("CH", "typeA", 1, 1, "graph"),
            ("CH", "typeA", 1, 2, "cancer"),
            ("CH", "typeA", 1, 2, "healthy"),
            ("CH", "typeA", 1, 2, "graph"),
            ("CH", "typeAB", 1, 2, "healthy"),
            ("CH", "typeAB", 1, 2, "graph"),
            ("C", "typeA", 1, 1, "cancer"),
            ("C", "typeA", 1, 1, "healthy"),
            ("C", "typeA", 1, 1, "graph"),
        ]

        actual_keys = []
        for file_name in self.image_files:
            file_name = os.path.basename(file_name)
            actual_keys.append(dataset._parse_ARCADE_filename(file_name))

        self.assertEqual(actual_keys, expected_keys)

    @patch("simulation_encoder.logger.ExperimentLogger")
    def test_missing_image_logging(self, mock_logger):
        _ = ARCADELoader(self.temp_dir.name, keys=["CH_typeAB"], logger=mock_logger)

        missing_key = ("CH", "typeAB", 1, 2)
        missing_key = "CH_typeAB_1_2"
        missing_images = ["cancer"]
        expected_message = f"Missing images from {missing_key}: {missing_images}"
        mock_logger.warning.assert_called_once_with(expected_message)


if __name__ == "__main__":
    unittest.main()
