import unittest
from unittest.mock import patch, MagicMock

import os
import tempfile
from PIL import Image

from simulation_encoder.loader import UnlabeledImageDataset


class TestUnlabeledImageDataset(unittest.TestCase):
    def setUp(self):
        print("HERE")
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
        for filename in self.image_files:
            # Mock the creation of images
            Image.new = MagicMock(return_value=Image.new("L", (10, 10)))
            Image.new("L", (10, 10)).save(os.path.join(self.temp_dir.name, filename))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_image_groups_creates_groups(self):
        dataset = UnlabeledImageDataset(self.temp_dir.name)
        groups = dataset.groups
        self.assertEqual(len(groups), 3)

        for group in groups:
            self.assertIn("cancer", group)
            self.assertIn("healthy", group)
            self.assertIn("graph", group)

    def test_getitem_returns_correct_shape(self):
        dataset = UnlabeledImageDataset(self.temp_dir.name)
        tensor = dataset[0]

        self.assertEqual(tensor.shape, (3, 10, 10))

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

        _ = UnlabeledImageDataset(self.temp_dir.name, logger=mock_logger)

        missing_key = ("CH", "type1", 1, 1)
        missing_images = ["healthy"]
        expected_message = f"Missing images from {missing_key}: {missing_images}"
        mock_logger.warning.assert_called_once_with(expected_message)


if __name__ == "__main__":
    unittest.main()
