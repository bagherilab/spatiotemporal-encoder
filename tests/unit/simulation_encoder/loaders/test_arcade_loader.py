import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from simulation_encoder.loaders.arcade_loader import ARCADELoader


def _write_images(directory, filenames):
    """Save real 10×10 greyscale PNGs to directory."""
    for filename in filenames:
        Image.new("L", (10, 10)).save(os.path.join(directory, filename))


_IMAGE_FILES = [
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
    "C_typeA_1_2_cells_cancer.png",
]

_ALL_KEYS = ["CH_typeA", "CH_typeAB", "C_typeA"]
_IMAGE_SIZE = 10


class TestARCADELoaderGroups(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        _write_images(self.temp_dir.name, _IMAGE_FILES)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_retrieve_data_creates_correct_group_count(self):
        loader = ARCADELoader(
            self.temp_dir.name,
            channels=["cancer", "graph"],
            keys=_ALL_KEYS,
            image_size=_IMAGE_SIZE,
            test_split=0.5,
        )
        self.assertEqual(len(loader.data), 5)

    def test_groups_contain_requested_channels(self):
        loader = ARCADELoader(
            self.temp_dir.name,
            channels=["cancer", "graph"],
            keys=_ALL_KEYS,
            image_size=_IMAGE_SIZE,
            test_split=0.5,
        )
        for group in loader.data:
            self.assertIn("cancer", group)
            self.assertIn("graph", group)

    def test_keys_filter_groups_correctly(self):
        loader = ARCADELoader(
            self.temp_dir.name,
            channels=["cancer"],
            keys=["CH_typeA"],
            image_size=_IMAGE_SIZE,
        )
        self.assertEqual(len(loader.data), 2)

    def test_file_parsing_returns_correct_tuples(self):
        dataset = ARCADELoader(
            self.temp_dir.name,
            channels=["cancer", "healthy", "graph"],
            keys=_ALL_KEYS,
            image_size=_IMAGE_SIZE,
        )
        expected = [
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
            ("C", "typeA", 1, 2, "cancer"),
        ]
        actual = [dataset._parse_filename(os.path.basename(f)) for f in _IMAGE_FILES]
        self.assertEqual(actual, expected)


class TestARCADELoaderSplits(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        _write_images(self.temp_dir.name, _IMAGE_FILES)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_val_test_indices_are_pairwise_disjoint(self):
        dataset = ARCADELoader(
            self.temp_dir.name,
            channels=["cancer"],
            keys=_ALL_KEYS,
            image_size=_IMAGE_SIZE,
            test_split=0.2,
            batch_size=1,
            random_seed=123,
        )
        train = set(dataset._train_indices)
        val = set(dataset._val_indices)
        test = set(dataset._test_indices)
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(train.isdisjoint(val))
        self.assertTrue(val.isdisjoint(test))

    def test_dataloaders_have_expected_lengths(self):
        dataset = ARCADELoader(
            self.temp_dir.name,
            channels=["cancer"],
            keys=_ALL_KEYS,
            image_size=_IMAGE_SIZE,
            val_split=0.0,
            test_split=0.5,
            batch_size=1,
            random_seed=123,
        )
        train_loader = dataset.get_dataloader("train")
        val_loader = dataset.get_dataloader("val")
        test_loader = dataset.get_dataloader("test")

        self.assertEqual(len(val_loader), 0)
        self.assertEqual(len(train_loader), 2)
        self.assertEqual(len(test_loader), 2)


class TestARCADELoaderLogging(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        _write_images(self.temp_dir.name, _IMAGE_FILES)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("simulation_encoder.logger.Logger")
    def test_missing_channel_triggers_warning(self, mock_logger):
        # CH_typeAB has healthy and graph but no cancer → 1 missing cancer
        ARCADELoader(
            self.temp_dir.name,
            channels=["cancer", "graph"],
            keys=["CH_typeAB"],
            image_size=_IMAGE_SIZE,
            logger=mock_logger,
        )
        expected_message = "Number of missing images in cancer - 1"
        mock_logger.warning.assert_called_once_with(expected_message)


if __name__ == "__main__":
    unittest.main()
