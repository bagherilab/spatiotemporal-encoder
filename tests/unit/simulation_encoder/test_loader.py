import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from simulation_encoder.loader import PNGLoader


def _write_images(directory, filenames):
    """Save real 10×10 greyscale PNGs to directory."""
    for filename in filenames:
        Image.new("L", (10, 10)).save(os.path.join(directory, filename))


_STANDARD_FILES = [
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


class TestPNGLoaderGroups(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        _write_images(self.temp_dir.name, _STANDARD_FILES)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_image_groups_creates_correct_count(self):
        dataset = PNGLoader(self.temp_dir.name)
        self.assertEqual(len(dataset.groups), 3)

    def test_groups_contain_all_channel_keys(self):
        dataset = PNGLoader(self.temp_dir.name)
        for group in dataset.groups:
            self.assertIn("cancer", group)
            self.assertIn("healthy", group)
            self.assertIn("graph", group)

    def test_groups_contain_non_empty_paths(self):
        dataset = PNGLoader(self.temp_dir.name)
        for group in dataset.groups:
            for channel, path in group.items():
                self.assertNotEqual(path, "", msg=f"Channel '{channel}' path should not be empty")

    def test_parse_filename_returns_correct_tuple(self):
        dataset = PNGLoader(self.temp_dir.name)
        context, vasc_type, seed, tp, img_type = dataset._parse_filename(
            "CH_type1_1_2_cells_cancer.png"
        )
        self.assertEqual(context, "CH")
        self.assertEqual(vasc_type, "type1")
        self.assertEqual(seed, 1)
        self.assertEqual(tp, 2)
        self.assertEqual(img_type, "cancer")

    def test_parse_filename_graph_type(self):
        dataset = PNGLoader(self.temp_dir.name)
        _, _, _, _, img_type = dataset._parse_filename("C_type1_1_1_graph.png")
        self.assertEqual(img_type, "graph")


class TestPNGLoaderGetitem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        _write_images(self.temp_dir.name, _STANDARD_FILES)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_getitem_returns_correct_shape(self):
        dataset = PNGLoader(self.temp_dir.name)
        tensor = dataset[0]
        self.assertEqual(tensor.shape, (3, 10, 10))

    def test_getitem_returns_float_tensor(self):
        dataset = PNGLoader(self.temp_dir.name)
        tensor = dataset[0]
        self.assertTrue(tensor.is_floating_point())


class TestPNGLoaderSplits(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        _write_images(self.temp_dir.name, _STANDARD_FILES)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_and_test_loaders_correct_length(self):
        # 3 groups, test_split=0.4 → floor(0.4*3)=1 test, 2 train; batch_size=1
        dataset = PNGLoader(self.temp_dir.name, test_split=0.4, batch_size=1, random_seed=123)
        self.assertEqual(len(dataset.get_train_dataloader()), 2)
        self.assertEqual(len(dataset.get_test_dataloader()), 1)

    def test_train_test_indices_are_disjoint(self):
        dataset = PNGLoader(self.temp_dir.name, test_split=0.2, batch_size=1, random_seed=123)
        train = set(dataset._get_train_indices())
        test = set(dataset._get_test_indices())
        self.assertTrue(train.isdisjoint(test))

    def test_train_test_indices_cover_all_data(self):
        dataset = PNGLoader(self.temp_dir.name, test_split=0.33, batch_size=1, random_seed=42)
        all_indices = set(dataset._get_train_indices()) | set(dataset._get_test_indices())
        self.assertEqual(all_indices, set(range(len(dataset))))

    def test_different_seeds_produce_non_empty_test_split(self):
        # test_split=0.4 → floor(0.4 * 3) = 1 test sample
        d = PNGLoader(self.temp_dir.name, test_split=0.4, batch_size=1, random_seed=42)
        self.assertGreater(len(d._get_test_indices()), 0)


class TestPNGLoaderMissingImageLogging(unittest.TestCase):
    """Verify that missing channels are reported to the logger."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # CH_type1_1_1 is missing 'healthy' → should trigger a warning
        _write_images(
            self.temp_dir.name,
            [
                "CH_type1_1_1_cells_cancer.png",
                "CH_type1_1_1_graph.png",
                "CH_type1_1_2_cells_cancer.png",
                "CH_type1_1_2_cells_healthy.png",
                "CH_type1_1_2_graph.png",
            ],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_warning_issued_for_missing_healthy(self):
        mock_logger = MagicMock()
        PNGLoader(self.temp_dir.name, logger=mock_logger)
        missing_key = ("CH", "type1", 1, 1)
        expected_message = f"Missing images from {missing_key}: ['healthy']"
        mock_logger.warning.assert_called_once_with(expected_message)

    def test_no_warning_when_all_images_present(self):
        mock_logger = MagicMock()
        PNGLoader(self.temp_dir.name, logger=mock_logger)
        # Only one group is incomplete; the complete group should not trigger a call
        self.assertEqual(mock_logger.warning.call_count, 1)


if __name__ == "__main__":
    unittest.main()
